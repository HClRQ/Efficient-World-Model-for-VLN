"""Precompute NON-POOLED SigLIP patch-grid features for TSSM training.

Replaces the legacy pooled cache (last_hidden_state.mean(dim=1) -> [T, 1152])
with a spatial grid: 729 patches (27x27) block-averaged into a GxG grid
(default 3x3) -> [T, G*G, 1152].  LS-NWM-style non-pooled observation.

Preprocessing matches the legacy cache exactly (verified by
diag_feat_mismatch.py on 2026-07-05): resize (384,384) BILINEAR, /255,
normalize (x-0.5)/0.5, NaVILA vision tower SiglipVisionModel, fp32.

Built-in provenance check: the mean of the G*G block means equals the global
patch mean by construction, so per-frame cos(grid.mean(0), legacy_cache[t])
must be ~1.0.  A low cos means preprocessing drift -> abort loudly.

Output: <out_dir>/features.pt  dict{episode_id: FloatTensor[T, G*G, 1152] fp16}

Usage:
    python RSSM_v4/precompute_features_patch.py \
        --annotation /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json \
        --image_root /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/train.sqlite \
        --verify_against /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/siglip/features.pt \
        --out_dir /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/siglip_p9
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_navila import build_episode_trajectories, _open_image, _sqlite_frame_path, _is_sqlite_store  # noqa: E402


class FlatFrameDataset(Dataset):
    """All frames of all episodes, flattened, with SigLIP preprocessing."""

    def __init__(self, episodes, image_root: str, size: int = 384):
        self.items = []  # (ep_idx, t, frame_path)
        for i, ep in enumerate(episodes):
            for t, fp in enumerate(ep.frame_paths):
                self.items.append((i, t, fp))
        self.image_root = image_root
        self.size = size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, t, fp = self.items[k]
        if _is_sqlite_store(self.image_root):
            path = _sqlite_frame_path(self.image_root, fp)
        else:
            path = os.path.join(self.image_root, fp)
        img = _open_image(path).resize((self.size, self.size), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(img).copy()).float().permute(2, 0, 1) / 255.0
        arr = (arr - 0.5) / 0.5
        return i, t, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--image_root", required=True, help="train.sqlite or frame dir")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--siglip", default="/cluster/project/rsl/ruiqzhang/codes/NaVILA/navila_pretrain/vision_tower")
    ap.add_argument("--layer", type=int, default=0,
                    help="0 = last_hidden_state (final layer + post_layernorm; legacy TSSM space). "
                         "Negative = raw hidden_states[idx] WITHOUT post_layernorm; -2 matches the "
                         "VLM tower (vision_encoder.py mm_vision_select_layer=-2, cls_patch).")
    ap.add_argument("--grid", type=int, default=3, help="GxG output grid (729=27x27 must divide)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--verify_against", default=None, help="legacy pooled features.pt for provenance check")
    ap.add_argument("--verify_min_cos", type=float, default=0.995)
    ap.add_argument("--save_every_eps", type=int, default=2000, help="partial save cadence (completed episodes)")
    ap.add_argument("--shard", type=int, default=0, help="shard index (episodes[shard::num_shards])")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--pooled", action="store_true",
                    help="输出legacy池化口径: last_hidden_state.mean(dim=1) -> [T,1152]")
    args = ap.parse_args()

    assert 27 % args.grid == 0, f"grid {args.grid} must divide 27"
    blk = 27 // args.grid
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from transformers import SiglipVisionModel
    print(f"[setup] loading SigLIP from {args.siglip} (fp32, {device})", flush=True)
    sig = SiglipVisionModel.from_pretrained(args.siglip, torch_dtype=torch.float32).to(device).eval()

    episodes = build_episode_trajectories(args.annotation)
    if args.num_shards > 1:
        episodes = episodes[args.shard::args.num_shards]
        print(f"[shard] {args.shard}/{args.num_shards}: {len(episodes)} episodes", flush=True)
    n_frames_total = sum(len(e.frame_paths) for e in episodes)
    print(f"[setup] {len(episodes)} episodes, {n_frames_total} frames, "
          f"grid {args.grid}x{args.grid} (block {blk}x{blk})", flush=True)

    legacy = None
    if args.verify_against:
        print(f"[setup] loading legacy cache for verification: {args.verify_against}", flush=True)
        legacy = torch.load(args.verify_against, map_location="cpu", weights_only=True)

    ds = FlatFrameDataset(episodes, args.image_root)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    partial_path = os.path.join(args.out_dir, "features.pt.partial")
    final_path = os.path.join(args.out_dir, "features.pt")

    # Preallocate per-episode buffers, fill as batches stream in.
    bufs = {i: torch.empty(len(e.frame_paths), args.grid * args.grid, 1152, dtype=torch.float16)
            for i, e in enumerate(episodes)}
    filled = {i: 0 for i in bufs}
    done: dict[str, torch.Tensor] = {}
    cos_sum, cos_min, cos_n = 0.0, 1.0, 0
    worst = ("", -1, 1.0)
    t0 = time.time()
    seen = 0
    last_partial = 0

    max_abs = 0.0
    with torch.no_grad():
        for bi, (ep_idx, ts, pix) in enumerate(loader):
            pix = pix.to(device, non_blocking=True)
            if args.layer == 0:
                out = sig(pixel_values=pix).last_hidden_state                 # [B,729,1152]
            else:
                out = sig(pixel_values=pix, output_hidden_states=True
                          ).hidden_states[args.layer]                         # raw, no post_ln
            B = out.shape[0]
            if args.pooled:
                grid = out.mean(dim=1, keepdim=True)        # [B,1,1152] legacy池化
            else:
                grid = (out.view(B, 27, 27, 1152)
                        .view(B, args.grid, blk, args.grid, blk, 1152)
                        .mean(dim=(2, 4))                    # [B, G, G, 1152]
                        .reshape(B, args.grid * args.grid, 1152))

            if legacy is not None:
                pooled = grid.mean(dim=1)                   # == mean over all 729 patches
                refs = torch.stack([
                    legacy[episodes[int(ep_idx[j])].episode_id][int(ts[j])]
                    for j in range(B)
                ]).float().to(device, non_blocking=True)
                cos = F.cosine_similarity(pooled, refs, dim=1)          # [B]
                bmin, bargmin = cos.min(dim=0)
                bmin = bmin.item()
                cos_sum += cos.sum().item(); cos_n += B
                if bmin < cos_min:
                    j = int(bargmin)
                    cos_min = bmin
                    worst = (episodes[int(ep_idx[j])].episode_id, int(ts[j]), bmin)
                if args.verify_min_cos > 0 and bmin < args.verify_min_cos:
                    j = int(bargmin)
                    raise RuntimeError(
                        f"PROVENANCE CHECK FAILED: ep {episodes[int(ep_idx[j])].episode_id} "
                        f"t={int(ts[j])} cos={bmin:.4f} < {args.verify_min_cos} — preprocessing drift?")

            ba = grid.abs().max().item()
            if ba > max_abs:
                max_abs = ba
                if max_abs > 60000:
                    raise RuntimeError(f"feature magnitude {max_abs:.0f} overflows fp16 storage")
            grid = grid.to(torch.float16).cpu()
            for j in range(B):
                i = int(ep_idx[j])
                bufs[i][int(ts[j])] = grid[j]
                filled[i] += 1
                if filled[i] == bufs[i].shape[0]:
                    done[episodes[i].episode_id] = bufs.pop(i)

            seen += B
            if bi % 50 == 0:
                el = time.time() - t0
                fps = seen / max(el, 1e-9)
                eta = (n_frames_total - seen) / max(fps, 1e-9) / 60
                msg = (f"[{seen}/{n_frames_total}] {fps:.0f} f/s eta {eta:.0f} min "
                       f"done_eps={len(done)}")
                if cos_n:
                    msg += f" | verify cos mean={cos_sum/cos_n:.5f} min={cos_min:.5f}"
                print(msg, flush=True)

            if len(done) - last_partial >= args.save_every_eps:
                torch.save(done, partial_path)
                last_partial = len(done)
                print(f"[partial] saved {len(done)} episodes -> {partial_path}", flush=True)

    assert not bufs, f"{len(bufs)} episodes incomplete — dataloader ordering bug?"
    torch.save(done, final_path)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    el = time.time() - t0
    print(f"[done] {len(done)} episodes, {seen} frames in {el/60:.1f} min -> {final_path}", flush=True)
    if cos_n:
        print(f"[verify] provenance vs legacy cache: mean cos={cos_sum/cos_n:.5f} "
              f"min cos={cos_min:.5f} (worst ep {worst[0]} t={worst[1]})", flush=True)
    sz = os.path.getsize(final_path) / 1e9
    print(f"[done] file size {sz:.2f} GB | layer={args.layer} siglip={args.siglip} "
          f"max_abs={max_abs:.1f}", flush=True)


if __name__ == "__main__":
    main()

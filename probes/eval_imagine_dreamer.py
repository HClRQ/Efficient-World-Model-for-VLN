#!/usr/bin/env python3
"""
TransDreamer-style world-model eval — action-conditioned open-loop prediction.

协议 (跟 PlaNet / DreamerV1-V4 一致):
  1) Burn-in:  用真实 obs feat_0..feat_{K-1} 的 posterior 路径推 state, 拿到 state_{K-1}
  2) Open-loop: 从 state_{K-1} 起, 只喂 GT actions a_{K-1}..a_{K+N-2}, 不再喂 obs.
                走 prior 路径 (model.imagine_step), 得到 state_K..state_{K+N-1}.
  3) 解码:     ô_t = decoder(state_t), 对比 GT feat_t, t=K..K+N-1.
  4) 对比:     与 GT feat_t 比 cos / MSE; 反查 NN 看落到哪个真帧
  5) 视觉:     每个数据集采 4 episode, 画 GT vs imagined 的帧带 PNG

数值默认 K=5 burn-in + N=25 open-loop predict, 报 horizon=1,5,10,15,20,25.
也可用 --burn_ins 5 10 15 做 context-length sweep，类似 TransDreamer
paper 里比较不同 context length 的 world-model generation quality。
视觉默认每个数据集 4 ep, 每 ep 画 8 个时间点 (h=0,3,6,9,12,15,20,25)

为什么这能间接证明 "h 编码了历史":
  open-loop 时 prior 拿不到 obs, 只能靠 state 里塞的过去信息 + 当前 action 去外推.
  如果 N=25 步后还能保持视觉合理, state_{K-1} 必然携带了大量过去 (场景/位置/物体).
  Drift 越慢, h 编码越深.

Usage:
  python eval_imagine_dreamer.py \\
      --ckpt /cluster/scratch/ruiqzhang/tssm_mix.pt \\
      --annotation .../R2R/annotations.json \\
      --cache_path .../R2R/feat/siglip \\
      --image_root .../R2R/train \\
      --out_png  logs/dreamer_eval_mix_r2r.png \\
      --burn_in 5 --predict_n 25
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_navila import (                                  # noqa: E402
    NaVILAR2RDataset, build_episode_trajectories, split_episodes,
    ACTION_VOCAB, _is_sqlite_store, _open_image,
)

ACTION_ID_TO_NAME = {v: k for k, v in ACTION_VOCAB.items()}
ACTION_ID_TO_SHORT = {
    0: "stop",
    1: "fwd25",
    2: "fwd50",
    3: "fwd75",
    4: "left15",
    5: "left30",
    6: "left45",
    7: "right15",
    8: "right30",
    9: "right45",
}


# --------------------------------------------------------------------------- #
def load_model(ckpt_path: str, device: torch.device):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = blob["config"]
    if not ("d_model" in cfg_dict or "n_heads" in cfg_dict):
        raise RuntimeError("eval_imagine_dreamer is TSSM-only")
    from navigation_tssm_feat import NavigationTSSM, TSSMConfig
    valid = {k: v for k, v in cfg_dict.items()
             if k in TSSMConfig.__dataclass_fields__}
    cfg = TSSMConfig(**valid)
    model = NavigationTSSM(cfg).to(device).eval()
    model.load_state_dict(blob["model"])
    print(f"[load] {ckpt_path}  step={blob.get('step','?')}  "
          f"feat_dim={cfg.feat_dim}  state_dim={cfg.h_dim+cfg.stoch*cfg.classes}")
    return model, cfg


@torch.no_grad()
def build_feat_pool(val_eps, ds: NaVILAR2RDataset,
                    max_pool: int = 16000) -> Tuple[torch.Tensor, list]:
    """Default pool: only val episodes of the CURRENT dataset.
    meta entries are (ep_id, step). Used when --cross_pool is off."""
    feats, meta = [], []
    for ep in val_eps:
        f = ds.features[ep.episode_id]
        for t in range(f.shape[0]):
            feats.append(f[t])
            meta.append((ep.episode_id, t))
            if len(feats) >= max_pool:
                break
        if len(feats) >= max_pool:
            break
    return torch.stack(feats).float(), meta


@torch.no_grad()
def build_cross_pool(feat_paths: List[Tuple[str, str]],
                     max_pool: int = 200000,
                     seed: int = 0,
                     force_include_eps: Optional[set] = None) -> Tuple[torch.Tensor, list]:
    """All-MP3D frame pool: pulls every (ep, step) feat from R2R + RxR
    (both use MP3D scenes), then optionally sub-samples for tractability.

    To avoid silently dropping the GT frames of test episodes (which
    would make NN@1 unhittable), `force_include_eps` is a set of
    "{tag}:{ep_id}" strings whose frames are ALWAYS kept regardless of
    sub-sampling.

    feat_paths : list of (dataset_tag, features.pt path) tuples.
                 Tag is internal-only (used to disambiguate ep_id
                 collisions between R2R and RxR), NOT shown in PNG labels.

    meta entries: (dataset_tag, ep_id, step) — caller maps tag→image_root
    for the visual NN retrieval, but doesn't show the tag in tile labels.
    """
    rng = np.random.default_rng(seed)
    must_keep, optional = [], []
    force_include_eps = force_include_eps or set()
    for tag, path in feat_paths:
        if not os.path.exists(path):
            print(f"[cross_pool] WARN: {path} missing, skipping {tag}")
            continue
        d = torch.load(path, map_location="cpu", weights_only=True)
        n_must, n_opt = 0, 0
        for ep_id, f in d.items():
            keep = f"{tag}:{ep_id}" in force_include_eps
            for t in range(f.shape[0]):
                entry = (tag, ep_id, t, f[t])
                if keep:
                    must_keep.append(entry); n_must += 1
                else:
                    optional.append(entry); n_opt += 1
        print(f"[cross_pool] loaded {n_must} must-keep + {n_opt} optional "
              f"frames from {tag} ({len(d)} episodes)")

    # sub-sample only the OPTIONAL pool (always keep all val/test frames)
    budget = max(0, max_pool - len(must_keep))
    if len(optional) > budget:
        idx = rng.choice(len(optional), budget, replace=False)
        optional = [optional[i] for i in idx]
        print(f"[cross_pool] sub-sampled optional to {budget} (must-keep "
              f"{len(must_keep)} preserved)")
    all_entries = must_keep + optional
    feats = torch.stack([e[3] for e in all_entries]).float()
    meta = [(e[0], e[1], e[2]) for e in all_entries]
    print(f"[cross_pool] final pool: {len(meta)} frames "
          f"(GT inclusion guaranteed for {len(must_keep)} val frames)")
    return feats, meta


# --------------------------------------------------------------------------- #
@torch.no_grad()
def burn_in_then_imagine(model, feat_seq: torch.Tensor,
                         action_seq: torch.Tensor, K: int, N: int,
                         device: torch.device) -> List[torch.Tensor]:
    """Burn-in K frames with posterior, imagine N more with prior only.
    Returns predicted features for GT steps K..K+N-1.

    Time alignment matches training/cache:
      state_t = filter(state_{t-1}, action_{t-1}, obs_t)
      state_{t+1} = imagine(state_t, action_t)
    """
    feat_seq   = feat_seq.float().to(device)
    action_seq = action_seq.long().to(device)

    # 1) burn-in posterior path: feeds GT obs
    state = model.get_initial_state(1, device, dtype=torch.float32)
    for t in range(K):
        a_prev = (torch.zeros(1, dtype=torch.long, device=device)
                  if t == 0 else action_seq[t - 1:t])
        state = model.inference_step(state, a_prev, feat_seq[t:t + 1])

    # 2) open-loop imagination: ONLY actions, NO more obs
    preds = []
    for h in range(1, N + 1):
        a = action_seq[K + h - 2:K + h - 1]
        state = model.imagine_step(state, a)
        pred = model.feat_decoder(state.full).squeeze(0)              # [D]
        preds.append(pred)
    return preds


def make_rollout_actions(action_seq: torch.Tensor,
                         K: int,
                         N: int,
                         num_actions: int,
                         mode: str,
                         seed: int,
                         episode_id,
                         include_stop: bool = False,
                         fixed_action_id: int = 1) -> torch.Tensor:
    """Return actions used by eval.

    Keep burn-in actions intact and replace only the open-loop controls
    consumed by imagine_step: a_{K-1}..a_{K+N-2}.
    """
    if mode == "gt":
        return action_seq
    if mode not in {"random", "fixed"}:
        raise ValueError(f"unknown action mode: {mode}")
    if not 0 <= fixed_action_id < num_actions:
        raise ValueError(f"fixed_action_id={fixed_action_id} outside "
                         f"[0,{num_actions})")

    out = action_seq.clone()
    if mode == "fixed":
        out[K - 1:K + N - 1] = fixed_action_id
        return out

    if num_actions <= 1:
        raise ValueError("random action mode requires num_actions > 1")
    low = 0 if include_stop else 1
    key = f"{seed}:{episode_id}:{K}:{N}:{num_actions}:{include_stop}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    local_seed = int.from_bytes(digest, "little") & ((1 << 63) - 1)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(local_seed)
    out[K - 1:K + N - 1] = torch.randint(
        low, num_actions, (N,), generator=gen, dtype=torch.long
    )
    return out


def action_label(actions, idx: int) -> str:
    if actions is None or idx < 0 or idx >= len(actions):
        return "?"
    value = actions[idx]
    if torch.is_tensor(value):
        value = value.item()
    action_id = int(value)
    return ACTION_ID_TO_SHORT.get(action_id,
                                  ACTION_ID_TO_NAME.get(action_id, f"a{action_id}"))


# --------------------------------------------------------------------------- #
def get_font(sz: int):
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(cand):
            return ImageFont.truetype(cand, sz)
    return ImageFont.load_default()


def build_frame_lookup(episodes, dataset_tag: str) -> Dict[Tuple[str, str, int], str]:
    lookup: Dict[Tuple[str, str, int], str] = {}
    for ep in episodes:
        for t, rel_path in enumerate(ep.frame_paths):
            lookup[(dataset_tag, str(ep.episode_id), t)] = rel_path
    return lookup


def load_real_frame(image_root: str, episode_id: str, step: int,
                    size: int = 144,
                    frame_lookup: Optional[Dict[Tuple[str, str, int], str]] = None,
                    dataset_tag: Optional[str] = None) -> Image.Image:
    rel_path = None
    if frame_lookup is not None and dataset_tag is not None:
        rel_path = frame_lookup.get((dataset_tag, str(episode_id), step))
    if rel_path is not None:
        if _is_sqlite_store(image_root):
            p = f"{image_root}::{rel_path}"
        else:
            p = rel_path if os.path.isabs(rel_path) else os.path.join(image_root, rel_path)
    else:
        rel_path = os.path.join(str(episode_id), f"frame_{step}.jpg")
        p = f"{image_root}::{rel_path}" if _is_sqlite_store(image_root) else os.path.join(image_root, rel_path)
    if not _is_sqlite_store(image_root) and not os.path.exists(p):
        return Image.new("RGB", (size, size), (60, 0, 0))
    try:
        return _open_image(p).resize((size, size), Image.BILINEAR)
    except FileNotFoundError:
        return Image.new("RGB", (size, size), (60, 0, 0))


def label_tile(img, top="", bottom=""):
    W, H = img.size
    top_bar = 16
    bottom_lines = bottom if isinstance(bottom, list) else str(bottom).split("\n")
    bottom_lines = [line for line in bottom_lines if line]
    bottom_bar = max(16, 13 * max(1, len(bottom_lines)))
    out = Image.new("RGB", (W, H + top_bar + bottom_bar), (40, 40, 40))
    out.paste(img, (0, top_bar))
    d = ImageDraw.Draw(out)
    f = get_font(10)
    if top:
        d.text((3, 1), top, fill=(255, 255, 255), font=f)
    for i, line in enumerate(bottom_lines):
        d.text((3, H + top_bar + 1 + 12 * i), line,
               fill=(180, 220, 180), font=f)
    return out


def make_frame_strip(ep_id, K, N, viz_steps,
                     gt_feats, pred_feats,
                     gt_actions, eval_actions,
                     image_root, dataset_tag,
                     pool_n, pool_meta, device,
                     image_roots=None,
                     frame_lookup=None):
    """One row per ep: [GT@anchor] / [Pred-NN@h] for each h.

    image_roots : dict {tag -> image_root path}. Required when pool_meta
                  entries are 3-tuples (cross-dataset pool); ignored when
                  meta is 2-tuples (same-dataset pool).
    dataset_tag : the source tag of the current episode ('r2r' / 'rxr').
                  Used to mark [SAME] when NN happens to be the exact frame.
    """
    is_cross = pool_meta and len(pool_meta[0]) == 3
    gt_tiles, pred_tiles = [], []
    for h in viz_steps:
        gt_idx = K + h - 1
        action_idx = K + h - 2
        gt_action = action_label(gt_actions, action_idx)
        eval_action = action_label(eval_actions, action_idx)
        gt_tiles.append(label_tile(
            load_real_frame(image_root, ep_id, gt_idx,
                            frame_lookup=frame_lookup,
                            dataset_tag=dataset_tag),
            top=f"GT t={gt_idx}",
            bottom=[f"h=+{h}", f"gt={gt_action}"]
        ))
        # Pred via NN retrieval (decoder→feat→nearest pool frame)
        p = pred_feats[h - 1].to(device).unsqueeze(0)
        p_n = F.normalize(p, dim=-1)
        nn_idx = (pool_n @ p_n.T).squeeze(-1).argmax().item()
        if is_cross:
            nn_tag, nn_ep, nn_t = pool_meta[nn_idx]
            nn_root = image_roots[nn_tag] if image_roots else image_root
            same = (nn_tag == dataset_tag and nn_ep == ep_id and nn_t == gt_idx)
        else:
            nn_ep, nn_t = pool_meta[nn_idx]
            nn_root = image_root
            same = (nn_ep == ep_id and nn_t == gt_idx)
        cos = F.cosine_similarity(p, gt_feats[h - 1].to(device).unsqueeze(0),
                                  dim=-1).item()
        # NOTE: don't show dataset tag in label (user wants unified MP3D pool view)
        pred_tiles.append(label_tile(
            load_real_frame(nn_root, nn_ep, nn_t,
                            frame_lookup=frame_lookup,
                            dataset_tag=nn_tag if is_cross else dataset_tag),
            top=f"Pred-NN h=+{h}",
            bottom=[f"eval={eval_action}",
                    f"cos={cos:+.2f}{' [SAME]' if same else ''}"]
        ))
    # h-stack each row
    def hstack(tiles, pad=3):
        H = max(t.size[1] for t in tiles)
        W = sum(t.size[0] for t in tiles) + pad * (len(tiles) - 1)
        out = Image.new("RGB", (W, H), (40, 40, 40))
        x = 0
        for t in tiles:
            out.paste(t, (x, 0)); x += t.size[0] + pad
        return out
    row_gt   = hstack(gt_tiles)
    row_pred = hstack(pred_tiles)
    # vertical: GT on top, Pred below; with thin label "ep N"
    W = max(row_gt.size[0], row_pred.size[0])
    H = row_gt.size[1] + row_pred.size[1] + 28
    out = Image.new("RGB", (W, H), (28, 28, 28))
    d = ImageDraw.Draw(out)
    d.text((6, 6), f"episode {ep_id}: {K}-frame burn-in, then open-loop predict {N} steps",
           fill=(255, 255, 200), font=get_font(13))
    out.paste(row_gt,   (0, 28))
    out.paste(row_pred, (0, 28 + row_gt.size[1]))
    return out


def vstack_with_header(rows, header):
    W = max(r.size[0] for r in rows)
    H = sum(r.size[1] for r in rows) + 6 * (len(rows) - 1) + 32
    out = Image.new("RGB", (W, H), (28, 28, 28))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, W, 30], fill=(60, 60, 90))
    d.text((10, 6), header, fill=(255, 255, 255), font=get_font(15))
    y = 32
    for r in rows:
        out.paste(r, (0, y)); y += r.size[1] + 6
    return out


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--annotation",  required=True)
    p.add_argument("--cache_path",  required=True)
    p.add_argument("--image_root",  required=True)
    p.add_argument("--out_png",     required=True)
    p.add_argument("--burn_in",     type=int, default=5,
                   help="K = burn-in length for single-context eval")
    p.add_argument("--burn_ins",    type=int, nargs="+", default=None,
                   help="optional list of K values for context-length sweep")
    p.add_argument("--viz_burn_in", type=int, default=None,
                   help="K used for PNG visualization; defaults to first burn_ins entry")
    p.add_argument("--predict_n",   type=int, default=25,
                   help="N = open-loop horizon (Dreamer V3 uses 45; we use 25 since R2R/RxR eps shorter)")
    p.add_argument("--num_viz_eps", type=int, default=4,
                   help="how many episodes to put in the visual PNG")
    p.add_argument("--report_horizons", type=int, nargs="+",
                   default=[1, 5, 10, 15, 20, 25])
    p.add_argument("--viz_horizons", type=int, nargs="+",
                   default=[1, 3, 6, 9, 12, 15, 20, 25])
    p.add_argument("--test_ratio", type=float, default=0.10,
                   help="held-out ratio used by train_tssm_feat.py")
    p.add_argument("--num_test_eps", type=int, default=0,
                   help="cap test episodes; 0 means all held-out episodes")
    p.add_argument("--num_val_eps", type=int, default=None,
                   help="deprecated alias for --num_test_eps")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--dataset_tag", default="r2r",
                   help="tag identifying which dataset the val episodes come from "
                        "(used for [SAME] match in cross-pool mode)")
    p.add_argument("--cross_pool", action="store_true",
                   help="search NN against ALL frames from ALL listed datasets "
                        "(stronger generalization signal than same-split val pool)")
    p.add_argument("--cross_feat_paths", nargs="+", default=[],
                   help="when --cross_pool, list of 'tag:path_to_features.pt' "
                        "entries, e.g. r2r:/.../R2R/feat/siglip/features.pt "
                        "rxr:/.../RxR/feat/siglip/features.pt")
    p.add_argument("--cross_image_roots", nargs="+", default=[],
                   help="when --cross_pool, list of 'tag:image_root' for the "
                        "frame-strip viz, mapping pool entries back to jpg files")
    p.add_argument("--cross_annotations", nargs="+", default=[],
                   help="optional list of 'tag:annotations.json' entries used "
                        "to resolve exact frame_paths for cross-pool NN viz")
    p.add_argument("--run_tag", default="",
                   help="optional extra tag appended to the PNG header "
                        "(e.g. date or variant name)")
    p.add_argument("--viz_dir", default="",
                   help="if set, write one PNG per viz episode into this dir "
                        "(uses ALL viz_eps, ignores --num_viz_eps) instead of "
                        "stacking them into the aggregate --out_png")
    p.add_argument("--cross_max_pool", type=int, default=200000,
                   help="cap on cross-dataset pool size (random sub-sample)")
    p.add_argument("--action_mode", choices=["gt", "random", "fixed"],
                   default="gt",
                   help="actions for open-loop imagination; random keeps burn-in "
                        "GT actions and randomizes a_{K-1}..a_{K+N-2}; "
                        "fixed uses --fixed_action_id for that same segment")
    p.add_argument("--random_include_stop", action="store_true",
                   help="when --action_mode random, sample from [0,num_actions); "
                        "default samples non-stop actions [1,num_actions)")
    p.add_argument("--fixed_action_id", type=int, default=1,
                   help="when --action_mode fixed, use this action id for "
                        "open-loop rollout; default 1 = move forward 25 cm")
    args = p.parse_args()

    burn_ins = args.burn_ins or [args.burn_in]
    if any(k < 1 for k in burn_ins):
        raise ValueError("--burn_in/--burn_ins must be >= 1")
    viz_K = args.viz_burn_in if args.viz_burn_in is not None else burn_ins[0]
    if viz_K < 1:
        raise ValueError("--viz_burn_in must be >= 1")
    N = args.predict_n
    for h in args.report_horizons + args.viz_horizons:
        if h > N:
            raise ValueError(f"horizon {h} > predict_n {N}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    if args.action_mode == "random":
        action_low = 0 if args.random_include_stop else 1
        print(f"[actions] mode=random open-loop only; sample ids "
              f"[{action_low}, {cfg.num_actions}) seed={args.seed}")
    elif args.action_mode == "fixed":
        fixed_name = action_label([args.fixed_action_id], 0)
        print(f"[actions] mode=fixed open-loop only; "
              f"action_id={args.fixed_action_id} ({fixed_name})")
    else:
        print("[actions] mode=gt")

    eps_all = build_episode_trajectories(args.annotation)
    frame_lookup = build_frame_lookup(eps_all, args.dataset_tag)
    _, val_eps_all = split_episodes(eps_all, val_ratio=args.test_ratio, seed=args.seed)
    cap = args.num_test_eps if args.num_val_eps is None else args.num_val_eps
    if cap and cap > 0:
        val_eps_all = val_eps_all[:cap]
    print(f"[split] held-out test episodes={len(val_eps_all)} "
          f"(test_ratio={args.test_ratio}, seed={args.seed})")
    eval_eps_by_k = {
        K: [ep for ep in val_eps_all if len(ep.actions) >= K + N]
        for K in sorted(set(burn_ins + [viz_K]))
    }
    for K in sorted(eval_eps_by_k):
        print(f"[val] K={K}: {len(eval_eps_by_k[K])} episodes with len >= {K+N}")

    ds = NaVILAR2RDataset(
        annotation_path=args.annotation, seq_len=max(burn_ins + [viz_K]) + N,
        mode="cached", cache_path=args.cache_path, episodes=val_eps_all,
    )
    image_roots_map = {args.dataset_tag: args.image_root}
    if args.cross_pool:
        feat_paths = []
        for entry in args.cross_feat_paths:
            tag, p_ = entry.split(":", 1)
            feat_paths.append((tag, p_))
        for entry in args.cross_image_roots:
            tag, root = entry.split(":", 1)
            image_roots_map[tag] = root
        for entry in args.cross_annotations:
            tag, annot = entry.split(":", 1)
            frame_lookup.update(build_frame_lookup(
                build_episode_trajectories(annot), tag
            ))
        # Force-include the val/test episodes' frames so NN@1 hit IS achievable.
        # Without this, random sub-sampling drops GT frames -> NN@1 == 0 by construction.
        force_eps = {f"{args.dataset_tag}:{ep.episode_id}" for ep in val_eps_all}
        pool, pool_meta = build_cross_pool(
            feat_paths, max_pool=args.cross_max_pool, seed=args.seed,
            force_include_eps=force_eps,
        )
        is_cross = True
    else:
        pool, pool_meta = build_feat_pool(val_eps_all, ds, max_pool=16000)
        is_cross = False
        print(f"[pool] {len(pool_meta)} val frames for NN retrieval (same-dataset only)")
    pool_n = F.normalize(pool, dim=-1).to(device)
    # Build a fast lookup index: meta_tuple -> index into pool. Used to compute
    # the GT-frame rank (size-independent honest metric).
    meta_idx = {m: i for i, m in enumerate(pool_meta)}

    # ---------------- numerical eval over ALL val_eps ----------------
    # Match key for [SAME] check depends on pool format:
    #   same-dataset pool: (ep_id, step)
    #   cross-dataset pool: (dataset_tag, ep_id, step)
    def make_key(ep_id, step):
        return (args.dataset_tag, ep_id, step) if is_cross else (ep_id, step)

    pool_size = pool_n.shape[0]
    for K in burn_ins:
        val_eps = eval_eps_by_k[K]
        agg = {h: {"sum_mse": 0.0, "sum_cos": 0.0,
                   "sum_hit1": 0, "sum_hit5": 0,
                   "sum_rank": 0, "sum_pct": 0.0,   # GT rank metric
                   "rank_count": 0, "count": 0}
               for h in args.report_horizons}
        for i, ep in enumerate(val_eps):
            feat_seq = ds.features[ep.episode_id]
            action_seq = torch.tensor(ep.actions, dtype=torch.long)
            action_seq = make_rollout_actions(
                action_seq, K, N, cfg.num_actions, args.action_mode,
                args.seed, ep.episode_id, args.random_include_stop,
                args.fixed_action_id,
            )
            preds = burn_in_then_imagine(model, feat_seq, action_seq, K, N, device)
            for h in args.report_horizons:
                pred = preds[h - 1].to(device)
                gt_idx = K + h - 1
                gt   = feat_seq[gt_idx].float().to(device)
                mse  = F.mse_loss(pred, gt).item()
                cos  = F.cosine_similarity(pred.unsqueeze(0), gt.unsqueeze(0),
                                           dim=-1).item()
                p_n  = F.normalize(pred.unsqueeze(0), dim=-1)
                sims = (pool_n @ p_n.T).squeeze(-1)            # [pool_size]
                top5 = sims.topk(5).indices.tolist()
                target_key = make_key(ep.episode_id, gt_idx)
                hit1 = pool_meta[top5[0]] == target_key
                hit5 = target_key in [pool_meta[j] for j in top5]

                # GT rank: where is the GT frame in the cosine-sorted pool?
                # Pool-size-INDEPENDENT signal of "how close model recon is to GT
                # vs all other MP3D frames". Lower rank = better.
                gt_pool_idx = meta_idx.get(target_key, None)
                if gt_pool_idx is not None:
                    gt_sim = sims[gt_pool_idx].item()
                    # rank = number of pool frames with cosine STRICTLY higher than GT
                    rank = int((sims > gt_sim).sum().item())
                    pct = rank / max(1, pool_size - 1)         # in [0, 1]
                    agg[h]["sum_rank"] += rank
                    agg[h]["sum_pct"]  += pct
                    agg[h]["rank_count"] += 1

                agg[h]["sum_mse"]  += mse
                agg[h]["sum_cos"]  += cos
                agg[h]["sum_hit1"] += int(hit1)
                agg[h]["sum_hit5"] += int(hit5)
                agg[h]["count"]    += 1
            if (i + 1) % 50 == 0:
                print(f"  K={K}: processed {i+1}/{len(val_eps)} episodes")

        # ---------------- print numerical table ----------------
        print(f"\n=== Open-loop prediction (burn-in K={K}, predict_n N={N}, "
              f"pool={pool_size}) ===")
        print(f"{'horizon':<8s} {'target_t':>8s} {'MSE':>8s} {'cos_sim':>8s} "
              f"{'NN@1':>7s} {'NN@5':>7s} "
              f"{'GT_rank':>9s} {'GT_pct':>8s} {'count':>6s}")
        print("-" * 80)
        for h in args.report_horizons:
            c = max(1, agg[h]["count"])
            rc = max(1, agg[h]["rank_count"])
            mse = agg[h]["sum_mse"] / c
            cos = agg[h]["sum_cos"] / c
            nn1 = agg[h]["sum_hit1"] / c
            nn5 = agg[h]["sum_hit5"] / c
            rank = agg[h]["sum_rank"] / rc
            pct  = agg[h]["sum_pct"]  / rc
            print(f"+{h:<7d} {K+h-1:>8d} {mse:>8.4f} {cos:>8.4f} "
                  f"{nn1*100:>6.2f}% {nn5*100:>6.2f}% "
                  f"{rank:>9.0f} {pct*100:>7.3f}% {c:>6d}")
    print()
    print("Reading guide:")
    print("  cos_sim:  closer to 1 = recon ~ GT in feat space")
    print("  NN@1/@5:  exact-frame retrieval (very harsh on big pool)")
    print("  GT_rank:  where GT sits among 1M MP3D frames sorted by cos to recon")
    print("            lower = better; rank=0 means GT is the closest frame")
    print("  GT_pct:   GT_rank / pool_size; 0% = perfect, 50% = random recon")

    # ---------------- visual frame strip ----------------
    viz_eps = eval_eps_by_k[viz_K]

    def _strip_for(ep):
        feat_seq = ds.features[ep.episode_id]
        gt_action_seq = torch.tensor(ep.actions, dtype=torch.long)
        eval_action_seq = make_rollout_actions(
            gt_action_seq, viz_K, N, cfg.num_actions, args.action_mode,
            args.seed, ep.episode_id, args.random_include_stop,
            args.fixed_action_id,
        )
        preds = burn_in_then_imagine(model, feat_seq, eval_action_seq,
                                     viz_K, N, device)
        gt_feats = [feat_seq[viz_K + h - 1].float() for h in range(1, N + 1)]
        return make_frame_strip(
            ep.episode_id, viz_K, N, args.viz_horizons,
            gt_feats, preds, gt_action_seq, eval_action_seq,
            args.image_root, args.dataset_tag,
            pool_n, pool_meta, device,
            image_roots=image_roots_map,
            frame_lookup=frame_lookup,
        )

    header_base = (f"TransDreamer-style open-loop prediction  |  "
                   f"ckpt={os.path.basename(args.ckpt)}  "
                   f"|  burn-in K={viz_K}, horizon N={N}  "
                   f"|  actions={args.action_mode}")
    if args.run_tag:
        header_base += f"  |  {args.run_tag}"

    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
        print(f"\n[viz] writing one PNG per episode -> {args.viz_dir}  "
              f"(total {len(viz_eps)} eps)")
        for i, ep in enumerate(viz_eps):
            strip = _strip_for(ep)
            out_img = vstack_with_header([strip],
                                         f"{header_base}  |  ep {ep.episode_id}")
            # filesystem-safe episode id
            safe_id = str(ep.episode_id).replace("/", "_")
            ep_path = os.path.join(args.viz_dir, f"ep_{safe_id}.png")
            out_img.save(ep_path)
            if (i + 1) % 50 == 0 or (i + 1) == len(viz_eps):
                print(f"  [viz] saved {i+1}/{len(viz_eps)}")
        print(f"\n[done] {len(viz_eps)} PNGs saved under {args.viz_dir}")
    else:
        print(f"\n[viz] making frame strip for first "
              f"{args.num_viz_eps} episodes ...")
        rows = [_strip_for(ep) for ep in viz_eps[:args.num_viz_eps]]
        out_img = vstack_with_header(rows, header_base)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_png)) or ".",
                    exist_ok=True)
        out_img.save(args.out_png)
        print(f"\n[done] PNG saved -> {args.out_png}  "
              f"({out_img.size[0]}x{out_img.size[1]})")


if __name__ == "__main__":
    main()

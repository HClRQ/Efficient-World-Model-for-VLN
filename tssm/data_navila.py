"""
NaVILA-aligned R2R dataset for RSSM pre-training.

Each episode in NaVILA's annotations.json contains ~K decision points
(samples), augmented with random duplicates. For RSSM we want a clean
per-episode trajectory:

    (o_0, a_0) -> (o_1, a_1) -> ... -> (o_{T-1}, a_{T-1}=stop)

where o_t = the RGB frame at decision point t (= sample.frames[-1] in the
annotation), and a_t is the compound action label decoded from sample.a.

Action vocab (10 classes):
    0  stop
    1  move forward 25 cm
    2  move forward 50 cm
    3  move forward 75 cm
    4  turn left  15 degree
    5  turn left  30 degree
    6  turn left  45 degree
    7  turn right 15 degree
    8  turn right 30 degree
    9  turn right 45 degree

The dataset can return full episodes from step 0 (for inference-matched TSSM
training) or fixed-length subsequences for BPTT. Two observation modes:
    mode='cached'  -> reads precomputed feature tensors from cache_dir
    mode='image'   -> loads raw JPGs (slow; only for debugging)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ACTION_VOCAB: Dict[str, int] = {
    "stop":                 0,
    "move forward 25 cm":   1,
    "move forward 50 cm":   2,
    "move forward 75 cm":   3,
    "turn left 15 degree":  4,
    "turn left 30 degree":  5,
    "turn left 45 degree":  6,
    "turn right 15 degree": 7,
    "turn right 30 degree": 8,
    "turn right 45 degree": 9,
}
NUM_ACTIONS = len(ACTION_VOCAB)
_ACTION_RE = re.compile(
    r"(stop|move forward \d+ cm|turn (?:left|right) \d+ degree)", re.IGNORECASE
)


def parse_action(text: str) -> int:
    m = _ACTION_RE.search(text.lower())
    if m is None:
        raise ValueError(f"unparseable action: {text!r}")
    key = m.group(1)
    if key == "stop":
        return 0
    return ACTION_VOCAB[key]


@dataclass
class EpisodeTraj:
    """One episode's full decision trajectory after dedup + sort."""
    episode_id: str
    frame_paths: List[str]   # length T, frame_paths[t] = relative path of o_t
    actions: List[int]       # length T, actions[t] = a_t taken from o_t


def build_episode_trajectories(
    annotation_path: str,
) -> List[EpisodeTraj]:
    """Load annotations.json and rebuild per-episode (o_t, a_t) trajectories.

    Handles NaVILA's "augmented with duplicate samples" by deduping on
    (step, frames[-1], a). Verifies every episode ends with stop.
    """
    with open(annotation_path) as f:
        ann = json.load(f)

    by_ep: Dict[str, Dict[int, Tuple[str, str]]] = defaultdict(dict)
    for s in ann:
        ep_str, step_str = s["video_id"].split("-")
        step = int(step_str)
        last_frame = s["frames"][-1]
        a = s["a"]
        # Dedup: keep the first occurrence of each step. Duplicate
        # augmentation samples are byte-identical for (frames, a) at
        # the same step, so this is safe.
        by_ep[ep_str].setdefault(step, (last_frame, a))

    trajs: List[EpisodeTraj] = []
    for ep, step_map in by_ep.items():
        steps = sorted(step_map.keys())
        if steps != list(range(len(steps))):
            raise RuntimeError(f"episode {ep} has gaps after dedup: {steps[:20]}...")
        frame_paths = [step_map[t][0] for t in steps]
        actions = [parse_action(step_map[t][1]) for t in steps]
        if actions[-1] != 0:
            raise RuntimeError(f"episode {ep} does not end with stop: a[-1]={actions[-1]}")
        trajs.append(EpisodeTraj(ep, frame_paths, actions))
    return trajs


# ---------------------------------------------------------------------------
# Image-mode helpers
# ---------------------------------------------------------------------------

_SQLITE_HANDLES = {}


def _is_sqlite_store(path: str) -> bool:
    return isinstance(path, str) and path.endswith((".sqlite", ".sqlite3", ".db"))


def _get_sqlite_conn(db_path: str):
    cache_key = (os.getpid(), db_path)
    conn = _SQLITE_HANDLES.get(cache_key)
    if conn is None:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        _SQLITE_HANDLES[cache_key] = conn
    return conn


def _sqlite_frame_path(db_path: str, frame_path: str) -> str:
    return f"{db_path}::{frame_path}"


def _frame_exists(root: str, frame_path: str) -> bool:
    if _is_sqlite_store(root):
        conn = _get_sqlite_conn(root)
        row = conn.execute("SELECT 1 FROM frames WHERE key = ? LIMIT 1", (frame_path,)).fetchone()
        return row is not None
    return os.path.exists(os.path.join(root, frame_path))


def _open_image(path: str) -> Image.Image:
    if isinstance(path, str) and "::" in path:
        db_path, key = path.split("::", 1)
        conn = _get_sqlite_conn(db_path)
        row = conn.execute("SELECT image FROM frames WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"{key} not found in {db_path}")
        return Image.open(BytesIO(row[0])).convert("RGB")
    return Image.open(path).convert("RGB")


def _load_image_tensor(path: str, size: int = 224) -> torch.Tensor:
    img = _open_image(path).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NaVILAR2RDataset(Dataset):
    """Per-episode RSSM dataset over NaVILA's R2R annotations.

    Each __getitem__ returns either one full episode from step 0 or one BPTT
    subsequence sampled from a chosen episode:

        {
            "feat":        FloatTensor[L, D]   # observation features o_0..o_{L-1}
                                               # (only when mode='cached')
            "image":       FloatTensor[L, 3, H, W]  # only when mode='image'
            "action":      LongTensor[L]       # a_0..a_{L-1}
            "is_first":    BoolTensor[L]       # True only at t=0 of episode
            "is_terminal": BoolTensor[L]       # True iff this step's action is stop
            "episode_id":  str
            "start":       int                 # global offset of this subsequence
        }

    Full episode mode:
        - if full_episode=True, return the whole trajectory from step 0.
          This matches posterior filtering at inference/cache time.

    Subsequence sampling:
        - if episode_len >= seq_len: random start in [0, episode_len-seq_len]
        - else: pad on the right by repeating the terminal step (with
          is_terminal=True throughout the pad) so the model still gets a
          consistent length; downstream loss should mask post-terminal steps.
    """

    def __init__(
        self,
        annotation_path: str,
        seq_len: int = 32,
        mode: str = "cached",                 # 'cached' | 'image'
        cache_path: Optional[str] = None,     # required if mode='cached'
                                              #   path to features.pt OR a directory
                                              #   containing features.pt[.partial] /
                                              #   features_rank*.pt
        image_root: Optional[str] = None,     # required if mode='image'
        image_size: int = 224,
        min_episode_len: int = 2,             # drop degenerate episodes
        episodes: Optional[List[EpisodeTraj]] = None,  # for splits / reuse
        cache_dir: Optional[str] = None,      # back-compat alias for cache_path
        full_episode: bool = False,           # return full episode from step 0
        feas_labels: Optional[Dict[str, torch.Tensor]] = None,  # 08-07 可行性标签
        feas_prefix: str = "",                # 键前缀(r2r:/rxrL:/rxr:)
    ):
        super().__init__()
        if mode not in ("cached", "image"):
            raise ValueError(f"mode must be 'cached' or 'image', got {mode!r}")
        if mode == "image" and image_root is None:
            raise ValueError("image_root required when mode='image'")
        if mode == "cached":
            cache_path = cache_path or cache_dir
            if cache_path is None:
                raise ValueError("cache_path required when mode='cached'")

        self.seq_len = seq_len
        self.full_episode = full_episode
        self.feas_labels = feas_labels
        self.feas_prefix = feas_prefix
        self.mode = mode
        self.image_root = image_root
        self.image_size = image_size

        if episodes is None:
            episodes = build_episode_trajectories(annotation_path)
        self.episodes = [e for e in episodes if len(e.actions) >= min_episode_len]

        # In cached mode we load the entire features dict into RAM once.
        # File layout (any of the following works):
        #   <cache_path>                     # direct path to a .pt dict
        #   <cache_path>/features.pt
        #   <cache_path>/features.pt.partial      (resume / mid-run)
        #   <cache_path>/features_rank*.pt        (sharded run)
        self.features: Dict[str, torch.Tensor] = {}
        if mode == "cached":
            self._load_features(cache_path)
            missing = [e.episode_id for e in self.episodes
                       if e.episode_id not in self.features]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} episodes have no cached features at {cache_path} "
                    f"(first few: {missing[:5]}). Run precompute_features.py first."
                )

    def _load_features(self, cache_path: str) -> None:
        import glob
        if os.path.isfile(cache_path):
            paths = [cache_path]
        else:
            candidates = [
                os.path.join(cache_path, "features.pt"),
                *sorted(glob.glob(os.path.join(cache_path, "features_rank*.pt"))),
                os.path.join(cache_path, "features.pt.partial"),
            ]
            paths = [p for p in candidates if os.path.exists(p)]
            if not paths:
                raise FileNotFoundError(
                    f"no features file found under {cache_path} "
                    f"(looked for features.pt / features_rank*.pt / features.pt.partial)"
                )
        for p in paths:
            d = torch.load(p, map_location="cpu", weights_only=True)
            assert isinstance(d, dict), f"{p} is not a dict-of-tensors checkpoint"
            self.features.update(d)

    # -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep = self.episodes[idx]
        T = len(ep.actions)
        if self.full_episode:
            start = 0
            actions = list(ep.actions)
            frame_paths = list(ep.frame_paths)
            L = T
            pad = 0
        else:
            L = self.seq_len
            if T >= L:
                start = int(np.random.randint(0, T - L + 1))
                end = start + L
                actions = ep.actions[start:end]
                frame_paths = ep.frame_paths[start:end]
                pad = 0
            else:
                start = 0
                actions = list(ep.actions) + [0] * (L - T)  # pad with stop
                frame_paths = list(ep.frame_paths) + [ep.frame_paths[-1]] * (L - T)
                pad = L - T

        # is_terminal[t] = True iff this step's action is stop (a==0).
        # is_first[t]    = True only when t == 0 of episode AND start == 0.
        is_terminal = torch.tensor([a == 0 for a in actions], dtype=torch.bool)
        is_first = torch.zeros(L, dtype=torch.bool)
        if start == 0:
            is_first[0] = True
        # Mark padded steps as terminal so the loss can mask them.
        if pad > 0:
            is_terminal[L - pad:] = True

        out: Dict[str, torch.Tensor] = {
            "action":      torch.tensor(actions, dtype=torch.long),
            "is_first":    is_first,
            "is_terminal": is_terminal,
            "episode_id":  ep.episode_id,
            "start":       start,
        }

        # 08-07 可行性标签: [L,39] float + 有效位(集缺标签或长度不符则无效)
        if self.feas_labels is not None:
            lab = self.feas_labels.get(self.feas_prefix + str(ep.episode_id))
            if lab is not None and lab.shape[0] == T:
                sub_l = lab[start:start + (L - pad)].float()
                if pad > 0:
                    sub_l = torch.cat([sub_l, sub_l[-1:].expand(pad, -1)], dim=0)
                out["feas"] = sub_l
                out["feas_ok"] = True
            else:
                out["feas"] = torch.zeros(L, 39)
                out["feas_ok"] = False

        if self.mode == "cached":
            feat = self.features[ep.episode_id]   # [T_ep, D] fp16, or [T_ep, P, D] patch cache
            flat_patch_cache = feat.dim() == 3
            if feat.dim() == 3:
                # non-pooled patch-grid cache (e.g. siglip8f_p9_lm2): flatten
                # [T, P, 1152] -> [T, P*1152]; model probes feat_dim from this.
                feat = feat.reshape(feat.shape[0], -1)
            assert feat.shape[0] == T, (
                f"cache mismatch for ep {ep.episode_id}: cache T={feat.shape[0]} vs "
                f"annotation T={T}. Re-run precompute_features.py."
            )
            sub = feat[start:start + (L - pad)]
            if pad > 0:
                sub = torch.cat([sub, sub[-1:].expand(pad, -1)], dim=0)
            # Patch caches (flattened 3D) stay fp16 end-to-end: at 10368-d a
            # fp32 full-episode batch is ~0.5-1GB and the prefetch pipeline
            # OOMs host RAM (job 5849363). Model forward .float()s on GPU.
            out["feat"] = sub if flat_patch_cache else sub.float()
        else:  # image mode
            if _is_sqlite_store(self.image_root):
                paths = [_sqlite_frame_path(self.image_root, p) for p in frame_paths]
            else:
                paths = [os.path.join(self.image_root, p) for p in frame_paths]
            imgs = [_load_image_tensor(p, self.image_size) for p in paths]
            out["image"] = torch.stack(imgs, dim=0)  # [L, 3, H, W]

        return out


# ---------------------------------------------------------------------------
# Train / val split helper
# ---------------------------------------------------------------------------

def split_episodes(
    episodes: List[EpisodeTraj],
    val_ratio: float = 0.02,
    seed: int = 0,
) -> Tuple[List[EpisodeTraj], List[EpisodeTraj]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train_eps = [e for i, e in enumerate(episodes) if i not in val_idx]
    val_eps   = [e for i, e in enumerate(episodes) if i in val_idx]
    return train_eps, val_eps


if __name__ == "__main__":
    # Smoke test
    ann_path = "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json"
    img_root = "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/train"

    eps = build_episode_trajectories(ann_path)
    lens = [len(e.actions) for e in eps]
    print(f"episodes: {len(eps)}")
    print(f"trajectory length: min={min(lens)} median={int(np.median(lens))} "
          f"mean={np.mean(lens):.1f} max={max(lens)}")

    ds = NaVILAR2RDataset(
        annotation_path=ann_path,
        mode="image",
        image_root=img_root,
        seq_len=8,
        episodes=eps[:4],
    )
    sample = ds[0]
    print({k: (v.shape if torch.is_tensor(v) else v) for k, v in sample.items()})

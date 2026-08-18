#!/usr/bin/env python3
"""
Train NavigationTSSM (transformer state-space model) on NaVILA's R2R
trajectories. newest

Mirrors RSSM_v4/train_rssm_feat.py byte-for-byte at the data / loop / logging
level; only the model and config classes are swapped.  Run script side-by-side
with the RSSM run to compare training curves and downstream cache utility.

Typical launch:
    python RSSM_v4/train_tssm_feat.py \
        --annotation /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json \
        --cache_path /cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/siglip \
        --save_dir   /cluster/scratch/ruiqzhang/tssm_feat_ckpt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_navila import (                                       # noqa: E402
    NaVILAR2RDataset, build_episode_trajectories, split_episodes, NUM_ACTIONS,
)
from navigation_tssm_feat import NavigationTSSM, TSSMConfig     # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def collate(batch):
    """Pad variable-length full episodes to the longest episode in the batch."""
    batch_size = len(batch)
    lengths = torch.tensor([b["action"].shape[0] for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = int(batch[0]["feat"].shape[-1])

    feat = batch[0]["feat"].new_zeros(batch_size, max_len, feat_dim)
    action = torch.zeros(batch_size, max_len, dtype=torch.long)
    is_first = torch.zeros(batch_size, max_len, dtype=torch.bool)
    is_terminal = torch.ones(batch_size, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = int(lengths[i].item())
        feat[i, :L] = b["feat"]
        action[i, :L] = b["action"]
        is_first[i, :L] = b["is_first"]
        is_terminal[i, :L] = b["is_terminal"]
        if L < max_len:
            feat[i, L:] = b["feat"][-1]

    out = {
        "feat": feat,
        "action": action,
        "is_first": is_first,
        "is_terminal": is_terminal,
        "length": lengths,
        "episode_id": [b["episode_id"] for b in batch],
        "start": [b["start"] for b in batch],
    }
    if "feas" in batch[0]:
        fe = torch.zeros(batch_size, max_len, 39)
        for i, b in enumerate(batch):
            L = int(lengths[i].item())
            fe[i, :L] = b["feas"][:L]
        out["feas"] = fe
        out["feas_ok"] = torch.tensor([b["feas_ok"] for b in batch], dtype=torch.bool)
    return out


def train(args):
    # 07-29 DDP(用户拍板双A100): torchrun 启动时自动进入分布式; 单进程行为完全不变。
    import os as _os
    ddp = "RANK" in _os.environ and int(_os.environ.get("WORLD_SIZE", "1")) > 1
    rank, world = 0, 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        torch.cuda.set_device(int(_os.environ["LOCAL_RANK"]))
        if rank != 0:
            logger.setLevel("WARNING")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ----- data -----
    # `--annotation` and `--cache_path` may be comma-separated lists of equal
    # length (e.g. R2R + RxR). When more than one is given we build one
    # NaVILAR2RDataset per (annot, cache) pair, do the train/test split inside
    # each pair, then ConcatDataset the train halves and test halves so the
    # DataLoader gives uniformly mixed batches across all datasets.
    from torch.utils.data import ConcatDataset
    annot_list = [s.strip() for s in args.annotation.split(",") if s.strip()]
    cache_list = [s.strip() for s in args.cache_path.split(",") if s.strip()]
    if len(annot_list) != len(cache_list):
        raise ValueError(
            f"--annotation and --cache_path must have the same number of "
            f"comma-separated entries (got {len(annot_list)} vs {len(cache_list)})"
        )

    feas_dict = None
    if args.feas_labels:
        feas_dict = torch.load(args.feas_labels, map_location="cpu", weights_only=True)
        logger.info(f"[Feas] loaded {len(feas_dict)} episode label tensors")

    def _feas_prefix(path):
        p = path.lower()
        if "stream" in p:
            return "rxr:"
        if "/rxr/" in p:
            return "rxrL:"
        return "r2r:"

    train_subsets, test_subsets = [], []
    for i, (annot, cache) in enumerate(zip(annot_list, cache_list)):
        eps_i = build_episode_trajectories(annot)
        tr_eps, te_eps = split_episodes(
            eps_i, val_ratio=args.test_ratio, seed=args.seed,
        )
        logger.info(
            f"[Data] dataset {i}: annot={os.path.basename(os.path.dirname(annot))}  "
            f"train={len(tr_eps)}  test={len(te_eps)}"
        )
        tr_ds_i = NaVILAR2RDataset(
            annotation_path=annot, seq_len=args.seq_len,
            mode="cached", cache_path=cache, episodes=tr_eps,
            full_episode=args.full_episode,
            feas_labels=feas_dict, feas_prefix=_feas_prefix(annot),
        )
        te_ds_i = NaVILAR2RDataset(
            annotation_path=annot, seq_len=args.seq_len,
            mode="cached", cache_path=cache, episodes=te_eps,
            full_episode=args.full_episode,
            feas_labels=feas_dict, feas_prefix=_feas_prefix(annot),
        )
        # Share features dict between train/test halves of the same dataset
        # to halve memory (~670MB per side, otherwise loaded twice).
        te_ds_i.features = tr_ds_i.features
        train_subsets.append(tr_ds_i)
        test_subsets.append(te_ds_i)

    train_ds = train_subsets[0] if len(train_subsets) == 1 else ConcatDataset(train_subsets)
    test_ds  = test_subsets[0]  if len(test_subsets)  == 1 else ConcatDataset(test_subsets)
    logger.info(
        f"[Data] total: train={sum(len(s) for s in train_subsets)}  "
        f"test={sum(len(s) for s in test_subsets)}  "
        f"(across {len(train_subsets)} dataset(s))  "
        f"full_episode={args.full_episode}"
    )

    _sampler = None
    _bs = args.batch_size
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        _sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False)
        _bs = max(1, args.batch_size // world)     # 全局批不变(64=2×32), 动力学可比
    train_loader = DataLoader(
        train_ds, batch_size=_bs, shuffle=(_sampler is None), sampler=_sampler,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), collate_fn=collate,
        pin_memory=True, drop_last=False,
    )

    probe = train_ds[0]["feat"]
    feat_dim = int(probe.shape[-1])
    logger.info(f"[Data] feat_dim={feat_dim}  (probed from first sample)")

    # ----- model -----
    cfg = TSSMConfig(
        feat_dim=feat_dim,
        encoder_dim=args.encoder_dim,
        h_dim=args.h_dim,
        stoch=args.stoch,
        classes=args.classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_dim=args.ffn_dim,
        transformer_dropout=args.transformer_dropout,
        max_seq_len=args.max_seq_len,
        num_actions=NUM_ACTIONS,
        free_nats=args.free_nats,
        kl_balance=args.kl_balance,
        kl_scale=args.kl_scale,
        deter_type=args.deter_type,
        recon_type=args.recon_type,
        rec_sigma=args.rec_sigma,
        imagination_loss_weight=args.imagination_loss_weight,
        imagination_horizon=args.imagination_horizon,
        imagination_n_starts=args.imagination_n_starts,
        future_weight=args.future_weight,
        future_horizon=args.future_horizon,
        future_stopgrad_decoder=args.future_stopgrad_decoder,
        nce_lag_fourier=args.nce_lag_fourier,
        nce_lag_sample=args.nce_lag_sample,
        nce_grade_tau=args.nce_grade_tau,
        slow_modules=args.slow_modules,
        nce_head_hidden=args.nce_head_hidden,
        nce_head_layers=args.nce_head_layers,
        nce_event_mode=args.nce_event_mode,
        nce_event_quantile=args.nce_event_quantile,
        nce_dim=args.nce_dim,
        future_n_starts=args.future_n_starts,
        recon_weight=args.recon_weight,
        retro_weight=args.retro_weight,
        nce_weight=args.nce_weight,
        nce_lags=args.nce_lags,
        fut_nce_weight=args.fut_nce_weight,
        fut_nce_steps=args.fut_nce_steps,
        fut_nce_lambda=args.fut_nce_lambda,
        fut_nce_starts=args.fut_nce_starts,
        fut_nce_hstart=args.fut_nce_hstart,
        fut_nce_spec_lags=args.fut_nce_spec_lags,
        attn_window=args.attn_window,
        core_type=args.core_type,
        fut_nce_norm=not args.fut_nce_raw,
        feas_weight=args.feas_weight,
        trav_weight=args.trav_weight,
        nce_tau=args.nce_tau,
        raw_fwd_weight=args.raw_fwd_weight,
        raw_retro_weight=args.raw_retro_weight,
        raw_lags=args.raw_lags,
        retro_lags=args.retro_lags,
        retro_n_starts=args.retro_n_starts,
        fwd_weight=args.fwd_weight,
        fwd_lags=args.fwd_lags,
        fwd_n_starts=args.fwd_n_starts,
        latent_weight=args.latent_weight,
        latent_lags=args.latent_lags,
        latent_n_starts=args.latent_n_starts,
        aux_actx=args.aux_actx,
        fast_aux=args.fast_aux,
        posterior_ctx=args.posterior_ctx,
        post_ctx_layers=args.post_ctx_layers,
        action_inv_weight=args.action_inv_weight,
        action_inv_hidden_dim=args.action_inv_hidden_dim,
        action_inv_layers=args.action_inv_layers,
        pose_delta_weight=args.pose_delta_weight,
        pose_delta_hidden_dim=args.pose_delta_hidden_dim,
        pose_delta_layers=args.pose_delta_layers,
        recon_groups=args.recon_groups,
        recon_sem_dim=args.recon_sem_dim,
        recon_sem_boost=args.recon_sem_boost,
        feat_scale=args.feat_scale,
    )
    tssm = NavigationTSSM(cfg).to(device)
    _raw_tssm = tssm
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        tssm = DDP(tssm, device_ids=[int(_os.environ["LOCAL_RANK"])],
                   find_unused_parameters=True)  # actx_gru等条件性模块可能整步未用
    logger.info(f"[TSSM] params: {sum(p.numel() for p in tssm.parameters())/1e6:.1f}M")
    logger.info(f"[TSSM] raw_fwd_w={cfg.raw_fwd_weight} raw_retro_w={cfg.raw_retro_weight} "
                f"raw_lags={cfg.raw_lags} raw_heads={_raw_tssm.raw_fwd_head is not None}/{_raw_tssm.raw_retro_head is not None}")

    optimizer = torch.optim.AdamW(tssm.parameters(), lr=args.lr,
                                  weight_decay=1e-4, eps=1e-8,
                                  fused=torch.cuda.is_available())
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-5)

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "tensorboard"), flush_secs=10)

    # ----- resume from checkpoint -----
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = _raw_tssm.load_state_dict(ckpt["model"], strict=False)
        if missing:
            logger.warning(f"[resume] missing model keys: {missing}")
        if unexpected:
            logger.warning(f"[resume] unexpected model keys: {unexpected}")
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as exc:
                logger.warning(f"[resume] skipped optimizer state: {exc}")
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except ValueError as exc:
                logger.warning(f"[resume] skipped scheduler state: {exc}")
        step = ckpt.get("step", 0)
        logger.info(f"[resume] loaded {args.resume}  step={step}")

    # ----- train loop -----
    tssm.train()
    log_keys = [
        "dyn_kl", "rep_kl", "recon", "imag_recon",
        "future", "retro", "fwd", "latent", "lat_std",   # 07-26: 漏加导致日志恒显0(训练其实在算)
        "raw_fwd", "raw_retro", "nce", "nce_acc",
        "fut_nce", "fut_nce_acc", "fut_acc_far",   # 08-06: 同一坑第三次就写进家法
        "feas", "trav", "feas_acc",
        "action_inv", "pose_delta", "pose_delta_fwd_mae",
        "pose_delta_yaw_mae_deg", "total",
    ]
    running = {k: 0.0 for k in log_keys}
    t_log = time.time()

    _epoch = 0
    while step < args.max_steps:
        if ddp and _sampler is not None:
            _sampler.set_epoch(_epoch)
        _epoch += 1
        for batch in train_loader:
            if step >= args.max_steps:
                break
            feat     = batch["feat"].to(device, non_blocking=True)            # [B, L, D]
            action   = batch["action"].to(device, non_blocking=True)          # [B, L]
            is_first = batch["is_first"].to(device, non_blocking=True)        # [B, L]
            terminal = batch["is_terminal"].to(device, non_blocking=True)     # [B, L]
            # Mask out steps strictly AFTER the terminal step (keep the
            # terminal step itself so the model still sees STOP).
            past_term = torch.cummax(terminal.long(), dim=1).values
            shifted = torch.zeros_like(past_term)
            shifted[:, 1:] = past_term[:, :-1]
            mask = shifted == 0

            kl_scale = min(1.0, step / max(1, args.kl_warmup))

            optimizer.zero_grad(set_to_none=True)
            _fe = batch.get("feas")
            _fo = batch.get("feas_ok")
            if _fe is not None:
                _fe = _fe.to(device, non_blocking=True)
                _fo = _fo.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                out = tssm(feat, action, is_first=is_first, mask=mask, kl_scale=kl_scale,
                           feas=_fe, feas_ok=_fo)

            out.loss.backward()
            nn.utils.clip_grad_norm_(tssm.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            for k in log_keys:
                running[k] += out.loss_dict[k]

            if step % args.log_every == 0 and step > 0:
                _do_test = (step % args.test_every == 0)
                avg = {k: v / args.log_every for k, v in running.items()}
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t_log

                # Quick test pass alongside training log so train/test are
                # plotted on the same x-axis.
                # 07-27: 全量test过一遍≈每周期1.5s(~25%墙钟); 降频到 test_every, 纯评估零训练影响
                if _do_test:
                    test_avg = run_test_loss(_raw_tssm, test_loader, device, writer, step)
                    _last_test = test_avg
                else:
                    test_avg = _last_test if '_last_test' in dir() else run_test_loss(_raw_tssm, test_loader, device, writer, step)
                tssm.train()

                # 07-28 用户令: 只打印本次运行激活的损失(权重>0), 死开关不再刷屏
                _act_keys = ["total", "recon", "dyn_kl", "rep_kl"]
                for _k, _w in [("nce", args.nce_weight), ("nce_acc", args.nce_weight),
                               ("fut_nce", args.fut_nce_weight), ("fut_nce_acc", args.fut_nce_weight),
                               ("fut_acc_far", args.fut_nce_weight),
                               ("feas", args.feas_weight), ("feas_acc", args.feas_weight),
                               ("trav", args.trav_weight),
                               ("future", args.future_weight), ("retro", args.retro_weight),
                               ("raw_fwd", args.raw_fwd_weight), ("raw_retro", args.raw_retro_weight),
                               ("fwd", args.fwd_weight), ("latent", args.latent_weight),
                               ("action_inv", args.action_inv_weight), ("pose_delta", args.pose_delta_weight)]:
                    if _w > 0 and _k not in _act_keys:
                        _act_keys.append(_k)
                _tr = " ".join(f"{k} {avg.get(k, 0.0):.4f}" for k in _act_keys)
                _te = " ".join(f"{k} {test_avg.get(k, 0.0):.4f}" for k in _act_keys if k in test_avg)
                logger.info(
                    f"step {step:>6} | train {_tr} | test {_te} | "
                    f"kl_s {kl_scale:.2f} lr {lr_now:.2e} | {elapsed:.0f}s"
                )
                for k, v in avg.items():
                    writer.add_scalar(f"train/{k}", v, step)
                writer.add_scalar("train/kl_scale", kl_scale, step)
                writer.add_scalar("train/lr", lr_now, step)
                running = {k: 0.0 for k in log_keys}
                t_log = time.time()

            if step % args.ckpt_every == 0 and step > 0 and rank == 0:
                ckpt_path = os.path.join(args.save_dir, f"checkpoint-{step}.pt")
                torch.save(
                    {"step": step, "model": _raw_tssm.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "config": cfg.__dict__},
                    ckpt_path,
                )
                logger.info(f"[ckpt] -> {ckpt_path}")

            step += 1

    final_path = os.path.join(args.save_dir, "checkpoint-final.pt")
    if rank == 0:
        torch.save({"step": step, "model": _raw_tssm.state_dict(), "config": cfg.__dict__},
                   final_path)
    logger.info(f"[ckpt] final -> {final_path}")


@torch.no_grad()
def run_test_loss(tssm, loader, device, writer, step):
    """Full pass over the held-out test loader; returns average loss dict."""
    tssm.eval()
    total = {
        "dyn_kl": 0.0,
        "rep_kl": 0.0,
        "recon": 0.0,
        "imag_recon": 0.0,
        "action_inv": 0.0,
        "pose_delta": 0.0,
        "pose_delta_fwd_mae": 0.0,
        "pose_delta_yaw_mae_deg": 0.0,
        "total": 0.0,
    }
    n = 0
    for batch in loader:
        feat     = batch["feat"].to(device, non_blocking=True)
        action   = batch["action"].to(device, non_blocking=True)
        is_first = batch["is_first"].to(device, non_blocking=True)
        terminal = batch["is_terminal"].to(device, non_blocking=True)
        past_term = torch.cummax(terminal.long(), dim=1).values
        shifted = torch.zeros_like(past_term)
        shifted[:, 1:] = past_term[:, :-1]
        mask = shifted == 0

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = tssm(feat, action, is_first=is_first, mask=mask, kl_scale=1.0)
        for k in total:
            total[k] += out.loss_dict.get(k, 0.0)
        n += 1
    avg = ({k: v / n for k, v in total.items()} if n > 0
           else {k: 0.0 for k in total})
    for k, v in avg.items():
        writer.add_scalar(f"test/{k}", v, step)
    return avg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--annotation",
                   default="/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json")
    p.add_argument("--cache_path",
                   default="/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/siglip")
    p.add_argument("--save_dir",
                   default="/cluster/scratch/ruiqzhang/tssm_feat_ckpt")
    p.add_argument("--resume",     default=None,
                   help="path to checkpoint .pt to resume from")
    p.add_argument("--test_ratio", type=float, default=0.10)
    p.add_argument("--seed",      type=int,   default=0)

    # batching
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--seq_len",     type=int, default=32,
                   help="window length used only when --windowed is set")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="lower for high-dim patch caches: in-flight batches = "
                        "workers * prefetch, ~0.5GB each at 10368-d")
    p.add_argument("--full_episode", dest="full_episode", action="store_true",
                   default=True,
                   help="train on full episodes from step 0 (default)")
    p.add_argument("--windowed", dest="full_episode", action="store_false",
                   help="use the old random fixed-length window training")

    # optimisation
    p.add_argument("--max_steps",   type=int,   default=200000)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--kl_warmup",   type=int,   default=20000,
                   help="steps over which kl_scale ramps from 0 -> 1")
    p.add_argument("--log_every",   type=int, default=50)
    p.add_argument("--test_every",  type=int, default=250, help="test全量评估间隔(07-27: 原每50步吃~25%墙钟)")
    p.add_argument("--ckpt_every",  type=int, default=5000)

    # model — shared with RSSM
    p.add_argument("--encoder_dim", type=int, default=512)
    p.add_argument("--h_dim",       type=int, default=512)
    p.add_argument("--stoch",       type=int, default=32)
    p.add_argument("--classes",     type=int, default=32)
    p.add_argument("--free_nats",   type=float, default=0.0)
    p.add_argument("--kl_balance",  type=float, default=0.8,
                   help="DreamerV2-style: weight on dyn term, (1-kb) on rep")
    p.add_argument("--kl_scale",    type=float, default=1.0,
                   help="global scalar on the KL term")

    # transformer-specific
    p.add_argument("--d_model",     type=int, default=512)
    p.add_argument("--n_heads",     type=int, default=8)
    p.add_argument("--n_layers",    type=int, default=4)
    p.add_argument("--ffn_dim",     type=int, default=2048)
    p.add_argument("--transformer_dropout", type=float, default=0.1)
    p.add_argument("--max_seq_len", type=int, default=2048)

    # TransDreamer-style aggregations
    p.add_argument("--deter_type",  type=str, default="last",
                   choices=["last", "concat_o"],
                   help="last = TSSM v4 default; concat_o = concat all transformer "
                        "layer outputs as deter (TransDreamer modules_transformer.py:295)")
    p.add_argument("--recon_type",  type=str, default="mse",
                   choices=["mse", "normal"],
                   help="mse = per-feature mean MSE; normal = -log_prob of "
                        "Independent Normal(recon, rec_sigma) (TransDreamer-style)")
    p.add_argument("--rec_sigma",   type=float, default=0.3,
                   help="Gaussian sigma used when --recon_type normal")

    # imagination loss: prior(h_t) -> z -> decoder -> obs_t MSE
    p.add_argument("--imagination_loss_weight", type=float, default=1.0)
    p.add_argument("--imagination_horizon",     type=int,   default=4)
    p.add_argument("--imagination_n_starts",    type=int,   default=8)
    # 多步开环未来预测 (07-25 predict-future线)
    p.add_argument("--future_weight",   type=float, default=0.0)
    p.add_argument("--future_stopgrad_decoder", action="store_true")
    p.add_argument("--nce_lag_fourier", action="store_true")
    p.add_argument("--nce_lag_sample", type=int, default=0)
    p.add_argument("--nce_grade_tau", type=float, default=0.0)
    p.add_argument("--slow_modules", type=str, default="")
    p.add_argument("--nce_head_hidden", type=int, default=512)
    p.add_argument("--nce_head_layers", type=int, default=3)
    p.add_argument("--nce_dim", type=int, default=256)
    p.add_argument("--nce_event_mode", action="store_true")
    p.add_argument("--nce_event_quantile", type=float, default=0.75)
    p.add_argument("--future_horizon",  type=int,   default=4)
    p.add_argument("--future_n_starts", type=int,   default=8)
    p.add_argument("--recon_weight",    type=float, default=1.0)
    # 反向预测 (双向自预测, Tang et al. ICML23)
    p.add_argument("--retro_weight",   type=float, default=0.0)
    p.add_argument("--nce_weight", type=float, default=0.0)
    p.add_argument("--nce_lags", type=str, default="3,5,10")
    p.add_argument("--fut_nce_weight", type=float, default=0.0)
    p.add_argument("--fut_nce_steps", type=int, default=10)
    p.add_argument("--fut_nce_lambda", type=float, default=0.75)
    p.add_argument("--fut_nce_starts", type=int, default=4)
    p.add_argument("--fut_nce_hstart", type=int, default=0)
    p.add_argument("--fut_nce_spec_lags", type=str, default="")
    p.add_argument("--attn_window", type=int, default=0)
    p.add_argument("--core_type", type=str, default="transformer")
    p.add_argument("--fut_nce_raw", action="store_true", help="TWISTER字面版裸点积(实测不稳, 仅消融)")
    p.add_argument("--feas_labels", type=str, default="")
    p.add_argument("--feas_weight", type=float, default=0.0)
    p.add_argument("--trav_weight", type=float, default=0.0)
    p.add_argument("--nce_tau", type=float, default=0.1)
    p.add_argument("--raw_fwd_weight", type=float, default=0.0)
    p.add_argument("--raw_retro_weight", type=float, default=0.0)
    p.add_argument("--raw_lags", type=str, default="3,5,10")
    p.add_argument("--retro_lags",     type=str,   default="1,2,5,10,20")
    p.add_argument("--retro_n_starts", type=int,   default=4)
    p.add_argument("--fwd_weight",     type=float, default=0.0)
    p.add_argument("--fwd_lags",       type=str,   default="1,2,5,10,20")
    p.add_argument("--fwd_n_starts",   type=int,   default=4)
    p.add_argument("--latent_weight",   type=float, default=0.0)
    p.add_argument("--latent_lags",     type=str,   default="5,10,20")
    p.add_argument("--latent_n_starts", type=int,   default=4)
    p.add_argument("--aux_actx",        type=str,   default="mean", choices=["mean","gru"])
    p.add_argument("--posterior_ctx",   type=str, default="myopic", choices=["myopic","transformer"])
    p.add_argument("--post_ctx_layers", type=int, default=2)
    p.add_argument("--fast_aux",       action="store_true", help="retro/fwd 向量化(数学等价, 约1.9x)")

    # WAM-style inverse action auxiliary loss.
    p.add_argument("--action_inv_weight",     type=float, default=0.0)
    p.add_argument("--action_inv_hidden_dim", type=int,   default=512)
    p.add_argument("--action_inv_layers",     type=int,   default=3)

    # Pose-delta auxiliary loss from h_{t+1} -> [forward, yaw].
    p.add_argument("--pose_delta_weight",     type=float, default=0.0)
    p.add_argument("--pose_delta_hidden_dim", type=int,   default=512)
    p.add_argument("--pose_delta_layers",     type=int,   default=3)

    # Non-pooled patch-grid cache (v5): see TSSMConfig docstrings.
    p.add_argument("--recon_sem_dim", type=int, default=0)
    p.add_argument("--recon_sem_boost", type=float, default=1.0)
    p.add_argument("--recon_groups", type=int,   default=1,
                   help="patch blocks per obs; normal recon = mean over blocks")
    p.add_argument("--feat_scale",   type=float, default=1.0,
                   help="divide raw features by this inside the model (lm2 magnitudes)")

    train(p.parse_args())

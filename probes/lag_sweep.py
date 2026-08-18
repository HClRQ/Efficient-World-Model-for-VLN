"""逐lag检索命中扫描 (08-04): 记忆保持曲线 + 插值判决.

对每个ckpt, 在heldout集上逐 k=1..30 用模型自己的NCE前向算检索命中:
  - 查表型(dense30/ncev5): 把 nce_lag_emb 临时钉到 k 对应的训练行;
  - fourier型: encode_lag(k) 原生支持任意k(含未训的洞 7/13/19/26)。
判读: fourier 臂在洞上的命中 ≈ 邻档 → 插值成立; 塌到随机 → 不成立。
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from navigation_tssm_feat import NavigationTSSM, TSSMConfig  # noqa: E402
from data_navila import build_episode_trajectories, split_episodes  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True, help="name=path,...")
    p.add_argument("--feat_cache", required=True)
    p.add_argument("--annotation", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_eps", type=int, default=60)
    p.add_argument("--min_len", type=int, default=32)
    p.add_argument("--max_lag", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fpath = os.path.join(args.feat_cache, "features.pt")
    features = torch.load(fpath if os.path.exists(fpath) else args.feat_cache,
                          map_location="cpu", weights_only=True)
    episodes = build_episode_trajectories(args.annotation)
    _, te = split_episodes(episodes, val_ratio=0.10, seed=0)
    import random
    random.seed(args.seed)
    pool = [e for e in te if e.episode_id in features
            and len(e.actions) >= args.min_len
            and features[e.episode_id].shape[0] == len(e.actions)]
    sample = random.sample(pool, min(args.n_eps, len(pool)))
    print(f"heldout sweep eps={len(sample)} (min_len={args.min_len})", flush=True)

    def flat(f):
        return f.reshape(f.shape[0], -1) if f.dim() == 3 else f

    results = {}
    for spec in args.ckpts.split(","):
        name, path = spec.split("=", 1)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = TSSMConfig(**{k: v for k, v in blob["config"].items()
                            if k in TSSMConfig.__dataclass_fields__})
        model = NavigationTSSM(cfg).to(device).eval()
        model.load_state_dict(blob["model"])
        trained_lags = [int(x) for x in str(cfg.nce_lags).split(",") if x.strip()]
        fourier = bool(getattr(cfg, "nce_lag_fourier", False))
        orig_emb = model.nce_lag_emb
        accs = {}
        with torch.no_grad():
            for k in range(1, args.max_lag + 1):
                if not fourier:
                    if k not in trained_lags:
                        continue                              # 查表型只能查训过的档
                    row = trained_lags.index(k)
                    _shim = (lambda r: (lambda idx: orig_emb(
                        torch.full_like(idx, r))))(row)
                    object.__setattr__(model, "nce_lag_emb", _shim)
                model.config.nce_lags = str(k)
                cfg.nce_lags = str(k)
                num = den = 0.0
                for e in sample:
                    F_ = flat(features[e.episode_id]).float().unsqueeze(0).to(device)
                    A = torch.tensor(e.actions, dtype=torch.long).unsqueeze(0).to(device)
                    out = model(F_, A, mask=torch.ones(1, F_.shape[1],
                                device=device, dtype=torch.bool))
                    acc = float(out.loss_dict.get("nce_acc", float("nan")))
                    if acc != acc:
                        continue
                    num += acc
                    den += 1
                if "nce_lag_emb" in model.__dict__:
                    del model.__dict__["nce_lag_emb"]
                accs[k] = num / max(den, 1)
        results[name] = (accs, set(trained_lags), fourier)
        hole = {7, 13, 19, 26}
        line = " ".join(f"{k}:{v:.2f}{'*' if k in hole else ''}"
                        for k, v in sorted(accs.items()))
        print(f"{name} (fourier={fourier}): {line}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, (accs, tl, fr) in results.items():
        ks = sorted(accs)
        ax.plot(ks, [accs[k] for k in ks], "o-", ms=4, label=name)
        if fr:
            holes = [k for k in ks if k in {7, 13, 19, 26}]
            ax.scatter(holes, [accs[k] for k in holes], s=120, facecolors="none",
                       edgecolors="red", zorder=5, label=f"{name} 未训档(插值测试)")
    ax.set_xlabel("lag k (steps back)"); ax.set_ylabel("retrieval acc (heldout)")
    ax.set_title("memory retention curve / interpolation test")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    os.makedirs(args.out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "lag_sweep.png"), dpi=150)
    print(f"saved {args.out_dir}/lag_sweep.png", flush=True)


if __name__ == "__main__":
    main()

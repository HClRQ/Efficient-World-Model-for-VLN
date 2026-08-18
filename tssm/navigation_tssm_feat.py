"""
NavigationTSSM (Transformer State-Space Model, feature-input variant) — v4.

================================================================
v4 (this file): port of TransDreamer (Chen et al. 2022) world-model design.

Why v3 failed:
  v3 token was (enc(obs_{t-1}), a_{t-1}); since obs_{t-1} ≈ obs_t for
  consecutive nav frames, the transformer learned obs autoregression and
  h_t became an obs_t predictor.  Both prior(h) and post(h, obs) then
  agreed without z carrying any info → KL pinned at free_nats floor for
  all 200k steps, z fully collapsed.  The fix1 (decoder z-only) and fix2
  (imagination loss = decode prior z to obs) made it worse: fix1 made h
  redundant; fix2 explicitly trained prior(h) → obs decoding.

  v4 design (matches TransDreamer's TSSM):
      token_t           = Linear(concat(z_flat_{t-1}, action_emb_{t-1}))
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          no obs in the token, ever — this is the key.
      h_t               = transformer(token_0..token_t)   [causal]
                        = "deterministic state from (z_{<t}, a_{<t})"
      prior(h_t)        -> z_t logits                      # has no obs info
      posterior(enc_t)  -> z_t logits                      # MYOPIC: obs only
      decoder(h_t, z_t) -> reconstruct obs_t               # z must carry obs

Why this avoids z-collapse:
  h_t never sees any obs (only past z's, which are themselves bottlenecked
  through 32×32 categoricals).  So decoder(h, z) -> obs_t REQUIRES z to
  carry obs_t — the bottleneck is structural, not metric.  This is the
  same reason RSSM doesn't collapse: in RSSM h = GRU(h, z, a) only sees
  obs through z.

KL balancing (DreamerV2 form):
    rep_kl  = KL(q(z|x), sg[p(z|h)])              # representation term
    dyn_kl  = KL(sg[q(z|x)], p(z|h))              # dynamics term
    kl_loss = (1 - kl_balance) * rep_kl + kl_balance * dyn_kl
    Most pressure (0.8) on dyn so prior catches up to posterior, with
    posterior held mostly free → encourages z to use its capacity.

What we drop from v3:
  - imagination loss in world-model loss (TransDreamer doesn't use it
    here — it lives in actor-critic training, which we don't need for
    behavior-cloning navigation cache).
  - state.last_obs_emb (no longer needed: token uses z_flat, which is
    always available in state.z).
  - dyn_weight/rep_weight (replaced by single kl_balance).

Cache state_dim depends on deter_type:
  - last:     h_dim + stoch*classes = 1536
  - concat_o: n_layers*d_model + stoch*classes = 3072

References:
    Chen et al. 2022, "TransDreamer: Reinforcement Learning with
        Transformer World Models", arXiv:2202.09481
      Hafner et al. 2022, "Mastering Atari with Discrete World Models" (DreamerV2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class TSSMConfig:
    """TSSM v4 (TransDreamer-style) configuration."""
    # ---- 观测编码 ----
    feat_dim: int = 1152
    encoder_dim: int = 512

    # ---- 状态维度 ----
    h_dim: int = 512
    stoch: int = 32
    classes: int = 32

    # ---- Transformer core ----
    # Reverted to the proven 17M-scale config (May 8 successful run).  The
    # 768/12/6/3072 attempt with batch=32 + lr=2e-4 overfit dynamics (test/dyn
    # ~1.07 vs prior best 0.83) and is documented in tssm_full_ep_66958478.
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    ffn_dim: int = 2048
    transformer_dropout: float = 0.1
    max_seq_len: int = 2048

    # ---- 动作 ----
    num_actions: int = 10
    action_embed_dim: int = 64

    # ---- DreamerV3 stochastic-latent tricks ----
    unimix: float = 0.01
    free_nats: float = 0.0           # TransDreamer atari config

    # ---- DreamerV2-style KL balancing ----
    kl_balance: float = 0.8          # weight on dyn term; (1-kl_balance) on rep
    kl_scale: float = 1.0            # global scale on full KL term

    # ---- MLP depth ----
    num_prior_layers: int = 2
    num_posterior_layers: int = 2
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2

    # ---- Deterministic state aggregation (TransDreamer modules_transformer.py:295) ----
    #   "last"     — use last transformer layer output (legacy v4 behaviour)
    #   "concat_o" — concat ALL layer outputs in channel dim; effective deter
    #                dim becomes n_layers * d_model.  Matches TransDreamer's
    #                Atari config (deter_type='concat_o').
    deter_type: str = "last"

    # ---- Reconstruction (TransDreamer-style Normal log-prob) ----
    #   "mse"    — legacy: per-feature mean MSE (low absolute scale)
    #   "normal" — -log_prob of Independent Normal(recon, rec_sigma); equals
    #              0.5 * sum_i ((feat-recon)/sigma)^2 + const.  TransDreamer
    #              Atari uses sigma=0.3 with sum over feature dim.
    recon_type: str = "mse"
    rec_sigma: float = 0.3

    # ---- Imagination loss in world-model loss (TransDreamer: 0) ----
    imagination_loss_weight: float = 0.0
    imagination_horizon: int = 4
    imagination_n_starts: int = 8
    imagination_stop_grad_encoder: bool = True

    future_stopgrad_decoder: bool = False   # 08-04: future分支冻解码器参数(防投毒), 梯度只进核心
    # ---- multi-step OPEN-LOOP future feat prediction (07-25 predict-future line) ----
    # From n_starts positions, roll the PRIOR forward K steps with REAL actions
    # (no future obs), decode, match real future SigLIP feat. Frozen-SigLIP target
    # => no collapse (DINO-WM/NE-Dreamer style). Grad-enabled rollout.
    future_weight: float = 0.0        # >0 turns it on
    future_horizon: int = 4           # K open-loop steps
    future_n_starts: int = 8          # random start positions / sequence
    recon_weight: float = 1.0         # downweight current-recon vs future

    # ---- BACKWARD / retro prediction (07-26, 用户提议; 依据 Tang et al. ICML23
    # "Understanding Self-Predictive Learning": 纯前向自预测只做转移矩阵的谱分解,
    # 只捕获部分特征方向; 双向(前向+反向)有更一般的不塌缩保证、捕获互补方向。
    # 我们实测正对上: 前向 RETR_IN 0.986(强), 向后 ΔR²≈0.02(弱) —— 只训了前向那半个谱。
    # 实现: 从 state_t 预测**过去** lag k 的观测编码 enc_{t-k}(动作条件+多滞后),
    # 目标 stop-grad(防塌, 照 semi-gradient 家法)。论文 α=1 等权, 我们默认给同权。
    retro_weight: float = 0.0         # >0 turns it on (论文 α=1)
    retro_lags: str = "1,2,5,10,20"   # 多滞后: 逼长程保留(1步太弱)
    retro_n_starts: int = 4
    # v9 (07-26): 严格中介前向预测 —— state-only MLP 头预测 enc_{t+k}, 不重跑注意力。
    # 动机: v7 的想象循环每步可回看原始历史 token(NextLat 所谓 shortcut/历史旁路),
    # "state 必须概括历史"的压力从未存在; v8 retro 无旁路故有效。前向镜像 retro 堵旁路。
    # ⚠ 必须多滞后: 一步前向在导航里近乎免费(连续帧高度相似), 单 k=1 会秒降到 0。
    fwd_weight: float = 0.0           # >0 turns it on
    fwd_lags: str = "1,2,5,10,20"
    fwd_n_starts: int = 4
    # ---- v9-latent (07-26 fable复核后): 隐空间转移一致性 ----
    # fwd_head 预测的是**观测编码** enc_{t+k}, 属"带state瓶颈的DINO-WM式解码";
    # NextLat Thm 3.2 的前提是**隐空间**转移一致性 p(h_{t+1}|h_t,x_{t+1}), 观测空间版
    # 撑不起该定理(只能作 motivation 引)。latent 头直接预测 sg[state_{t+k}],
    # 与 retro(向后)合成 Tang et al. ICML23 的**双向隐空间自预测**, 不塌缩保证最强。
    # ⚠ 目标是网络自产量, 无 recon 锚保护 -> 塌缩风险高于 retro, 必须 sg + 监控 lat_std。
    # ⚠ lags 去掉 1,2: 前向短滞后近乎马尔可夫(下一帧≈当前帧+动作), 不产生记忆压力
    #    却占采样预算(fable复核②)。
    latent_weight: float = 0.0
    latent_lags: str = "5,10,20"
    latent_n_starts: int = 4
    # 动作上下文编码: mean 丢序("左转再直走"与"直走再左转"相同), k=20 时几乎只剩
    # "直行为主/转弯为主"一个标量 -> 任务不可约误差巨大, 记忆梯度信噪比被压扁(fable复核③)。
    aux_actx: str = "mean"    # mean | gru(有序, 单层GRU)
    fast_aux: bool = False    # retro/fwd 向量化(数学等价, RNG流不同); 默认关以免影响在跑的臂
    # ---- v11 (07-27): history-conditioned posterior, TransDreamer 原文点名但放弃的设计 ----
    # q(z_t | f_transformer(x_1:t, a_1:t)) —— posterior 走自己的因果观测塔。观测是给定的
    # (非采样), 无递归、训练全并行。z 从"孤帧快照"变"信念码"; KL 的靶子含历史后,
    # prior(h)→z 首次真正逼 h 编码历史(直击 §10 "state≈当前帧预测器"病根)。
    # 想象路径(prior→动力学塔)零改动。myopic = v4/原文默认。
    posterior_ctx: str = "myopic"     # myopic | transformer
    post_ctx_layers: int = 2
    post_ctx_heads: int = 4

    # ---- WAM-style inverse action auxiliary loss ----
    # Predict a_t from consecutive encoder embeddings [e_t, e_{t+1}].
    # Keep this off by default so old checkpoints instantiate without the head.
    action_inv_weight: float = 0.0
    action_inv_hidden_dim: int = 512
    action_inv_layers: int = 3

    # ---- Pose-delta auxiliary loss ----
    # Predict the local command delta carried by a_t from h_{t+1}. Targets are
    # normalized as [forward_m / 0.75, yaw_rad / 45deg].
    pose_delta_weight: float = 0.0
    pose_delta_hidden_dim: int = 512
    pose_delta_layers: int = 3

    # ---- Non-pooled patch-grid observations (v5) ----
    # recon_groups: number of patch blocks concatenated into feat_dim (e.g. 9
    #   for the p9 cache). "normal" recon divides by this so the loss is the
    #   MEAN over patches, preserving the recon:KL balance tuned at 1152.
    # feat_scale: raw features are divided by this inside the model. lm2
    #   (hidden_states[-2], no post_layernorm) features have much larger
    #   magnitude than post-LN features; scaling keeps decoder outputs and
    #   rec_sigma in the regime the recipe was tuned for. Stored in ckpt
    #   config — downstream decode must multiply back (see decode_feats_raw).
    recon_groups: int = 1
    # 07-28 分段配平: 拼接特征(如 siglip1152|vggt2048)时, 前 sem_dim 维的重建误差
    # 乘 sem_boost, 抵消几何段维数优势(parity boost = geo_dim/sem_dim)。0=关。
    recon_sem_dim: int = 0
    recon_sem_boost: float = 1.0
    # 07-28 biraw(用户拍板): 双向**原始特征**回归专线(绕过enc, 治语义历史饿死)。
    # 尊重TransDreamer原框架: 纯外挂辅助头, 不触碰core/posterior/prior/recon。
    raw_fwd_weight: float = 0.0
    raw_retro_weight: float = 0.0
    raw_lags: str = "3,5,10"
    # 07-29 InfoNCE-retro(用户拍板): 判别式历史检索目标——同集帧作困难负例,
    # 均值先验零得分, 唯一得分方式=z编码"时刻身份"。日志自带留出可比的top1命中率。
    nce_weight: float = 0.0
    nce_lags: str = "3,5,10"
    nce_dim: int = 256
    nce_tau: float = 0.1
    nce_lag_fourier: bool = False   # 08-04: lag编码=k的连续函数(正弦16维→MLP64), 支持插值查询
    nce_lag_sample: int = 0          # 08-05 graded30: >0则每步随机抽N档考(治梯度打架)
    nce_grade_tau: float = 0.0       # >0则按exp(-k/tau)分级加权(近档精读远档略读)
    slow_modules: str = ""           # 08-05 cw30快慢双状态: "4:256,16:256"=周期:维度; 空=关
    nce_head_hidden: int = 512       # 08-05 big30: 检索头宽度
    nce_head_layers: int = 3         # 检索头深度
    nce_event_mode: bool = False     # 08-05 evt30: 事件记忆考纲(惊奇锚定, 替代固定lag)
    nce_event_quantile: float = 0.75 # 集内惊奇分位阈 → 事件帧
    nce_event_ranks: str = "1,2,3"   # 考"最近第m个事件帧"
    # 08-06 futNCE→TWISTER对齐(ICLR25 AC-CPC, 用户令"认真对齐结构"):
    # 视距t=0..steps-1全档; 每档独立(查询,键)双MLP; 动作有序串接进查询(顺序天然保);
    # 负例=batch×时间全局展平(跨集); 裸点积无温度; 权重0.75^t衰减归一。
    # 对TWISTER的四处有记录偏差: ①锚=后验state(LLM吃的就是它) ②键=冻结SigLIP过MLP
    # (特征预算好无法图像增广) ③每档锚点采样fut_nce_starts个(算力, 他们全位置)
    # ④总权重对齐我们retro的8而非他家1.0(两家recon尺度不可比)。
    fut_nce_weight: float = 0.0
    fut_nce_steps: int = 10          # 视距0..9(含t=0当下对齐档, TWISTER同款)
    fut_nce_lambda: float = 0.75     # 视距指数衰减, 内部归一到总和1
    fut_nce_starts: int = 4          # 每档锚点采样数
    fut_nce_hstart: int = 0          # 08-07用户定案: 合流版起考视距=1(TWISTER的t=0靠增广
                                     # 才不平凡, 我们冻结特征无增广→t=0是送分题白吃27%预算);
                                     # 默认0保持已训ckpt兼容, 新臂显式传1
    fut_nce_spec_lags: str = ""      # 08-07: 专精考纲模式(镜像retro哲学)。设"3,5,10"则只考
                                     # 这些视距、等权重, 覆盖steps/lambda/hstart(dense0.6 vs
                                     # 专精0.95的分配定律, 未来侧从未验过——补历史盲区)
    core_type: str = "transformer"   # 08-08 真·RSSM档: "gru"=GRU核(Dreamer式递归,
                                     # h_t=GRU(h_{t-1},[z_{t-1},a_{t-1}])), 躯干其余全同
    attn_window: int = 0             # 08-08 递归孪生实验: >0则全部注意力限带宽W(W=1≈纯递归/
                                     # RSSM信息拓扑, 信息只能逐token转抄); 0=全因果(TSSM本尊)。
                                     # 训练期生效=trained-in-regime对照, 回答"为何不需要recurrent"
    # 08-07 稳定性修复(实测踩坑): TWISTER原版用裸点积(无归一化无温度), 在Atari的小
    # 特征尺度上没事; 我们的键来自10368维SigLIP、查询来自3072维state, MLP输出范数
    # 无界 → logits爆炸: 11k步内 fut_nce 208→2064 而命中原地踏步0.25, 且占总损失97%
    # 反噬主模型(retro命中0.60→0.53, recon 49.3→50.6)。改为与retro头同款 L2归一化+
    # 温度τ(logits夹在±1/τ)。置False可复现TWISTER字面版(仅供消融/复现性说明)。
    fut_nce_norm: bool = True
    # 08-07 可行性/可通行性辅助头(navmesh离线标签, mine_feasibility_labels.py):
    # feas=前进25/50/75cm三位"通/阻"; trav=相对朝向12方向×3半径的局部地板图36位。
    # 监督=评测同款try_step/is_navigable, 终点判官过闸(R2R丢0.1%/RxR丢1.3%)。
    feas_weight: float = 0.0
    trav_weight: float = 0.0

    feat_scale: float = 1.0


# ===========================================================================
# Shared blocks
# ===========================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


def make_causal_mask(T: int, device, window: int = 0) -> torch.Tensor:
    """True=禁止. window>0时叠加带宽限制(dist>window禁): W=1即递归类拓扑."""
    m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    if window and window > 0:
        idx = torch.arange(T, device=device)
        m = m | ((idx.unsqueeze(1) - idx.unsqueeze(0)) > window)
    return m


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(num_layers):
        inp = in_dim if i == 0 else hidden_dim
        out = out_dim if i == num_layers - 1 else hidden_dim
        layers.append(nn.Linear(inp, out))
        if i < num_layers - 1:
            layers.append(RMSNorm(out))
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class FeatEncoder(nn.Module):
    def __init__(self, cfg: TSSMConfig):
        super().__init__()
        self.in_norm = nn.LayerNorm(cfg.feat_dim)
        self.mlp = build_mlp(
            cfg.feat_dim, cfg.encoder_dim, cfg.encoder_dim, cfg.num_encoder_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.in_norm(x))


class FeatDecoder(nn.Module):
    """concat(h, z) -> reconstructed SigLIP feat at the SAME step.

    v4 takes BOTH h and z again (reverting v3's z-only fix).  Safe now
    because v4's h never sees obs, so decoder cannot shortcut through h.
    """

    def __init__(self, cfg: TSSMConfig, deter_dim: int, extra_dim: int = 0):
        super().__init__()
        in_dim = deter_dim + cfg.stoch * cfg.classes + extra_dim
        self.mlp = build_mlp(
            in_dim, cfg.encoder_dim, cfg.feat_dim, cfg.num_decoder_layers
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)


class StochasticEncoder(nn.Module):
    def __init__(self, in_dim: int, config: TSSMConfig, num_layers: int):
        super().__init__()
        self.config = config
        out_dim = config.stoch * config.classes
        self.net = build_mlp(in_dim, config.h_dim, out_dim, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.reshape(*logits.shape[:-1], self.config.stoch, self.config.classes)


# ===========================================================================
# Transformer core — v4: tokens are (z_flat, action_emb), NOT (obs, action)
# ===========================================================================

def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerCore(nn.Module):
    """Causal self-attention over (z_flat_{t-1}, action_emb_{t-1}) tokens.

    The caller is responsible for shifting z and action by one step before
    handing them in (so token at position t carries info up to step t-1).
    h_t output therefore has no obs_t leak — and no obs leak at all,
    since obs is never an input to this transformer.
    """

    def __init__(self, cfg: TSSMConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.stoch * cfg.classes + cfg.action_embed_dim
        self.input_proj = nn.Linear(in_dim, cfg.d_model)
        self.register_buffer("pos_emb", _sinusoidal_pe(cfg.max_seq_len, cfg.d_model),
                             persistent=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        # h_proj only used when deter_type == "last"; for "concat_o" we keep
        # the raw per-layer concat (matches TransDreamer modules_transformer.py:423).
        if cfg.deter_type == "concat_o":
            self.deter_dim = cfg.n_layers * cfg.d_model
            self.h_proj = None
        elif cfg.deter_type == "last":
            self.deter_dim = cfg.h_dim
            self.h_proj = nn.Linear(cfg.d_model, cfg.h_dim)
        else:
            raise ValueError(f"unknown deter_type: {cfg.deter_type}")

    def _build_tokens(self, z_flat: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_flat, action_emb], dim=-1)
        return self.input_proj(x)

    def _run_layers(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run all transformer encoder layers with a causal mask.

        Returns deter according to cfg.deter_type:
          - "last":      [B, T, d_model]               (last layer output)
          - "concat_o":  [B, T, n_layers * d_model]    (concat all layers)
        """
        if self.cfg.deter_type == "concat_o":
            outs = []
            for layer in self.transformer.layers:
                x = layer(x, src_mask=mask)
                outs.append(x)
            if self.transformer.norm is not None:
                outs[-1] = self.transformer.norm(outs[-1])
            return torch.cat(outs, dim=-1)
        # deter_type == "last" -> rely on PyTorch's optimised path
        return self.transformer(x, mask=mask)

    def forward_seq(
        self,
        shifted_z_seq: torch.Tensor,      # [B, T, stoch*classes]   (z_{t-1} at pos t)
        shifted_action_seq: torch.Tensor, # [B, T, action_emb]      (a_{t-1} at pos t)
    ) -> torch.Tensor:
        """Parallel forward.  h_seq[t] is the deterministic state derived
        from (z_{<t}, a_{<t}); has no obs information.

        Output shape depends on cfg.deter_type:
          - "last":     [B, T, h_dim]
          - "concat_o": [B, T, n_layers * d_model]
        """
        B, T, _ = shifted_z_seq.shape
        assert T <= self.cfg.max_seq_len
        device = shifted_z_seq.device

        tokens = self._build_tokens(shifted_z_seq, shifted_action_seq)
        x = tokens + self.pos_emb[:T].unsqueeze(0)
        causal = make_causal_mask(T, device, getattr(self.cfg, "attn_window", 0))
        out = self._run_layers(x, causal)
        if self.h_proj is not None:
            out = self.h_proj(out)
        return out

    @torch.no_grad()
    def step_with_history(
        self,
        z_flat_prev: torch.Tensor,         # [B, stoch*classes]   (z_{t-1})
        action_emb_prev: torch.Tensor,     # [B, action_emb]      (a_{t-1})
        history_tokens: torch.Tensor,      # [B, t, d_model]      past projected tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Online single step.  Caller passes the SHIFTED inputs (z and a
        from previous step); returns h at the new last position."""
        device = z_flat_prev.device
        new_token = self._build_tokens(z_flat_prev, action_emb_prev).unsqueeze(1)
        new_history = torch.cat([history_tokens, new_token], dim=1)
        T = new_history.shape[1]
        assert T <= self.cfg.max_seq_len
        x = new_history + self.pos_emb[:T].unsqueeze(0)
        causal = make_causal_mask(T, device, getattr(self.cfg, "attn_window", 0))
        out = self._run_layers(x, causal)
        h_last = out[:, -1]
        if self.h_proj is not None:
            h_last = self.h_proj(h_last)
        return h_last, new_history


class GRUCore(nn.Module):
    """真·RSSM核 (08-08): h_t = GRU(h_{t-1}, proj(z_{t-1}, a_{t-1}))。
    与TransformerCore同接口同deter_dim(=n_layers*d_model), 躯干其余零改动。
    history槽位复用为GRU隐藏态[B,1,deter_dim]。"""

    def __init__(self, cfg: TSSMConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.stoch * cfg.classes + cfg.action_embed_dim
        self.input_proj = nn.Linear(in_dim, cfg.d_model)
        self.deter_dim = cfg.n_layers * cfg.d_model      # 与concat_o口径对齐(2048)
        self.gru = nn.GRU(cfg.d_model, self.deter_dim, batch_first=True)
        self.h_proj = None
        # 兼容占位(imagination旧路径引用, 本档不启用)
        self.register_buffer("pos_emb", _sinusoidal_pe(cfg.max_seq_len, cfg.d_model),
                             persistent=False)

    def _build_tokens(self, z_flat: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        return self.input_proj(torch.cat([z_flat, action_emb], dim=-1))

    def forward_seq(self, shifted_z_seq: torch.Tensor,
                    shifted_action_seq: torch.Tensor) -> torch.Tensor:
        tokens = self._build_tokens(shifted_z_seq, shifted_action_seq)   # [B,T,d]
        h_seq, _ = self.gru(tokens)                                       # [B,T,deter]
        return h_seq

    def step_with_history(self, z_flat_prev, action_emb_prev, history_tokens):
        tok = self._build_tokens(z_flat_prev, action_emb_prev).unsqueeze(1)  # [B,1,d]
        if history_tokens is None or history_tokens.shape[1] == 0 \
                or history_tokens.shape[-1] != self.deter_dim:
            hid = torch.zeros(1, tok.shape[0], self.deter_dim,
                              device=tok.device, dtype=tok.dtype)
        else:
            hid = history_tokens[:, -1].unsqueeze(0).contiguous()        # [1,B,deter]
        out, hid_new = self.gru(tok, hid)
        h_last = out[:, -1]
        return h_last, hid_new.transpose(0, 1)                            # [B,1,deter]


@dataclass
class TSSMState:
    """TSSM v4 state.

    `z` is the current step's posterior (or prior, in imagination) sample;
    it doubles as the obs slot of the NEXT token (replacing v3's
    `last_obs_emb`).  `history` holds past projected transformer tokens.
    """
    h: torch.Tensor                              # [B, h_dim]
    z: torch.Tensor                              # [B, stoch, classes]
    history: Optional[torch.Tensor] = None       # [B, t, d_model]
    ctx_hist: Optional[torch.Tensor] = None      # [B, t, enc] v11 观测塔 token 缓冲(仅 posterior_ctx=transformer)
    r: Optional[torch.Tensor] = None             # [B, S] cw30 慢寄存器(clockwork保持)
    tick: int = 0                                # 步计数(慢寄存器更新拍判定)

    @property
    def z_flat(self) -> torch.Tensor:
        return self.z.reshape(self.z.shape[0], -1)

    @property
    def full(self) -> torch.Tensor:
        parts = [self.h, self.z_flat]
        if self.r is not None:
            parts.append(self.r)
        return torch.cat(parts, dim=-1)

    def detach(self) -> "TSSMState":
        return TSSMState(
            h=self.h.detach(),
            z=self.z.detach(),
            history=self.history.detach() if self.history is not None else None,
            r=self.r.detach() if self.r is not None else None,
            tick=self.tick,
        )


@dataclass
class TSSMOutput:
    loss: torch.Tensor
    loss_dict: Dict[str, float]
    states: List[TSSMState]
    prior_logits: List[torch.Tensor] = field(default_factory=list)
    post_logits: List[torch.Tensor] = field(default_factory=list)


# ===========================================================================
# State container
# ===========================================================================


# ===========================================================================
# Main model
# ===========================================================================

class NavigationTSSM(nn.Module):
    """Transformer State-Space Model on NaVILA SigLIP features (v4)."""

    def __init__(self, config: Optional[TSSMConfig] = None):
        super().__init__()
        self.config = config or TSSMConfig()
        cfg = self.config

        self.z_dim_flat = cfg.stoch * cfg.classes
        self.encoder_dim = cfg.encoder_dim
        self.feat_dim = cfg.feat_dim
        self.d_model = cfg.d_model

        self.feat_encoder = FeatEncoder(cfg)
        self.action_embed = nn.Embedding(cfg.num_actions, cfg.action_embed_dim)
        self.core = (GRUCore(cfg) if getattr(cfg, "core_type", "transformer") == "gru"
                     else TransformerCore(cfg))

        # deter_dim is decided by the core (depends on deter_type).
        self.h_dim = self.core.deter_dim   # alias for legacy code paths
        self.deter_dim = self.core.deter_dim
        self.slow_specs = []                        # [(period, dim)]
        _sm = str(getattr(cfg, "slow_modules", "") or "").strip()
        if _sm:
            for part in _sm.split(","):
                K, D = part.split(":")
                self.slow_specs.append((int(K), int(D)))
        self.slow_total = sum(d for _, d in self.slow_specs)
        self.state_dim = self.deter_dim + self.z_dim_flat + self.slow_total

        # v4: prior takes only h_t (h is the past summary).
        self.prior = StochasticEncoder(self.deter_dim, cfg, cfg.num_prior_layers)
        # v4 (TransDreamer-style myopic posterior): obs only, no h.
        self.posterior = StochasticEncoder(
            cfg.encoder_dim, cfg, cfg.num_posterior_layers
        )

        # v4: decoder takes cat(h, z) again — safe because h has no obs leak.
        self.feat_decoder = FeatDecoder(cfg, deter_dim=self.deter_dim,
                                        extra_dim=self.slow_total)
        self.slow_proj = self.slow_gate = None
        if self.slow_total:
            self.slow_proj = nn.Linear(self.deter_dim, self.slow_total)
            self.slow_gate = nn.Linear(self.deter_dim, self.slow_total)
        self.action_inv_head = None
        if cfg.action_inv_weight > 0:
            self.action_inv_head = build_mlp(
                2 * cfg.encoder_dim,
                cfg.action_inv_hidden_dim,
                cfg.num_actions,
                cfg.action_inv_layers,
            )
        # 反向预测头(07-26): [state, lag_emb, action_ctx] -> enc_{t-k}
        self.retro_head = None
        self.retro_lag_emb = None
        if cfg.retro_weight > 0:
            _nlag = len([x for x in str(cfg.retro_lags).split(",") if x.strip()])
            self.retro_lag_emb = nn.Embedding(max(1, _nlag), 64)
            self.retro_head = build_mlp(
                self.deter_dim + self.z_dim_flat + 64 + cfg.action_embed_dim,
                cfg.projector_hidden if hasattr(cfg, "projector_hidden") else 512,
                cfg.encoder_dim, 3,
            )
        # v9 前向预测头(07-26): [state, lag_emb, action_ctx] -> enc_{t+k}, 镜像 retro_head
        self.fwd_head = None
        self.fwd_lag_emb = None
        if cfg.fwd_weight > 0:
            _nlag_f = len([x for x in str(cfg.fwd_lags).split(",") if x.strip()])
            self.fwd_lag_emb = nn.Embedding(max(1, _nlag_f), 64)
            self.fwd_head = build_mlp(
                self.deter_dim + self.z_dim_flat + 64 + cfg.action_embed_dim,
                cfg.projector_hidden if hasattr(cfg, "projector_hidden") else 512,
                cfg.encoder_dim, 3,
            )
        # 07-28 biraw: 双向原始特征回归头(目标=裸SigLIP特征, 非enc)
        self.raw_fwd_head = None
        self.raw_retro_head = None
        self.raw_lag_emb = None
        if cfg.raw_fwd_weight > 0 or cfg.raw_retro_weight > 0:
            _nlag_rw = len([x for x in str(cfg.raw_lags).split(",") if x.strip()])
            self.raw_lag_emb = nn.Embedding(max(1, _nlag_rw), 64)
            _rin = self.deter_dim + self.z_dim_flat + 64 + cfg.action_embed_dim
            if cfg.raw_fwd_weight > 0:
                self.raw_fwd_head = build_mlp(_rin, 512, cfg.feat_dim, 3)
            if cfg.raw_retro_weight > 0:
                self.raw_retro_head = build_mlp(_rin, 512, cfg.feat_dim, 3)

        # 07-29 InfoNCE-retro 头
        self.nce_q_head = None
        self.nce_k_head = None
        self.nce_lag_emb = None
        if cfg.nce_weight > 0:
            _nl = len([x for x in str(cfg.nce_lags).split(",") if x.strip()])
            if cfg.nce_lag_fourier:
                self.nce_lag_emb = build_mlp(16, 64, 64, 2)   # 输入=fourier(k)
            else:
                self.nce_lag_emb = nn.Embedding(max(1, _nl), 64)
            _slow_extra = sum(int(p.split(":")[1]) for p in
                              str(getattr(cfg, "slow_modules", "") or "").split(",")
                              if ":" in p)
            self.nce_q_head = build_mlp(
                self.deter_dim + self.z_dim_flat + _slow_extra + 64 + cfg.action_embed_dim,
                cfg.nce_head_hidden, cfg.nce_dim, cfg.nce_head_layers)
            self.nce_k_head = build_mlp(cfg.feat_dim, cfg.nce_head_hidden, cfg.nce_dim, cfg.nce_head_layers)

        # 08-06 futNCE→TWISTER对齐: 每视距t一对独立MLP(查询吃state+有序动作串接, 键吃帧特征)
        # TWISTER ContrastiveNetwork同构: hidden512/out512/2层, 每档input维=state+t*action_emb
        self.fut_q_mlps = None
        self.fut_k_mlps = None
        if cfg.fut_nce_weight > 0:
            _slow_extra_f = sum(int(p.split(":")[1]) for p in
                                str(getattr(cfg, "slow_modules", "") or "").split(",")
                                if ":" in p)
            _sd = self.deter_dim + self.z_dim_flat + _slow_extra_f
            if str(cfg.fut_nce_spec_lags).strip():
                _fh = [int(x) for x in str(cfg.fut_nce_spec_lags).split(",") if x.strip()]
            else:
                _fh = list(range(cfg.fut_nce_hstart, cfg.fut_nce_steps))
            self.fut_horizons = _fh                       # 普通属性, 不进state_dict
            self.fut_q_mlps = nn.ModuleList([
                build_mlp(_sd + t * cfg.action_embed_dim, 512, 512, 2)
                for t in _fh])
            self.fut_k_mlps = nn.ModuleList([
                build_mlp(cfg.feat_dim, 512, 512, 2)
                for t in _fh])

        # 08-07 可行性/可通行性头
        self.feas_head = None
        self.trav_head = None
        if cfg.feas_weight > 0 or cfg.trav_weight > 0:
            _slow_x = sum(int(p.split(":")[1]) for p in
                          str(getattr(cfg, "slow_modules", "") or "").split(",")
                          if ":" in p)
            _sdim = self.deter_dim + self.z_dim_flat + _slow_x
            if cfg.feas_weight > 0:
                self.feas_head = build_mlp(_sdim, 256, 3, 2)
            if cfg.trav_weight > 0:
                self.trav_head = build_mlp(_sdim, 256, 36, 2)

        # v11 观测塔(07-27): posterior 的因果上下文编码器, 吃 (enc_t, a_{t-1}) 序列。
        # 输入是给定观测(非采样) -> 无递归, 训练全并行。TransDreamer 原文 §3.1 点名
        # 但因复杂度放弃的设计 q(z_t | f_transformer(x_1:t, a_1:t))。
        self.post_ctx_in = None
        self.post_ctx = None
        self.post_ctx_pos = None
        if str(getattr(cfg, "posterior_ctx", "myopic")) == "transformer":
            self.post_ctx_in = nn.Linear(cfg.encoder_dim + cfg.action_embed_dim,
                                         cfg.encoder_dim)
            _pl = nn.TransformerEncoderLayer(
                d_model=cfg.encoder_dim, nhead=cfg.post_ctx_heads,
                dim_feedforward=cfg.encoder_dim * 2,
                dropout=cfg.transformer_dropout, batch_first=True,
                norm_first=True, activation="gelu")
            self.post_ctx = nn.TransformerEncoder(_pl, num_layers=cfg.post_ctx_layers)
            self.post_ctx_pos = nn.Parameter(
                torch.randn(cfg.max_seq_len, cfg.encoder_dim) * 0.02)
        # v9-latent 头(07-26): [state, lag_emb, 有序动作上下文] -> sg[state_{t+k}]
        self.latent_head = None
        self.latent_lag_emb = None
        _sdim = self.deter_dim + self.z_dim_flat
        if cfg.latent_weight > 0:
            _nlag_l = len([x for x in str(cfg.latent_lags).split(",") if x.strip()])
            self.latent_lag_emb = nn.Embedding(max(1, _nlag_l), 64)
            self.latent_head = build_mlp(
                _sdim + 64 + cfg.action_embed_dim,
                cfg.projector_hidden if hasattr(cfg, "projector_hidden") else 512,
                _sdim, 3,
            )
        # 有序动作上下文(可选): 单层 GRU, <0.1M 参数
        self.actx_gru = None
        if str(getattr(cfg, "aux_actx", "mean")) == "gru":
            self.actx_gru = nn.GRU(cfg.action_embed_dim, cfg.action_embed_dim,
                                   batch_first=True)
        self.pose_delta_head = None
        if cfg.pose_delta_weight > 0:
            self.pose_delta_head = build_mlp(
                self.deter_dim,
                cfg.pose_delta_hidden_dim,
                2,
                cfg.pose_delta_layers,
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # unimix + ST sample + KL
    # ------------------------------------------------------------------

    def _apply_unimix(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.unimix > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / self.config.classes
            probs = (1 - self.config.unimix) * probs + self.config.unimix * uniform
            logits = torch.log(probs + 1e-8)
        return logits

    def _sample_stochastic(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self._apply_unimix(logits)
        probs = F.softmax(logits, dim=-1)
        if self.training:
            sample = torch.distributions.Categorical(logits=logits).sample()
            one_hot = F.one_hot(sample, self.config.classes).to(probs.dtype)
            return one_hot + probs - probs.detach()
        else:
            indices = logits.argmax(dim=-1)
            return F.one_hot(indices, self.config.classes).to(probs.dtype)

    def _kl_divergence(self, logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
        lp = F.log_softmax(self._apply_unimix(logits_p), dim=-1)
        lq = F.log_softmax(self._apply_unimix(logits_q), dim=-1)
        p = lp.exp()
        kl = (p * (lp - lq)).sum(dim=-1)
        return kl.sum(dim=-1)

    def _action_pose_delta_target(self, actions: torch.Tensor) -> torch.Tensor:
        """Return normalized local pose delta targets for discrete nav actions.

        Shape: actions[...] -> [..., 2], where dim 0 is forward meters / 0.75
        and dim 1 is yaw radians / radians(45). Left turn is positive yaw.
        """
        target = actions.new_zeros(*actions.shape, 2).float()
        target[..., 0] = torch.where(actions == 1, 0.25 / 0.75, target[..., 0])
        target[..., 0] = torch.where(actions == 2, 0.50 / 0.75, target[..., 0])
        target[..., 0] = torch.where(actions == 3, 0.75 / 0.75, target[..., 0])
        target[..., 1] = torch.where(actions == 4, 15.0 / 45.0, target[..., 1])
        target[..., 1] = torch.where(actions == 5, 30.0 / 45.0, target[..., 1])
        target[..., 1] = torch.where(actions == 6, 45.0 / 45.0, target[..., 1])
        target[..., 1] = torch.where(actions == 7, -15.0 / 45.0, target[..., 1])
        target[..., 1] = torch.where(actions == 8, -30.0 / 45.0, target[..., 1])
        target[..., 1] = torch.where(actions == 9, -45.0 / 45.0, target[..., 1])
        return target

    # ------------------------------------------------------------------
    # primitives
    # ------------------------------------------------------------------

    def _slow_step(self, r: torch.Tensor, h_new: torch.Tensor, tick: int) -> torch.Tensor:
        """clockwork: 各模块仅在 tick%K==0 时门控写入, 其余步原样保持."""
        cand = torch.tanh(self.slow_proj(h_new))
        gate = torch.sigmoid(self.slow_gate(h_new))
        out, off = [], 0
        for K, D in self.slow_specs:
            seg = slice(off, off + D)
            if tick % K == 0:
                out.append((1 - gate[:, seg]) * r[:, seg] + gate[:, seg] * cand[:, seg])
            else:
                out.append(r[:, seg])
            off += D
        return torch.cat(out, dim=-1)

    def get_initial_state(
        self, batch_size: int, device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> TSSMState:
        h = torch.zeros(batch_size, self.deter_dim, device=device, dtype=dtype)
        # init z = uniform 1/classes; z_flat is then constant 1/C across
        # all dims.  Matches what the training loop uses for the t=0 token.
        z = (torch.ones(batch_size, self.config.stoch, self.config.classes,
                        device=device, dtype=dtype) / self.config.classes)
        history = torch.zeros(batch_size, 0, self.d_model, device=device, dtype=dtype)
        r = (torch.zeros(batch_size, self.slow_total, device=device, dtype=dtype)
             if self.slow_total else None)
        return TSSMState(h=h, z=z, history=history, r=r, tick=0)

    def encode_feats(self, feat: torch.Tensor) -> torch.Tensor:
        # dtype 跟随权重(07-27 v12): DeepSpeed bf16 下权重为 bf16, 硬 .float() 会撞 LN;
        # 独立 fp32 训练时权重 fp32, 行为与原 .float() 等价。
        _dt = next(self.feat_encoder.parameters()).dtype
        return self.feat_encoder(feat.to(_dt) / self.config.feat_scale)

    def decode_feats_raw(self, state_full: torch.Tensor) -> torch.Tensor:
        """Decoder output in RAW feature space (undoes feat_scale).

        feat_decoder itself lives in scaled space; external consumers
        (gist decoding, KNN eval) should use this instead.
        """
        return self.feat_decoder(state_full) * self.config.feat_scale

    def encode_lag(self, k, n: int, device) -> torch.Tensor:
        """lag k → [n,64]; fourier模式吃任意实数k(插值), 查表模式吃列表索引."""
        if getattr(self.config, "nce_lag_fourier", False):
            fr = torch.tensor([1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256],
                              device=device)
            if torch.is_tensor(k) and k.dim() >= 1:            # 逐元素k: [...]→[...,64]
                kv = k.float().unsqueeze(-1)
                ff = torch.cat([torch.sin(kv * fr * 3.14159),
                                torch.cos(kv * fr * 3.14159)], dim=-1)
                return self.nce_lag_emb(ff)
            kv = torch.as_tensor(float(k), device=device)
            ff = torch.cat([torch.sin(kv * fr * 3.14159), torch.cos(kv * fr * 3.14159)])
            return self.nce_lag_emb(ff).unsqueeze(0).expand(n, -1)
        idx = torch.full((n,), int(k), dtype=torch.long, device=device)
        return self.nce_lag_emb(idx)

    def step_prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.prior(h)
        z = self._sample_stochastic(logits)
        return logits, z

    def step_posterior(self, enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Myopic posterior: depends on obs only, not on h."""
        logits = self.posterior(enc)
        z = self._sample_stochastic(logits)
        return logits, z

    def _post_ctx_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """v11: 观测塔因果前向。tokens [B,T,enc](post_ctx_in 之后, 未加位置)。"""
        T = tokens.shape[1]
        x = tokens + self.post_ctx_pos[:T].unsqueeze(0)
        # 递归孪生实验时观测塔同受窗宽限制(信息拓扑整体递归化, 单变量纯净)
        causal = make_causal_mask(T, tokens.device, getattr(self.config, "attn_window", 0))
        return self.post_ctx(x, mask=causal)

    def _post_ctx_tokens(self, enc: torch.Tensor, a_emb_shifted: torch.Tensor) -> torch.Tensor:
        return self.post_ctx_in(torch.cat([enc, a_emb_shifted], dim=-1))

    # ------------------------------------------------------------------
    # public inference interface
    # ------------------------------------------------------------------

    def get_features(self, state: TSSMState) -> torch.Tensor:
        return state.full

    @torch.no_grad()
    def inference_step(
        self,
        state: TSSMState,
        action_prev: torch.Tensor,
        feat_current: torch.Tensor,
    ) -> TSSMState:
        """Posterior filter step: incorporate (a_{t-1}, obs_t) into state.

        v4: token is built from state.z_flat (= z_{t-1}) and
        action_embed(action_prev).  Posterior is myopic on obs only.
        """
        a_emb = self.action_embed(action_prev)
        h_new, new_history = self.core.step_with_history(
            state.z_flat, a_emb, state.history
        )
        enc = self.encode_feats(feat_current)
        if self.post_ctx is not None:
            ch = state.ctx_hist
            if ch is None:
                ch = torch.zeros(enc.shape[0], 0, enc.shape[-1],
                                 device=enc.device, dtype=enc.dtype)
            # t=0 用零向量动作嵌入, 与训练前向 zero_a 严格一致(动力学塔的历史差异不扩散到新塔)
            a_ctx_in = (torch.zeros_like(a_emb) if ch.shape[1] == 0 else a_emb).to(enc.dtype)
            tok = self._post_ctx_tokens(enc, a_ctx_in).unsqueeze(1)             # [B,1,enc]
            ch = torch.cat([ch, tok], dim=1)
            c_last = self._post_ctx_forward(ch)[:, -1]
            logits = self.posterior(c_last)
            z_new = self._sample_stochastic(logits)
            r_new = (self._slow_step(state.r, h_new, state.tick)
                     if self.slow_total else None)
            return TSSMState(h=h_new, z=z_new, history=new_history, ctx_hist=ch,
                             r=r_new, tick=state.tick + 1)
        _, z_new = self.step_posterior(enc)
        r_new = (self._slow_step(state.r, h_new, state.tick)
                 if self.slow_total else None)
        return TSSMState(h=h_new, z=z_new, history=new_history,
                         r=r_new, tick=state.tick + 1)

    @torch.no_grad()
    # ⚰已失效(07-30盘点): 想象rollout, TransDreamer原教旨部件, 导航线imag权重恒0从未启用
    def imagine_step(self, state: TSSMState, action: torch.Tensor) -> TSSMState:
        """Open-loop step: action only, no obs."""
        a_emb = self.action_embed(action)
        h_new, new_history = self.core.step_with_history(
            state.z_flat, a_emb, state.history
        )
        _, z_new = self.step_prior(h_new)
        r_new = (self._slow_step(state.r, h_new, state.tick)
                 if self.slow_total else None)
        return TSSMState(h=h_new, z=z_new, history=new_history,
                         ctx_hist=state.ctx_hist, r=r_new, tick=state.tick + 1)

    @torch.no_grad()
    # ⚰已失效(07-30盘点): 同imagine_step, 从未启用
    def imagine_with_actions(
        self,
        initial_state: TSSMState,
        action_seq: torch.Tensor,
    ) -> List[TSSMState]:
        H = action_seq.shape[1]
        states: List[TSSMState] = []
        state = initial_state
        for t in range(H):
            state = self.imagine_step(state, action_seq[:, t])
            states.append(state)
        return states

    @torch.no_grad()
    # ⚰已失效(07-30盘点): 惊奇度诊断, 无生产调用方
    def compute_innovation(
        self,
        state: TSSMState,
        action_prev: torch.Tensor,
        feat_current: torch.Tensor,
    ) -> Tuple[TSSMState, torch.Tensor]:
        a_emb = self.action_embed(action_prev)
        h_new, new_history = self.core.step_with_history(
            state.z_flat, a_emb, state.history
        )
        prior_logits, _ = self.step_prior(h_new)
        enc = self.encode_feats(feat_current)
        post_logits, z_new = self.step_posterior(enc)
        innovation = self._kl_divergence(post_logits, prior_logits)
        return (TSSMState(h=h_new, z=z_new, history=new_history), innovation)

    # ------------------------------------------------------------------
    # backward / retro prediction loss (Tang et al. ICML23 bidirectional)
    # ------------------------------------------------------------------
    # ♻退役返聘(07-30盘点): v8retro系enc空间回溯头; v5/ncev5配方RETRO_W=0不再训,
    # 但qwen在用的v885k state由它训出, 在线加载老ckpt仍需此代码路径存在
    def _retro_prediction_loss(self, h_seq, z_flat, enc_seq, actions, m, cfg):
        """从 state_t 反向预测 lag k 的**过去**观测编码 enc_{t-k}。
        - 动作条件: 喂 [state_t, lag_emb(k), 该段动作的均值嵌入], 让反向预测知道"怎么走过来的"
        - 目标 stop-grad(semi-gradient): 防止表征塌到常数(论文的不塌缩关键)
        - 多滞后(默认 1..20): 一步反向太弱(s_{t+1} 本就由 s_t 递推, loss 秒降到0),
          长滞后才逼真正的长程保留 —— 正对我们 §10 探针测出的缺口。
        """
        B, T = actions.shape
        lags = [int(x) for x in str(cfg.retro_lags).split(",") if int(x) < T]
        if not lags:
            return h_seq.new_zeros(())
        a_emb_all = self.action_embed(actions).to(h_seq.dtype)                # [B,T,emb]
        if getattr(cfg, "fast_aux", False):
            return self._aux_pred_loss_fast(
                h_seq, z_flat, enc_seq, a_emb_all, m, lags,
                cfg.retro_n_starts, self.retro_head, self.retro_lag_emb, back=True)
        num = h_seq.new_zeros(()); den = h_seq.new_zeros(())
        for k in lags:
            n = min(cfg.retro_n_starts, T - k)
            if n <= 0:
                continue
            for _ in range(n):
                t = int(torch.randint(k, T, (1,)).item())
                st = torch.cat([h_seq[:, t], z_flat[:, t]], dim=-1)           # state_t
                lag_e = self.retro_lag_emb(torch.full((B,), lags.index(k),
                                                      dtype=torch.long, device=st.device))
                a_ctx = a_emb_all[:, t - k:t].mean(dim=1)                     # 该段动作上下文
                pred = self.retro_head(torch.cat([st, lag_e, a_ctx], dim=-1))
                tgt = enc_seq[:, t - k].detach()                              # stop-grad 目标
                mm = (m[:, t] * m[:, t - k])
                # BYOL式归一化L2 (= 2−2cos): 目标是**学到的**表征, 用裸MSE会给出
                # "整体缩小enc"的塌缩通道; 归一化后尺度不变, 堵掉该漏洞。
                # (Tang et al. 建在 BYOL 上, BYOL 原始损失即归一化L2)
                per = 2.0 - 2.0 * F.cosine_similarity(pred, tgt, dim=-1)
                num = num + (per * mm).sum(); den = den + mm.sum()
        return num / den.clamp(min=1.0)

    def _actx(self, win: torch.Tensor) -> torch.Tensor:
        """动作窗口 [B,S,k,emb] -> 上下文 [B,S,emb]。gru 保序, mean 丢序。"""
        if self.actx_gru is None:
            return win.mean(dim=2)
        B, S, k, E = win.shape
        out, _ = self.actx_gru(win.reshape(B * S, k, E))
        return out[:, -1].reshape(B, S, E)

    # ⚰已失效(07-30盘点): v9-latent臂阵亡, latent_weight恒0
    def _latent_pred_loss(self, h_seq, z_flat, a_emb_all, m, cfg):
        """v9-latent: state_t + 有序动作上下文 -> sg[state_{t+k}] (隐空间转移一致性)。

        与 _fwd_prediction_loss 的区别: 目标不是观测编码 enc_{t+k} 而是**状态本身**。
        ⚠ sg 是生死线: 目标由本网络产生, 无 sg 则"所有 state 塌成同一常数"损失恰为 0。
        sg 使其成为 semi-gradient(BYOL/SimSiam 家法), 堵死该塌缩通道。
        且 latent 目标不像 enc 那样被 recon 锚住 -> 额外回报 lat_std 作塌缩监控。
        """
        B, T = m.shape
        dev = h_seq.device
        lags = [int(x) for x in str(cfg.latent_lags).split(",") if int(x) < T]
        if not lags:
            return h_seq.new_zeros(()), h_seq.new_zeros(())
        num = h_seq.new_zeros(()); den = h_seq.new_zeros(())
        std_acc = h_seq.new_zeros(()); nstd = 0
        for li, k in enumerate(lags):
            n = min(cfg.latent_n_starts, T - k)
            if n <= 0:
                continue
            t = torch.randint(0, T - k, (n,), device=dev)                 # [n]
            st = torch.cat([h_seq[:, t], z_flat[:, t]], dim=-1)           # [B,n,S]
            tgt = torch.cat([h_seq[:, t + k], z_flat[:, t + k]],
                            dim=-1).detach()                              # stop-grad
            idx = t.view(n, 1) + torch.arange(k, device=dev).view(1, k)   # [n,k]
            a_ctx = self._actx(a_emb_all[:, idx])                         # [B,n,emb]
            lg = self.latent_lag_emb(
                torch.full((n,), li, dtype=torch.long, device=dev)
            ).unsqueeze(0).expand(B, n, -1).to(st.dtype)
            pred = self.latent_head(
                torch.cat([st, lg, a_ctx], dim=-1).reshape(B * n, -1))
            per = 2.0 - 2.0 * F.cosine_similarity(pred, tgt.reshape(B * n, -1), dim=-1)
            mm = (m[:, t] * m[:, t + k]).reshape(B * n)
            num = num + (per * mm).sum(); den = den + mm.sum()
            # 塌缩监控: 目标 state 在 batch 维的平均标准差(归一化后); ->0 即塌
            tn = F.normalize(tgt.reshape(B * n, -1).float(), dim=-1)
            std_acc = std_acc + tn.std(dim=0).mean(); nstd += 1
        return num / den.clamp(min=1.0), std_acc / max(1, nstd)

    # ⚰已失效(07-30盘点): biraw双向原始特征头共用引擎; 07-29机制判决(预测式目标不存情节细节)后整线阵亡
    def _aux_pred_loss_fast(self, h_seq, z_flat, enc_seq, a_emb_all, m,
                            lags, n_starts, head, lag_emb, back: bool):
        """retro/fwd 的**向量化**实现 (07-26): 把 len(lags)×n_starts 次串行小 MLP
        合并成 1 次大调用。原实现每次只算 [B, ~3200] 的 3 层 MLP —— 计算量微不足道,
        全是 kernel 启动 + Python 开销; v9(40次循环)实测占单步耗时约 47%。

        与循环版**数学等价**(同样的 2-2cos + stop-grad + 有效样本加权), 仅采样的
        RNG 流不同(统计等价)。动作上下文用 cumsum 取区间均值, 避免逐次切片。
        back=True: state_t -> enc_{t-k}; back=False: state_t -> enc_{t+k}。
        """
        B, T = m.shape
        dev = h_seq.device
        cs = torch.cat([a_emb_all.new_zeros(B, 1, a_emb_all.shape[-1]),
                        a_emb_all.cumsum(dim=1)], dim=1)                      # [B,T+1,emb]
        ts, ks, ids = [], [], []
        for li, k in enumerate(lags):
            n = min(n_starts, T - k)
            if n <= 0:
                continue
            lo, hi = (k, T) if back else (0, T - k)
            ts.append(torch.randint(lo, hi, (n,), device=dev))
            ks.append(torch.full((n,), k, dtype=torch.long, device=dev))
            ids.append(torch.full((n,), li, dtype=torch.long, device=dev))
        if not ts:
            return h_seq.new_zeros(())
        t = torch.cat(ts); k = torch.cat(ks); li = torch.cat(ids)             # [S]
        S = t.shape[0]
        other = t - k if back else t + k                                      # 目标时刻
        lo_i = torch.minimum(t, other); hi_i = torch.maximum(t, other)        # 动作区间 [lo,hi)

        st = torch.cat([h_seq[:, t], z_flat[:, t]], dim=-1)                   # [B,S,state]
        a_ctx = (cs[:, hi_i] - cs[:, lo_i]) / k.view(1, S, 1).to(cs.dtype)    # [B,S,emb]
        lg = lag_emb(li).unsqueeze(0).expand(B, S, -1).to(st.dtype)           # [B,S,64]
        pred = head(torch.cat([st, lg, a_ctx], dim=-1).reshape(B * S, -1))
        tgt = enc_seq[:, other].detach().reshape(B * S, -1)                   # stop-grad
        per = 2.0 - 2.0 * F.cosine_similarity(pred, tgt, dim=-1)              # [B*S]
        mm = (m[:, t] * m[:, other]).reshape(B * S)
        return (per * mm).sum() / mm.sum().clamp(min=1.0)

    # ⚰已失效(07-30盘点): v7/v9前向预测(enc空间), 死于马尔可夫稀释(当前帧信息足够预测下一帧)
    def _fwd_prediction_loss(self, h_seq, z_flat, enc_seq, actions, m, cfg):
        """v9: 从 state_t **只经 MLP** 预测 lag k 的**未来**观测编码 enc_{t+k}。
        与 _future_prediction_loss 的本质区别: 不重跑注意力, 原始历史 token 不可达,
        state_t 是唯一信息通道 —— NextLat/严格中介式堵旁路。其余纪律照抄 retro:
        多滞后 + stop-grad 目标 + BYOL 归一化 L2 (2-2cos) 防塌缩。
        a_ctx = [t, t+k) 的动作均值嵌入: 预测头知道"接下来怎么走", 让 state 专注
        "此地是哪、还剩什么"而非背动作序列。"""
        B, T = actions.shape
        lags = [int(x) for x in str(cfg.fwd_lags).split(",") if int(x) < T]
        if not lags:
            return h_seq.new_zeros(())
        a_emb_all = self.action_embed(actions).to(h_seq.dtype)                # [B,T,emb]
        if getattr(cfg, "fast_aux", False):
            return self._aux_pred_loss_fast(
                h_seq, z_flat, enc_seq, a_emb_all, m, lags,
                cfg.fwd_n_starts, self.fwd_head, self.fwd_lag_emb, back=False)
        num = h_seq.new_zeros(()); den = h_seq.new_zeros(())
        for k in lags:
            n = min(cfg.fwd_n_starts, T - k)
            if n <= 0:
                continue
            for _ in range(n):
                t = int(torch.randint(0, T - k, (1,)).item())
                st = torch.cat([h_seq[:, t], z_flat[:, t]], dim=-1)           # state_t
                lag_e = self.fwd_lag_emb(torch.full((B,), lags.index(k),
                                                    dtype=torch.long, device=st.device))
                a_ctx = a_emb_all[:, t:t + k].mean(dim=1)                     # 将走的动作
                pred = self.fwd_head(torch.cat([st, lag_e, a_ctx], dim=-1))
                tgt = enc_seq[:, t + k].detach()                              # stop-grad 目标
                mm = (m[:, t] * m[:, t + k])
                per = 2.0 - 2.0 * F.cosine_similarity(pred, tgt, dim=-1)
                num = num + (per * mm).sum(); den = den + mm.sum()
        return num / den.clamp(min=1.0)

    # ------------------------------------------------------------------
    # multi-step open-loop future prediction loss (grad-enabled)
    # ------------------------------------------------------------------
    # ⚰已失效(07-30盘点): future头, 与fwd同系死因; FUTURE_W恒0
    def _future_prediction_loss(self, shifted_z, shifted_a, z_flat, h_seq,
                                actions, feats, m, cfg):
        """From n_starts positions t, roll the PRIOR forward K steps with the REAL
        actions (open-loop obs), decode each imagined state, match the real future
        SigLIP feat. Re-runs the transformer core (grad-enabled; the @no_grad
        imagine_step is inference-only). Target=frozen SigLIP => no collapse."""
        B, T = actions.shape
        K = cfg.future_horizon
        vmax = T - K - 1
        if vmax < 0:
            return h_seq.new_zeros(())
        n = min(cfg.future_n_starts, vmax + 1)
        # 起点按真实集长采样(07-26 codex审修): 均匀[0,T_pad)会偏向长集晚段(T_pad由
        # batch最长集决定), 短集几乎采不到。以随机episode的有效长度为参考采起点。
        lens = m.sum(dim=1)                                                   # [B]
        starts = []
        for _ in range(n):
            L_ref = int(lens[torch.randint(0, B, (1,), device=m.device)].item())
            hi = min(max(L_ref - K - 1, 0), vmax)
            starts.append(int(torch.randint(0, hi + 1, (1,)).item()))
        all_tokens = self.core._build_tokens(shifted_z, shifted_a)            # [B,T,d] grad
        a_emb_all = self.action_embed(actions).to(all_tokens.dtype)           # [B,T,emb]
        pos_emb = self.core.pos_emb
        num = h_seq.new_zeros(())
        den = h_seq.new_zeros(())
        for t0 in starts:
            hist = all_tokens[:, :t0 + 1]                                     # [B,t0+1,d]
            z_prev = z_flat[:, t0]                                            # posterior z_t
            for j in range(K):
                new_tok = self.core._build_tokens(z_prev, a_emb_all[:, t0 + j]).unsqueeze(1)
                hist = torch.cat([hist, new_tok], dim=1)
                Tn = hist.shape[1]
                x = hist + pos_emb[:Tn].unsqueeze(0)
                causal = make_causal_mask(Tn, x.device, getattr(self.core.cfg, "attn_window", 0))
                out = self.core._run_layers(x, causal)
                h = out[:, -1]
                if self.core.h_proj is not None:
                    h = self.core.h_proj(h)
                z_prev = self._sample_stochastic(self.prior(h)).reshape(B, -1)  # prior z (open-loop)
                _dx = torch.cat([h, z_prev], dim=-1)
                if cfg.future_stopgrad_decoder:
                    from torch.func import functional_call
                    _pb = {k: v.detach() for k, v in
                           self.feat_decoder.named_parameters()}
                    _pb.update({k: v for k, v in self.feat_decoder.named_buffers()})
                    pred = functional_call(self.feat_decoder, _pb, (_dx,))    # 梯度只回core/prior
                else:
                    pred = self.feat_decoder(_dx)                             # [B, feat]
                mm = m[:, t0 + j + 1]
                # 与 recon 同量纲(normal: 0.5·Σdim/σ²), 否则 1152 维求和 vs mean 差 ~144x,
                # recon_weight 0.3 仍 ~43x 主导, "future 为主信号"失效 (07-25 fable review)
                if cfg.recon_type == "normal":
                    per = (0.5 * (pred - feats[:, t0 + j + 1]).pow(2).sum(dim=-1)
                           / (cfg.rec_sigma ** 2) / max(1, cfg.recon_groups))
                else:
                    per = F.mse_loss(pred, feats[:, t0 + j + 1], reduction="none").mean(dim=-1)
                # 按有效样本数加权(codex审修): 原per-term等权会让"仅少数长集有效"的
                # 晚段term与满员term同权 → 长集晚段过权+高方差。
                num = num + (per * mm).sum()
                den = den + mm.sum()
        return num / den.clamp(min=1.0)

    # ------------------------------------------------------------------
    # training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        feats: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_first: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        kl_scale: float = 1.0,
        feas: Optional[torch.Tensor] = None,        # [B,T,39] 0/1标签(前3=feas,后36=trav)
        feas_ok: Optional[torch.Tensor] = None,     # [B] 该集有无标签
    ) -> TSSMOutput:
        """Sequence-parallel forward.

        v4 (TransDreamer-style):
          1) enc_seq    = encoder(feats)                          # parallel
          2) post_logits= q(enc_seq)                              # myopic, parallel
             z_post     = sample_ST(post_logits)
          3) shifted_z  = [init_z, z_post[0..T-2]]
             shifted_a  = [zero,   action_emb[0..T-2]]
             token_t    = Linear(cat(shifted_z[t], shifted_a[t]))
             h_seq      = transformer(tokens) [causal]
          4) prior_logits = p(h_seq)
          5) recon      = decoder(cat(h_seq, z_post_flat))
          6) loss = recon + kl_scale * (kl_balance*dyn + (1-kl_balance)*rep)
        """
        B, T = feats.shape[:2]
        device = feats.device
        cfg = self.config

        _dt = next(self.feat_encoder.parameters()).dtype
        feats = feats.to(_dt) / cfg.feat_scale          # dtype跟随权重(07-27, 见 encode_feats 注)
        if actions is None:
            actions = torch.zeros(B, T, dtype=torch.long, device=device)
        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # 1) encode all observations in parallel
        enc_seq = self.feat_encoder(
            feats.reshape(B * T, -1)
        ).view(B, T, self.encoder_dim)

        # 2) posterior: myopic q(z_t|enc_t) 或 v11 观测塔 q(z_t|enc_1:t, a_1:t)
        if self.post_ctx is not None:
            a_emb_seq = self.action_embed(actions).to(enc_seq.dtype)          # [B,T,emb]
            a_shift = torch.cat([torch.zeros_like(a_emb_seq[:, :1]),
                                 a_emb_seq[:, :-1]], dim=1)                   # a_{t-1}
            ctx = self._post_ctx_forward(self._post_ctx_tokens(enc_seq, a_shift))
            post_logits = self.posterior(ctx)                                 # [B, T, S, C]
        else:
            post_logits = self.posterior(enc_seq)                             # [B, T, S, C]
        z_sample = self._sample_stochastic(post_logits)                       # [B, T, S, C]
        z_flat = z_sample.reshape(B, T, -1)                                   # [B, T, S*C]

        # 3) shifted (z_{t-1}, a_{t-1}) tokens for causal transformer
        init_z_flat = (
            torch.ones(B, 1, self.z_dim_flat, device=device, dtype=z_flat.dtype)
            / cfg.classes
        )
        shifted_z = torch.cat([init_z_flat, z_flat[:, :-1]], dim=1)           # [B, T, S*C]
        zero_a = torch.zeros(B, 1, cfg.action_embed_dim, device=device, dtype=z_flat.dtype)
        a_emb_curr = self.action_embed(actions).to(z_flat.dtype)
        shifted_a = torch.cat([zero_a, a_emb_curr[:, :-1]], dim=1)            # [B, T, emb]

        # 4) transformer forward
        h_seq = self.core.forward_seq(shifted_z, shifted_a)                   # [B, T, h_dim]

        # 5) prior(h_t) -> z_t logits  (no obs anywhere)
        prior_logits = self.prior(h_seq)                                      # [B, T, S, C]

        # 6) reconstruction with cat(h, z [, r])
        r_seq = None
        if self.slow_total:
            _r = h_seq.new_zeros(B, self.slow_total)
            _rs = []
            for _t in range(T):
                _r = self._slow_step(_r, h_seq[:, _t], _t)
                _rs.append(_r)
            r_seq = torch.stack(_rs, dim=1)                                   # [B, T, S]
            recon_in = torch.cat([h_seq, z_flat, r_seq], dim=-1)
        else:
            recon_in = torch.cat([h_seq, z_flat], dim=-1)                     # [B, T, h+S*C]
        recon = self.feat_decoder(recon_in)                                   # [B, T, feat]

        # 7) losses
        m = mask.float()
        weight = m.sum().clamp(min=1e-8)

        if cfg.recon_type == "normal":
            # TransDreamer modules_transformer.py:117 — Independent Normal(mean, sigma)
            # negative log-prob (without constant) = 0.5 * sum_i ((feat-recon)/sigma)^2
            # recon_groups: mean over patch blocks so the p9 cache keeps the
            # loss scale of the 1152-d recipe.
            _se = (recon - feats).pow(2)
            if cfg.recon_sem_dim > 0 and cfg.recon_sem_boost != 1.0:
                _se = torch.cat([_se[..., :cfg.recon_sem_dim] * cfg.recon_sem_boost,
                                 _se[..., cfg.recon_sem_dim:]], dim=-1)
            sq_err = _se.sum(dim=-1)                                          # [B, T]
            per_recon = 0.5 * sq_err / (cfg.rec_sigma ** 2) / max(1, cfg.recon_groups)
        elif cfg.recon_type == "mse":
            per_recon = F.mse_loss(recon, feats, reduction="none").mean(dim=-1)
        else:
            raise ValueError(f"unknown recon_type: {cfg.recon_type}")
        recon_val = (per_recon * m).sum() / weight

        # DreamerV2 KL balancing
        free = torch.tensor(cfg.free_nats, device=device)
        rep_kl_raw = self._kl_divergence(post_logits, prior_logits.detach())  # rep term
        dyn_kl_raw = self._kl_divergence(post_logits.detach(), prior_logits)  # dyn term
        rep_kl = torch.maximum(rep_kl_raw, free)
        dyn_kl = torch.maximum(dyn_kl_raw, free)
        rep_val = (rep_kl * m).sum() / weight
        dyn_val = (dyn_kl * m).sum() / weight
        kl_term = (1.0 - cfg.kl_balance) * rep_val + cfg.kl_balance * dyn_val

        # Optional imagination loss (off by default in v4 — TransDreamer doesn't
        # use it either; kept for ablation).
        imag_val = torch.tensor(0.0, device=device)
        if cfg.imagination_loss_weight > 0:
            z_prior = self._sample_stochastic(prior_logits)
            z_prior_flat = z_prior.reshape(B, T, -1)
            imag_recon_in = torch.cat([h_seq, z_prior_flat], dim=-1)
            imag_recon = self.feat_decoder(imag_recon_in)
            if cfg.recon_type == "normal":
                sq_err_imag = (imag_recon - feats).pow(2).sum(dim=-1)
                per_imag = 0.5 * sq_err_imag / (cfg.rec_sigma ** 2) / max(1, cfg.recon_groups)
            else:
                per_imag = F.mse_loss(imag_recon, feats, reduction="none").mean(dim=-1)
            imag_val = (per_imag * m).sum() / weight

        action_inv_val = torch.tensor(0.0, device=device)
        action_inv_acc = torch.tensor(0.0, device=device)
        if cfg.action_inv_weight > 0 and self.action_inv_head is not None and T > 1:
            # WAM-style inverse dynamics: infer the executed action from
            # consecutive observation embeddings, not from h/z where action is
            # already part of the transition input.
            pair_mask = (mask[:, :-1] & mask[:, 1:]).float()
            pair_weight = pair_mask.sum().clamp(min=1e-8)
            action_logits = self.action_inv_head(
                torch.cat([enc_seq[:, :-1], enc_seq[:, 1:]], dim=-1)
            )
            per_action = F.cross_entropy(
                action_logits.reshape(-1, cfg.num_actions).float(),
                actions[:, :-1].reshape(-1),
                reduction="none",
            ).view(B, T - 1)
            action_inv_val = (per_action * pair_mask).sum() / pair_weight
            pred_action = action_logits.argmax(dim=-1)
            action_inv_acc = (
                ((pred_action == actions[:, :-1]).float() * pair_mask).sum()
                / pair_weight
            )

        pose_delta_val = torch.tensor(0.0, device=device)
        pose_delta_fwd_mae = torch.tensor(0.0, device=device)
        pose_delta_yaw_mae_deg = torch.tensor(0.0, device=device)
        if cfg.pose_delta_weight > 0 and self.pose_delta_head is not None and T > 1:
            # h_{t+1} is the transition state after token(z_t, a_t), so it is
            # the right place to ask whether the TSSM transition encodes the
            # local pose delta implied by action a_t.
            pair_mask = (mask[:, :-1] & mask[:, 1:]).float()
            pair_weight = pair_mask.sum().clamp(min=1e-8)
            pose_pred = self.pose_delta_head(h_seq[:, 1:]).float()
            pose_target = self._action_pose_delta_target(actions[:, :-1]).to(pose_pred)
            per_pose = F.smooth_l1_loss(pose_pred, pose_target, reduction="none").mean(dim=-1)
            pose_delta_val = (per_pose * pair_mask).sum() / pair_weight
            err = (pose_pred.detach() - pose_target).abs()
            pose_delta_fwd_mae = (err[..., 0] * 0.75 * pair_mask).sum() / pair_weight
            pose_delta_yaw_mae_deg = (
                err[..., 1] * 45.0 * pair_mask
            ).sum() / pair_weight

        future_val = torch.tensor(0.0, device=device)
        if cfg.future_weight > 0:
            future_val = self._future_prediction_loss(
                shifted_z, shifted_a, z_flat, h_seq, actions, feats, m, cfg)
        fwd_val = torch.tensor(0.0, device=device)
        if cfg.fwd_weight > 0:
            fwd_val = self._fwd_prediction_loss(h_seq, z_flat, enc_seq, actions, m, cfg)
        latent_val = torch.tensor(0.0, device=device)
        lat_std = torch.tensor(0.0, device=device)
        if cfg.latent_weight > 0:
            a_emb_seq = self.action_embed(actions).to(h_seq.dtype)
            latent_val, lat_std = self._latent_pred_loss(
                h_seq, z_flat, a_emb_seq, m, cfg)
        retro_val = torch.tensor(0.0, device=device)
        if cfg.retro_weight > 0 and self.retro_head is not None:
            retro_val = self._retro_prediction_loss(h_seq, z_flat, enc_seq, actions, m, cfg)

        raw_fwd_val = torch.tensor(0.0, device=device)
        raw_retro_val = torch.tensor(0.0, device=device)
        if (cfg.raw_fwd_weight > 0 or cfg.raw_retro_weight > 0):
            _rlags = [int(x) for x in str(cfg.raw_lags).split(",") if x.strip()]
            _rlags = [k for k in _rlags if k < T]
            _aemb = self.action_embed(actions).to(h_seq.dtype)
            if _rlags and self.raw_fwd_head is not None:
                raw_fwd_val = self._aux_pred_loss_fast(
                    h_seq, z_flat, feats, _aemb, m, _rlags,
                    cfg.retro_n_starts, self.raw_fwd_head, self.raw_lag_emb, back=False)
            if _rlags and self.raw_retro_head is not None:
                raw_retro_val = self._aux_pred_loss_fast(
                    h_seq, z_flat, feats, _aemb, m, _rlags,
                    cfg.retro_n_starts, self.raw_retro_head, self.raw_lag_emb, back=True)

        nce_val = torch.tensor(0.0, device=device)
        nce_acc = torch.tensor(0.0, device=device)
        if (cfg.nce_weight > 0 and self.nce_q_head is not None
                and getattr(cfg, "nce_event_mode", False)):
            # ---- evt30: 惊奇锚定考纲. 事件=集内innov(=rep_kl_raw)前(1-q)分位帧 ----
            innov = rep_kl_raw.detach()                                       # [B,T]
            Kf = F.normalize(self.nce_k_head(feats), dim=-1)                  # [B,T,dc]
            a_emb_n = self.action_embed(actions).to(h_seq.dtype)
            cs_n = torch.cat([a_emb_n.new_zeros(B, 1, a_emb_n.shape[-1]),
                              a_emb_n.cumsum(dim=1)], dim=1)
            ranks = [int(x) for x in str(cfg.nce_event_ranks).split(",") if x.strip()]
            qv = float(cfg.nce_event_quantile)
            ev_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            for b in range(B):
                vt = m[b].bool()
                if vt.sum() < 4:
                    continue
                thr = torch.quantile(innov[b][vt], qv)
                ev_mask[b] = (innov[b] >= thr) & vt
            n = min(cfg.retro_n_starts, T - 2)
            ts = torch.randint(2, T, (n,), device=device)
            losses_n, accs_n, wsum = [], [], 0.0
            for mi, rank in enumerate(ranks):
                kmat = torch.zeros(B, n, dtype=torch.long, device=device)
                for b in range(B):
                    ev_idx = ev_mask[b].nonzero(as_tuple=True)[0]
                    for i in range(n):
                        prior_ev = ev_idx[ev_idx < ts[i]]
                        if prior_ev.numel() >= rank:
                            kmat[b, i] = int(ts[i]) - int(prior_ev[-rank])
                valid = (kmat > 0).float()                                     # [B,n]
                if valid.sum() == 0:
                    continue
                ksafe = kmat.clamp(min=1)
                st = torch.cat([h_seq[:, ts], z_flat[:, ts]]
                               + ([r_seq[:, ts]] if r_seq is not None else []), dim=-1)
                tgt = ts.unsqueeze(0).expand(B, -1) - ksafe                    # [B,n] 事件帧
                idx3 = tgt.unsqueeze(-1).expand(-1, -1, cs_n.shape[-1])
                a_ctx = (cs_n[:, ts] - torch.gather(cs_n, 1, idx3)) / ksafe.unsqueeze(-1).float()
                lg = self.encode_lag(ksafe, n, device).to(st.dtype)            # [B,n,64]
                q = F.normalize(self.nce_q_head(
                    torch.cat([st, lg, a_ctx], dim=-1)), dim=-1)
                logits = torch.einsum("bnd,btd->bnt", q, Kf) / cfg.nce_tau
                logits = logits.masked_fill(~m.bool().unsqueeze(1), float("-inf"))
                mm = m[torch.arange(B, device=device).unsqueeze(1), ts.unsqueeze(0)] * valid
                ce = F.cross_entropy(logits.reshape(B * n, T), tgt.clamp(min=0).reshape(B * n),
                                     reduction="none").view(B, n)
                ce = torch.where(mm > 0, ce, torch.zeros_like(ce))
                losses_n.append(ce.sum()); wsum += float(mm.sum())
                accs_n.append(((logits.argmax(-1) == tgt).float() * mm).sum())
            if wsum > 0:
                nce_val = torch.stack(losses_n).sum() / wsum
                nce_acc = torch.stack(accs_n).sum() / wsum
        elif cfg.nce_weight > 0 and self.nce_q_head is not None:
            _nlags = [int(x) for x in str(cfg.nce_lags).split(",") if x.strip()]
            _nlags = [k for k in _nlags if k < T]
            if getattr(cfg, "nce_lag_sample", 0) and len(_nlags) > cfg.nce_lag_sample:
                _gt = float(getattr(cfg, "nce_grade_tau", 0.0) or 0.0)
                if _gt > 0:                                   # 重要性抽样: 近档常考远档偶考
                    _p = torch.tensor([float(torch.exp(torch.tensor(-k / _gt)))
                                       for k in _nlags])
                    _pick = torch.multinomial(_p / _p.sum(), cfg.nce_lag_sample,
                                              replacement=False).tolist()
                else:
                    _pick = torch.randperm(len(_nlags))[: cfg.nce_lag_sample].tolist()
                _nlags = [_nlags[i] for i in sorted(_pick)]
            if _nlags:
                Kf = F.normalize(self.nce_k_head(feats), dim=-1)          # [B,T,dc]
                a_emb_n = self.action_embed(actions).to(h_seq.dtype)
                cs_n = torch.cat([a_emb_n.new_zeros(B, 1, a_emb_n.shape[-1]),
                                  a_emb_n.cumsum(dim=1)], dim=1)
                losses_n, accs_n, wsum = [], [], 0.0
                for li, k in enumerate(_nlags):
                    n = min(cfg.retro_n_starts, T - k)
                    if n <= 0:
                        continue
                    ts = torch.randint(k, T, (n,), device=device)
                    if r_seq is not None:
                        st = torch.cat([h_seq[:, ts], z_flat[:, ts],
                                        r_seq[:, ts]], dim=-1)
                    else:
                        st = torch.cat([h_seq[:, ts], z_flat[:, ts]], dim=-1) # [B,n,sd]
                    a_ctx = (cs_n[:, ts] - cs_n[:, ts - k]) / float(k)
                    if getattr(cfg, "nce_lag_fourier", False):
                        lg = self.encode_lag(k, n, device).unsqueeze(0).expand(B, -1, -1)
                    else:
                        lg = self.nce_lag_emb(torch.full((n,), li, dtype=torch.long,
                                              device=device)).unsqueeze(0).expand(B, -1, -1)
                    q = F.normalize(self.nce_q_head(
                        torch.cat([st, lg.to(st.dtype), a_ctx], dim=-1)), dim=-1)  # [B,n,dc]
                    logits = torch.einsum("bnd,btd->bnt", q, Kf) / cfg.nce_tau     # [B,n,T]
                    logits = logits.masked_fill(~m.bool().unsqueeze(1), float("-inf"))
                    tgt = (ts - k).unsqueeze(0).expand(B, -1)                       # [B,n]
                    mm = m[:, ts] * m[:, ts - k]                                    # 双端有效
                    ce = F.cross_entropy(logits.reshape(B * n, T), tgt.reshape(B * n),
                                         reduction="none").view(B, n)
                    # inf(目标在padding区被mask) × 0 = nan; 用where选择性丢弃
                    ce = torch.where(mm > 0, ce, torch.zeros_like(ce))
                    _gtau = float(getattr(cfg, "nce_grade_tau", 0.0) or 0.0)
                    _wk = float(torch.exp(torch.tensor(-k / _gtau))) if _gtau > 0 else 1.0
                    losses_n.append(ce.sum() * _wk); wsum += float(mm.sum()) * _wk
                    accs_n.append(((logits.argmax(-1) == tgt).float() * mm).sum() * _wk)
                if wsum > 0:
                    nce_val = torch.stack([x for x in losses_n]).sum() / wsum
                    nce_acc = torch.stack([x for x in accs_n]).sum() / wsum

        # ---- 08-06 futNCE→TWISTER对齐(AC-CPC): 视距0..S-1全档, 每档独立(q,k)MLP,
        # 动作有序串接, 负例=batch×时间全局展平(跨集), 裸点积无温度, 0.75^t衰减归一 ----
        fut_nce_val = torch.tensor(0.0, device=device)
        fut_nce_acc = torch.tensor(0.0, device=device)
        fut_acc_far = torch.tensor(0.0, device=device)
        if cfg.fut_nce_weight > 0 and self.fut_q_mlps is not None:
            a_emb_f = self.action_embed(actions).to(h_seq.dtype)
            if r_seq is not None:
                st_all = torch.cat([h_seq, z_flat, r_seq], dim=-1)          # [B,T,sd]
            else:
                st_all = torch.cat([h_seq, z_flat], dim=-1)
            _spec = bool(str(getattr(cfg, "fut_nce_spec_lags", "")).strip())
            _horiz = [t_ for t_ in self.fut_horizons if t_ < T]
            lam = float(cfg.fut_nce_lambda)
            wnorm = (float(len(_horiz)) if _spec                 # 专精: 等权
                     else sum(lam ** t_ for t_ in _horiz))       # TWISTER: 衰减
            losses_f, accs_f = [], []
            valid_flat = m.bool().reshape(B * T)                             # 全局键有效位
            for t_h in _horiz:
                n = min(cfg.fut_nce_starts, T - t_h)
                if n <= 0:
                    continue
                ts = torch.randint(0, T - t_h, (n,), device=device)
                qin = st_all[:, ts]                                          # [B,n,sd]
                if t_h > 0:
                    # 动作有序串接: a_{ts..ts+t-1} 按位展平(TWISTER concat同款, 顺序天然保)
                    widx = ts.unsqueeze(1) + torch.arange(t_h, device=device)  # [n,t]
                    win = a_emb_f[:, widx]                                     # [B,n,t,e]
                    qin = torch.cat([qin, win.reshape(B, n, -1)], dim=-1)
                _mi = self.fut_horizons.index(t_h)                            # 头索引查表
                if cfg.fut_nce_norm:                                          # 默认: 稳定版
                    q = F.normalize(self.fut_q_mlps[_mi](qin).float(), dim=-1)
                    k_all = F.normalize(
                        self.fut_k_mlps[_mi](feats).float(), dim=-1).reshape(B * T, -1)
                    logits = torch.einsum("bnd,md->bnm", q, k_all) / cfg.nce_tau
                else:                                                         # TWISTER字面版
                    q = self.fut_q_mlps[_mi](qin)                             # [B,n,512]
                    k_all = self.fut_k_mlps[_mi](feats).reshape(B * T, -1)    # [B*T,512]
                    logits = torch.einsum("bnd,md->bnm", q, k_all)            # 裸点积, 无温度
                logits = logits.masked_fill(~valid_flat.view(1, 1, B * T), float("-inf"))
                # 正例=全局展平里自己那集的ts+t位: 索引 b*T + ts+t
                tgt = (torch.arange(B, device=device).unsqueeze(1) * T
                       + (ts + t_h).unsqueeze(0))                            # [B,n]
                mm = m[:, ts] * m[:, ts + t_h]
                ce = F.cross_entropy(logits.reshape(B * n, B * T), tgt.reshape(B * n),
                                     reduction="none").view(B, n)
                ce = torch.where(mm > 0, ce, torch.zeros_like(ce))
                w_t = (1.0 / wnorm) if _spec else (lam ** t_h) / wnorm
                denom = mm.sum().clamp(min=1)
                losses_f.append(w_t * ce.sum() / denom)
                acc_t = ((logits.argmax(-1) == tgt).float() * mm).sum() / denom
                accs_f.append(w_t * acc_t)
                if t_h == _horiz[-1]:
                    fut_acc_far = acc_t
            if losses_f:
                fut_nce_val = torch.stack(losses_f).sum()
                fut_nce_acc = torch.stack(accs_f).sum()

        # ---- 08-07 可行性/可通行性BCE ----
        feas_val = torch.tensor(0.0, device=device)
        trav_val = torch.tensor(0.0, device=device)
        feas_acc = torch.tensor(0.0, device=device)
        if ((self.feas_head is not None or self.trav_head is not None)
                and feas is not None and feas_ok is not None and feas_ok.any()):
            if r_seq is not None:
                st_ft = torch.cat([h_seq, z_flat, r_seq], dim=-1)
            else:
                st_ft = torch.cat([h_seq, z_flat], dim=-1)                 # [B,T,sd]
            vmask = (m * feas_ok.float().unsqueeze(1))                     # [B,T]
            den_f = vmask.sum().clamp(min=1)
            tgt_all = feas.to(st_ft.dtype)
            if self.feas_head is not None:
                lg3 = self.feas_head(st_ft)                                # [B,T,3]
                bce = F.binary_cross_entropy_with_logits(
                    lg3, tgt_all[..., :3], reduction="none").mean(-1)
                feas_val = (bce * vmask).sum() / den_f
                feas_acc = ((((lg3 > 0).float() == tgt_all[..., :3]).float()
                             .mean(-1) * vmask).sum() / den_f)
            if self.trav_head is not None:
                lg36 = self.trav_head(st_ft)
                bce36 = F.binary_cross_entropy_with_logits(
                    lg36, tgt_all[..., 3:], reduction="none").mean(-1)
                trav_val = (bce36 * vmask).sum() / den_f

        total_loss = (
            cfg.recon_weight * recon_val
            + cfg.nce_weight * nce_val
            + cfg.fut_nce_weight * fut_nce_val
            + cfg.feas_weight * feas_val
            + cfg.trav_weight * trav_val
            + cfg.raw_fwd_weight * raw_fwd_val
            + cfg.raw_retro_weight * raw_retro_val
            + kl_scale * cfg.kl_scale * kl_term
            + cfg.imagination_loss_weight * imag_val
            + cfg.future_weight * future_val
            + cfg.retro_weight * retro_val
            + cfg.fwd_weight * fwd_val
            + cfg.latent_weight * latent_val
            + cfg.action_inv_weight * action_inv_val
            + cfg.pose_delta_weight * pose_delta_val
        )

        loss_dict = {
            "dyn_kl": dyn_val.item(),
            "rep_kl": rep_val.item(),
            "recon": recon_val.item(),
            "imag_recon": imag_val.item(),
            "future": future_val.item(),
            "retro": retro_val.item(),
            "nce": nce_val.item(),
            "nce_acc": nce_acc.item(),
            "fut_nce": fut_nce_val.item(),
            "fut_nce_acc": fut_nce_acc.item(),
            "fut_acc_far": fut_acc_far.item(),
            "feas": feas_val.item(),
            "trav": trav_val.item(),
            "feas_acc": feas_acc.item(),
            "raw_fwd": raw_fwd_val.item(),
            "raw_retro": raw_retro_val.item(),
            "fwd": fwd_val.item(),
            "latent": latent_val.item(),
            "lat_std": lat_std.item(),
            "action_inv": action_inv_val.item(),
            "action_inv_acc": action_inv_acc.item(),
            "pose_delta": pose_delta_val.item(),
            "pose_delta_fwd_mae": pose_delta_fwd_mae.item(),
            "pose_delta_yaw_mae_deg": pose_delta_yaw_mae_deg.item(),
            "total": total_loss.item(),
            "kl_scale": float(kl_scale),
        }

        all_states: List[TSSMState] = [
            TSSMState(h=h_seq[:, t], z=z_sample[:, t], history=None)
            for t in range(T)
        ]
        prior_list = [prior_logits[:, t] for t in range(T)]
        post_list = [post_logits[:, t] for t in range(T)]

        return TSSMOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            states=all_states,
            prior_logits=prior_list,
            post_logits=post_list,
        )


# ===========================================================================
# Smoke test
# ===========================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = TSSMConfig()
    model = NavigationTSSM(cfg)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    B, T = 4, 16
    feats = torch.randn(B, T, cfg.feat_dim)
    actions = torch.randint(0, cfg.num_actions, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, -3:] = False

    print("\n[Test] training forward + backward")
    model.train()
    out = model(feats, actions, mask=mask, kl_scale=1.0)
    print("[train] losses:", out.loss_dict)
    out.loss.backward()
    print("[train] backward OK")
    # Sanity: with random data, KL should be > free_nats almost immediately
    # because prior(h) is essentially zero-info (h is from random init z).
    print(f"[sanity] dyn_kl={out.loss_dict['dyn_kl']:.4f} "
          f"rep_kl={out.loss_dict['rep_kl']:.4f}")

    print("\n[Test] inference + imagination")
    model.eval()
    state = model.get_initial_state(B, feats.device)
    print(f"[init] z_flat first row mean={state.z_flat[0].mean().item():.4f} "
          f"(should be 1/{cfg.classes} = {1.0/cfg.classes:.4f})")
    for t in range(T):
        a_prev = (torch.zeros(B, dtype=torch.long)
                  if t == 0 else actions[:, t - 1])
        state = model.inference_step(state, a_prev, feats[:, t])
    print(f"[inference] state.h {tuple(state.h.shape)}  history "
          f"{tuple(state.history.shape)}")

    H, K = 4, 3
    candidates = torch.randint(0, cfg.num_actions, (K, B, H))
    for k in range(K):
        imagined = model.imagine_with_actions(state, candidates[k])
        feats_out = torch.stack(
            [model.get_features(s) for s in imagined], dim=1
        )
        print(f"[imagine] candidate {k}: feats {tuple(feats_out.shape)}")

    new_state, innov = model.compute_innovation(state, actions[:, -1], feats[:, -1])
    print("[innovation]", innov.tolist())

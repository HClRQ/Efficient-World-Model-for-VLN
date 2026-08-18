"""
TSSMMemoryAdapter — bridges our standalone, frozen P0-fix TSSM to NaVILA.

设计思路 (Design):
  NaVILA 已经有 temporal-state 集成的脚手架 (llava/model/llava_arch.py)，预期接口是：
      state_tokens, rssm_loss = self.tssm(frame_vec, actions, collisions)
  其中 state_tokens 形状 [B, T, num_state_tokens, llm_dim]，作为额外 memory
  token 拼到视觉 token 之后。

  This adapter mimics that interface but:
    1. TSSM core is FROZEN (loaded from P0 fix checkpoint, no grad)
    2. State is read from a pre-computed cache, keyed by (episode_id, step)
    3. Only the projector (state_dim → llm_dim) is trained from scratch
    4. Returns rssm_loss = 0 (no world-model loss during NaVILA finetune)

为什么这样设计 (Why):
  - 预 cache 让 NaVILA 训练吞吐不被 RSSM 在线 forward 拖慢
  - frozen RSSM 保证我们花了一周训出的世界模型不被 NaVILA 的 action 监督污染
  - projector + LoRA 是仅训练新加的连接层，避免动 NaVILA 的核心权重

输入接口 (Input interface) — called from llava_arch.encode_images:
    frame_vec    : [B, T, 1152]  ignored (kept for v3 signature compatibility)
    actions      : [B, T] long   ignored
    collisions   : [B, T]        ignored
    episode_ids  : list[str] (length B)        ← NEW
    steps        : list[int] (length B)        ← NEW

输出 (Output):
    state_tokens : [B, 1, num_state_tokens, llm_dim]
                   the FINAL frame's TSSM state, expanded to N memory tokens.
                   Note T=1 here because the cached state already summarizes
                   the whole trajectory up to the current decision step.
    rssm_loss    : torch.tensor(0.0)
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSSMMemoryAdapter(nn.Module):
    """
    Wraps a pre-computed TSSM state cache + a learnable projector that maps
    `state_dim` into `llm_dim` (4096 for LLaMA-3 8B). Use explicit
    state_start/state_dim slices when intentionally training on h-only, z-only,
    or legacy partial states. The concat_o TSSM cache stores
    [h(2048), z(1024)] = 3072 dims.

    state_dim          : TSSM state slice width fed to the projector
                         (3072 = full concat_o state [h 2048 + z 1024];
                          smaller values require allow_partial_state=True)
    llm_dim            : target LLaMA hidden dim
    num_state_tokens   : how many memory tokens to emit per sample
                         (>1 means we split the projection into N tokens via
                         a wider output layer; 1 is the simplest form)
    cache_path         : .pt file produced by RSSM_v4/cache_tssm_states_v4.py
    projector_hidden   : hidden width of the small MLP projector
    """

    def __init__(
        self,
        state_dim: int = 1536,
        llm_dim: int = 4096,
        num_state_tokens: int = 1,
        cache_path: Optional[str] = None,
        projector_hidden: int = 2048,
        online_tssm_ckpt: Optional[str] = None,
        online_siglip_path: Optional[str] = None,
        allow_partial_state: bool = False,
        state_start: int = 0,
        stage1_projector_path: Optional[str] = None,
    ):
        super().__init__()
        self.state_dim        = state_dim
        self.llm_dim          = llm_dim
        self.num_state_tokens = num_state_tokens
        self.cache_state_dim  = state_dim
        self.allow_partial_state = allow_partial_state
        self.history_states = max(1, int(os.environ.get("TSSM_HISTORY_STATES", "1")))
        self.history_span = os.environ.get("TSSM_HISTORY_SPAN", "full").lower()
        # Sub-slice of the cached full state to feed the projector:
        # full state = [h(2048), z(1024)]=3072. Use state_start to pick a window:
        #   full: state_start=0,    state_dim=3072
        #   h:    state_start=0,    state_dim=2048
        #   z:    state_start=2048, state_dim=1024
        self.state_start = state_start
        # PSEUDO-FRAME mode (FOREWARN-style "feed through the pretrained interface"):
        # instead of injecting memory tokens, decode K history-slot states with the
        # frozen TSSM feat_decoder into SigLIP-space gist vectors; llava_arch tiles
        # them across the patch grid and feeds them through the PRETRAINED
        # mm_projector as historical video frames. LLM reads its native format.
        self.pseudo_frames = int(os.environ.get("TSSM_PSEUDO_FRAMES", "0"))
        # ORACLE-FUTURE slot (TSSM_FUTURE_SLOT=1, diagnostic-only): one extra
        # pseudo-frame filled with the GT features of the NEXT waypoint (linear
        # segment boundaries, same rule as the high-level WM). Ceiling probe for
        # "does foresight help" (+5pp@50ep gate); the deployable version would
        # swap GT for WM-predicted futures (train/eval same sensor, per H1 law).
        self.future_slot = os.environ.get("TSSM_FUTURE_SLOT", "0") == "1"
        self._online_future_feat = None    # eval stash: sim-rendered waypoint view
        if self.future_slot and os.environ.get("TSSM_FUTURE_ONLINE", "0") == "1":
            # EVAL mode: the future view is rendered in the simulator and encoded
            # online per step (encode_future_view) — skip the ~13GB training
            # feature caches entirely.
            self._fut_feats = {"r2r": {}, "rxr": {}}
            self._fut_nseg = {}
            self._fut_maxchunk = int(os.environ.get("TSSM_FUTURE_MAXCHUNK", "10"))
            print("[TSSMMemoryAdapter] ORACLE-FUTURE slot on (ONLINE sim-rendered mode)")
        elif self.future_slot:
            import json as _json
            self._fut_feats = {}
            for tag, path in [("r2r", os.environ.get("TSSM_FUTURE_FEATS_R2R",
                    "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/siglip8f_p9_lm2/features.pt")),
                              ("rxr", os.environ.get("TSSM_FUTURE_FEATS_RXR",
                    "/cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/feat/siglip8f_p9_lm2/features.pt"))]:
                self._fut_feats[tag] = torch.load(path, map_location="cpu")
            self._fut_nseg = _json.load(open(os.environ.get("TSSM_FUTURE_NSEG",
                    "/cluster/scratch/ruiqzhang/NaVILA-Dataset/hwm_waypoints_nseg.json")))
            self._fut_maxchunk = int(os.environ.get("TSSM_FUTURE_MAXCHUNK", "10"))
            print(f"[TSSMMemoryAdapter] ORACLE-FUTURE slot on "
                  f"(r2r feats {len(self._fut_feats['r2r'])} eps, rxr {len(self._fut_feats['rxr'])})")

        self.feat_decoder = None
        # SINGLE-STATE mode (TSSM_HIST_DECODER=<ckpt>): decode K past-view gists
        # from ONE current state via the offline-trained history-unpacking head
        # (val cos .96-.99). Memory input = one 3072-dim vector; the K frames are
        # just its rendering. Decouples "#input states" from token/channel format.
        self.hist_decoder = None
        hd_path = os.environ.get("TSSM_HIST_DECODER", "")
        if self.pseudo_frames > 0 and hd_path:
            blob = torch.load(hd_path, map_location="cpu")
            hcfg = blob["cfg"]
            net = nn.Sequential(
                nn.LayerNorm(hcfg["state_dim"]),
                nn.Linear(hcfg["state_dim"], hcfg["hidden"]), nn.GELU(),
                nn.Linear(hcfg["hidden"], hcfg["hidden"]), nn.GELU(),
                nn.Linear(hcfg["hidden"], hcfg["K"] * hcfg["feat_dim"]),
            )
            sd = {k[len("net."):]: v for k, v in blob["model"].items() if k.startswith("net.")}
            net.load_state_dict(sd)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self.hist_decoder = net
            self._hist_K = hcfg["K"]
            self._hist_feat_dim = hcfg["feat_dim"]
            assert self._hist_K == self.pseudo_frames, "hist decoder K must match pseudo_frames"
            print(f"[TSSMMemoryAdapter] SINGLE-STATE history decoder loaded from {hd_path} "
                  f"(K={self._hist_K}, val_cos={blob.get('val_cos')})")
        if self.pseudo_frames > 0:
            self._load_feat_decoder(os.environ.get(
                "TSSM_DECODER_CKPT",
                "/cluster/scratch/ruiqzhang/tssm_full_ep_mix_td_kl04_kb09_act01_pose01.pt"))
        # online (eval) mode: roll TSSM forward in real time on Habitat-sim
        # observations.  Mutually exclusive with cache lookup at inference time:
        # if `_online_state_buf` is set before forward(), it overrides the
        # cache lookup; otherwise we fall back to cache (training path).
        self._online_state_buf: Optional[torch.Tensor] = None
        self._online_tssm = None      # NavigationTSSM, instantiated lazily
        self._online_siglip = None    # SigLIP encoder pack
        self._online_state = None     # current TSSMState for the (single) agent
        # FOREWARN-style history: keep every full state this episode so the
        # online buffer can expose the same K-slot sequence the cache path does
        # (lookup_state_sequence). Reset at each episode boundary.
        self._online_history: list = []
        self._online_ckpt_path = online_tssm_ckpt
        self._online_siglip_path = online_siglip_path
        # v12: 联合模式必须构造期建子模块, 否则优化器已按旧参数表建好 → 静默不更新
        if os.environ.get("TSSM_JOINT", "0") == "1":
            self._ensure_joint(torch.device("cpu"))

        # Projector: state_dim → (num_state_tokens * llm_dim)
        # TSSM_PROJ_SLICE=1 (qwen Stage-1 同款瘦身设计): 把 state 切成 N 段,
        # 每段过【共享】MLP + 可学习 token embedding。参数量 ~5M,与 N 无关。
        # 旧 fat 版末层 = hidden × N×4096 (64tok=5.4亿参数), 24G 卡 OOM 实锤。
        self.proj_slice = os.environ.get("TSSM_PROJ_SLICE", "0") == "1"
        # NAV-3L (07-22, 用户+导师锁): 主干3072→1536→1024 + raw-state skip →
        # gated fusion → 4×llm; 双辅助头 temporal(挂h,归一步号)+action(挂
        # mean-pool token,最近1/2步动作)直接监督, 破"联合LoRA=毒药"(辅助头给
        # projector独立于LoRA的锚)。progress头(instruction-conditioned)v1压住。
        # 门控 TSSM_PROJ_MODE=nav3l; 与 slice/xattn/fat 并列的第四分支。
        self.proj_mode = os.environ.get("TSSM_PROJ_MODE", "")
        # C-line (TSSM_PROJ_XATTN=1, 07-18): instruction-conditioned projector.
        # The TSSM state is instruction-blind (world model never saw text), so
        # the fat MLP cannot emit "which instruction chunk am I in" (gate
        # probe: R2R exact 38% ≈ chance+8, within±1 85%). Here the N output
        # slots cross-attend over the instruction's own FROZEN LLM embeddings
        # (reached via _embed_ref, set by llava_arch), with the state injected
        # into every query — each token becomes an instruction-chunk view of
        # the state. Same stage-1 data/supervision as the fat version. ~37M.
        self.proj_xattn = os.environ.get("TSSM_PROJ_XATTN", "0") == "1"
        if self.proj_xattn:
            H = projector_hidden
            self.x_state_in = nn.Sequential(nn.LayerNorm(state_dim),
                                            nn.Linear(state_dim, H, bias=True))
            self.x_instr_in = nn.Linear(llm_dim, H, bias=True)
            # raw embed_tokens carries NO position info (LLaMA position = RoPE,
            # applied inside attention) — without PE the instruction is a bag
            # of words and the ordinal "which chunk am I in" task is unlearnable.
            # Sinusoidal: zero params, valid at any length ≤ 512.
            import math as _math
            _pe = torch.zeros(512, H)
            _pos = torch.arange(512).unsqueeze(1).float()
            _div = torch.exp(torch.arange(0, H, 2).float() * (-_math.log(10000.0) / H))
            _pe[:, 0::2] = torch.sin(_pos * _div)
            _pe[:, 1::2] = torch.cos(_pos * _div)
            self.register_buffer("x_pe", _pe, persistent=False)
            self.x_query = nn.Parameter(torch.zeros(num_state_tokens, H))
            # std 0.5 ≈ st_h's post-Linear scale (LN→3072→1024 gives ~0.58):
            # at 0.02 the 16 slot identities would be 3% perturbations on a
            # shared state vector — queries homogenize, slots can't specialize.
            nn.init.normal_(self.x_query, std=0.5)
            self.x_attn = nn.ModuleList(
                [nn.MultiheadAttention(H, 8, batch_first=True) for _ in range(2)])
            self.x_ln_q = nn.ModuleList([nn.LayerNorm(H) for _ in range(2)])
            self.x_ln_kv = nn.ModuleList([nn.LayerNorm(H) for _ in range(2)])
            self.x_ffn = nn.ModuleList(
                [nn.Sequential(nn.LayerNorm(H), nn.Linear(H, 4 * H), nn.GELU(),
                               nn.Linear(4 * H, H)) for _ in range(2)])
            self.x_out = nn.Linear(H, llm_dim, bias=True)
            nn.init.normal_(self.x_out.weight, std=0.02)
            nn.init.zeros_(self.x_out.bias)
            # v2 sink fix (07-18): the original KV slot-0 was LN(st_h) — since
            # Q also carries st_h, the algebraic kinship made slot 0 an
            # attention black hole (paired-loss gap −35% @0.28ep vs fat arm).
            # Replace with a STATIC learnable sink: still a never-masked
            # graceful-degradation slot, but content-free — attending to it
            # yields nothing sample-specific, so instruction attention is the
            # only way to reduce loss. Optional train-time sink dropout
            # (TSSM_XATTN_SINK_DROP) forces excursions into the text tokens.
            self.x_sink = nn.Parameter(torch.randn(1, 1, H) * 0.5)
            # ---- dual readout (07-19, GPT-spec): world-state branch --------
            # Existing branch = "state asks the instruction where we are"
            # (progress). New branch = "instruction asks the STATE what
            # history/world info matters" — K/V must be MULTIPLE state slots,
            # not one vector. Slot tokenizer reuses the slice pattern (chunk
            # state + shared MLP + slot embeddings; cheap, real diversity).
            # Mode via TSSM_XATTN_MODE: progress_only(default, byte-compatible
            # with old ckpts) | world_only | dual | prog_direct.
            self.xattn_mode = os.environ.get("TSSM_XATTN_MODE", "progress_only")
            self.n_state_slots = int(os.environ.get("TSSM_STATE_SLOTS", "32"))
            _tot = num_state_tokens
            _dp, _dw = {"progress_only": (_tot, 0), "world_only": (0, _tot)}.get(
                self.xattn_mode, (_tot // 2, _tot - _tot // 2))
            self.n_prog = int(os.environ.get("TSSM_XATTN_NPROG", _dp))
            self.n_world = int(os.environ.get("TSSM_XATTN_NWORLD", _dw))
            self.x_fusion = os.environ.get("TSSM_XATTN_FUSION", "concat")
            self.world_aux = os.environ.get("TSSM_WORLD_AUX_LOSS", "none")
            if self.world_aux == "siglip_gist":
                gp = os.environ.get("TSSM_GIST_TARGETS",
                    "/cluster/scratch/ruiqzhang/tssm_gist_targets_p9.pt")
                self._gist_targets = torch.load(gp, map_location="cpu",
                                                weights_only=True)
                self.x_world_aux_head = nn.Linear(llm_dim, 9 * 1152)
                self.world_aux_w = float(os.environ.get("TSSM_WORLD_AUX_WEIGHT", "0.5"))
                print(f"[TSSMMemoryAdapter] world aux=siglip_gist "
                      f"targets={len(self._gist_targets)} eps w={self.world_aux_w}",
                      flush=True)
            if self.xattn_mode != "progress_only":
                assert state_dim % self.n_state_slots == 0, \
                    f"state_dim({state_dim}) % TSSM_STATE_SLOTS({self.n_state_slots}) != 0"
                chunk = state_dim // self.n_state_slots
                self.x_stok_ln_in = nn.LayerNorm(state_dim)
                self.x_stok_mlp = nn.Sequential(nn.Linear(chunk, 512), nn.GELU(),
                                                nn.Linear(512, H))
                # slot diversity comes from the per-slot state CHUNK content;
                # the identity embedding is positional seasoning only — keep it
                # small (0.1) or state-swaps barely move the slots at init.
                self.x_stok_emb = nn.Parameter(torch.zeros(self.n_state_slots, H))
                nn.init.normal_(self.x_stok_emb, std=0.1)
                self.x_stok_ln = nn.LayerNorm(H)
                if self.xattn_mode != "prog_direct":
                    self.x_wq_base = nn.Parameter(torch.zeros(self.n_world, H))
                    nn.init.normal_(self.x_wq_base, std=0.5)
                    self.x_wq_proj = nn.Linear(H, H, bias=True)
                    self.x_wattn = nn.ModuleList(
                        [nn.MultiheadAttention(H, 8, batch_first=True) for _ in range(2)])
                    self.x_wln_q = nn.ModuleList([nn.LayerNorm(H) for _ in range(2)])
                    self.x_wln_kv = nn.ModuleList([nn.LayerNorm(H) for _ in range(2)])
                    self.x_wffn = nn.ModuleList(
                        [nn.Sequential(nn.LayerNorm(H), nn.Linear(H, 4 * H), nn.GELU(),
                                       nn.Linear(4 * H, H)) for _ in range(2)])
                if self.x_fusion == "residual":
                    self.x_fuse_alpha = nn.Parameter(torch.tensor(0.1))
                # ---- GPT-review增量(07-19晚, 全部默认=现役行为) ----------
                # world blocks 数量可配 (下一组实验 1 vs 2)
                nb = int(os.environ.get("TSSM_WORLD_NUM_BLOCKS", "2"))
                if self.xattn_mode != "prog_direct" and nb != 2:
                    self.x_wattn = self.x_wattn[:nb]
                    self.x_wln_q = self.x_wln_q[:nb]
                    self.x_wln_kv = self.x_wln_kv[:nb]
                    self.x_wffn = self.x_wffn[:nb]
                # factorized_global StateTokenizer (每槽看全量state, ~8.6M)
                self.x_stok_kind = os.environ.get("TSSM_STATE_TOKENIZER", "chunk")
                if self.x_stok_kind == "factorized_global":
                    bdim = int(os.environ.get("TSSM_STATE_TOKENIZER_BOTTLENECK", "256"))
                    self.x_stok_g1 = nn.Linear(state_dim, bdim)
                    self.x_stok_g2 = nn.Linear(bdim, self.n_state_slots * H)
                # branch-specific LN before shared x_out (默认关: 保护现役ckpt,
                # LayerNorm即使init也非恒等)
                if os.environ.get("TSSM_BRANCH_OUT_NORM", "0") == "1":
                    self.x_prog_onorm = nn.LayerNorm(H)
                    self.x_world_onorm = nn.LayerNorm(H)
                # world delta gain: qw0 + g*(qw-qw0); float 或 "learnable"
                _g = os.environ.get("TSSM_WORLD_DELTA_GAIN", "1.0")
                if _g == "learnable":
                    self.x_wgain = nn.Parameter(torch.tensor(2.0))
                else:
                    self.x_wgain = float(_g)
                # world专属辅助监督: 历史地标召回 (multi-label BCE, 经rssm_loss
                # 管路 ×tssm_loss_weight(0.1) 汇入总损失; 表由
                # build_landmark_labels.py 生成)
                if self.world_aux == "landmark_recall":
                    lp = os.environ.get("TSSM_LANDMARK_LABELS",
                        "/cluster/scratch/ruiqzhang/tssm_landmark_labels.pt")
                    blob = torch.load(lp, map_location="cpu")
                    self._lm_first_seen = blob["first_seen"]
                    self._lm_V = len(blob["vocab"])
                    self.x_world_aux_head = nn.Linear(H, self._lm_V)
                    self.world_aux_w = float(os.environ.get("TSSM_WORLD_AUX_WEIGHT", "1.0"))
                    print(f"[TSSMMemoryAdapter] world aux=landmark_recall "
                          f"V={self._lm_V} eps={len(self._lm_first_seen)}", flush=True)
                # 可选1层指令上下文化器(默认关; layer-0嵌入的轻量补丁)
                if os.environ.get("TSSM_INSTR_CONTEXTUALIZER", "0") == "1":
                    self.x_instr_ctx = nn.TransformerEncoderLayer(
                        H, 8, 4 * H, batch_first=True, norm_first=True)
                self._xattn_dbg_n = 0
            self.proj = None
            self.proj_slice = False
            self._instr_ids: dict = {}
            self._instr_tok = None
            self._instr_map_loaded = False
            self._online_instr_ids = None
        elif self.proj_slice:
            assert state_dim % num_state_tokens == 0, \
                f"slice mode needs state_dim({state_dim}) % N({num_state_tokens}) == 0"
            chunk = state_dim // num_state_tokens
            self.slice_norm = nn.LayerNorm(state_dim)
            self.slice_mlp = nn.Sequential(
                nn.Linear(chunk, 1024, bias=True), nn.GELU(),
                nn.Linear(1024, llm_dim, bias=True),
            )
            self.slice_tok_emb = nn.Parameter(torch.zeros(num_state_tokens, llm_dim))
            nn.init.normal_(self.slice_mlp[-1].weight, std=0.02)
            nn.init.zeros_(self.slice_mlp[-1].bias)
            nn.init.normal_(self.slice_tok_emb, std=0.02)
            self.proj = None
        elif self.proj_mode == "nav3l":
            # 主干 + raw-state skip + gated fusion + N·llm 出口
            # 07-22: 中间宽度改吃 projector_hidden (config rssm_v4_proj_hidden)。
            # 老 nav3l ckpt config=1024 → 形状与硬编码时代逐位一致, 向后兼容。
            _H = int(projector_hidden)
            self.n3_hid = _H
            self.n3_ln = nn.LayerNorm(state_dim)
            self.n3_l1 = nn.Linear(state_dim, int(1.5 * _H))
            self.n3_l2 = nn.Linear(int(1.5 * _H), _H)
            self.n3_skip = nn.Linear(state_dim, _H)
            self.n3_gate = nn.Linear(2 * _H, _H)
            nn.init.constant_(self.n3_gate.bias, -1.5)   # 起步偏 skip
            self.n3_out = nn.Linear(_H, num_state_tokens * llm_dim)
            nn.init.normal_(self.n3_out.weight, std=0.02)
            nn.init.zeros_(self.n3_out.bias)
            # 辅助头 (x_aux_ 前缀 → 部署自动剥离; v1 只 temporal+action)
            self.x_aux_temporal = nn.Sequential(
                nn.Linear(_H, 256), nn.SiLU(), nn.Linear(256, 1))
            self.x_aux_action = nn.Sequential(
                nn.Linear(llm_dim, 256), nn.SiLU(), nn.Linear(256, 2 * 9))
            # 08-18 hist8: 槽序嵌入(内容级次序印章, 零初始化=开局无扰动)。仅多槽时建,
            # 老K=1臂不受影响; 位置在融合层f上, 先盖章再出token。
            if self.history_states > 1:
                self.n3_slot_emb = nn.Parameter(torch.zeros(16, _H))
            self.n3_step_norm = float(os.environ.get("TSSM_NAV3L_STEPNORM", "64"))
            self.n3_a2w = float(os.environ.get("TSSM_NAV3L_A2W", "0.4"))
            self.proj = None
            self.proj_slice = False

            self._n3_dbg = 0
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(state_dim),
                nn.Linear(state_dim, projector_hidden, bias=True),
                nn.GELU(),
                nn.Linear(projector_hidden, num_state_tokens * llm_dim, bias=True),
            )
            # Init the last layer small so the new tokens don't shock LLaMA at
            # step 0. 把最后一层初始化得很小，避免训练初期把 LLaMA 打懵。
            nn.init.normal_(self.proj[-1].weight, std=0.02)
            nn.init.zeros_(self.proj[-1].bias)
        # ---- xattn-tape (07-28 用户拍板"开工"): 情景记忆读取器 ----
        # h(16 state token)=参数化记忆; 本模块=对整条 (z_t, a_t) 原始磁带做
        # cross-attention, K个query 检索历史, 输出 K 个 token 追加在 state token 后。
        # 磁带来源: 训练=state缓存每行的 z 段(2048:3072) + tssm_action_seqs;
        #          评测=step_online 逐步累积(训评同构)。TSSM_TAPE_XATTN=1 开。
        self.tape_xattn = os.environ.get("TSSM_TAPE_XATTN", "0") == "1"
        if self.tape_xattn:
            _TD = int(os.environ.get("TSSM_TAPE_DMODEL", "512"))
            self.tape_k = int(os.environ.get("TSSM_TAPE_QUERIES", "16"))
            self.tape_zdim = int(os.environ.get("TSSM_TAPE_ZDIM", "1024"))
            self.tape_zoff = int(os.environ.get("TSSM_TAPE_ZOFF", "2048"))
            self.tape_max = int(os.environ.get("TSSM_TAPE_MAX", "200"))
            self.tp_z_in = nn.Linear(self.tape_zdim, _TD)
            self.tp_a_emb = nn.Embedding(10, 64)
            self.tp_a_in = nn.Linear(64, _TD)
            self.tp_pos = nn.Parameter(torch.randn(self.tape_max, _TD) * 0.02)
            self.tp_q = nn.Parameter(torch.randn(self.tape_k, _TD) * 0.02)
            self.tp_lnq1 = nn.LayerNorm(_TD); self.tp_lnk1 = nn.LayerNorm(_TD)
            self.tp_attn1 = nn.MultiheadAttention(_TD, 8, batch_first=True)
            self.tp_ffn1 = nn.Sequential(nn.LayerNorm(_TD), nn.Linear(_TD, 2*_TD),
                                         nn.SiLU(), nn.Linear(2*_TD, _TD))
            self.tp_lnq2 = nn.LayerNorm(_TD)
            self.tp_attn2 = nn.MultiheadAttention(_TD, 8, batch_first=True)
            self.tp_ffn2 = nn.Sequential(nn.LayerNorm(_TD), nn.Linear(_TD, 2*_TD),
                                         nn.SiLU(), nn.Linear(2*_TD, _TD))
            self.tp_out = nn.Linear(_TD, llm_dim)
            nn.init.normal_(self.tp_out.weight, std=0.02)
            nn.init.zeros_(self.tp_out.bias)
            self._online_actions = []
            # 07-28 用户点单: 磁带专属辅助损失, 与"当前该关注什么"的16个query分开,
            # 避免任务混杂。8个独立query, 各自负责回归均匀抽取的历史时刻SigLIP embedding
            # (feat的语义段, 不是z/动作), 逼磁带真正检索原始语义而非重复state已有的东西。
            self.tape_aux_w = float(os.environ.get("TSSM_TAPE_AUX_W", "0.05"))
            self.tape_aux_n = int(os.environ.get("TSSM_TAPE_AUX_N", "8"))
            self.tape_sem_dim = int(os.environ.get("TSSM_TAPE_SEM_DIM", "1152"))  # SigLIP段维度
            if self.tape_aux_w > 0:
                self.tp_aux_q = nn.Parameter(torch.randn(self.tape_aux_n, _TD) * 0.02)
                self.tp_aux_out = nn.Linear(_TD, self.tape_sem_dim)
                nn.init.normal_(self.tp_aux_out.weight, std=0.02)
                nn.init.zeros_(self.tp_aux_out.bias)
            print(f"[TSSMMemoryAdapter] TAPE-XATTN on: K={self.tape_k} d={_TD} "
                  f"z[{self.tape_zoff}:{self.tape_zoff+self.tape_zdim}] max={self.tape_max} "
                  f"aux_w={self.tape_aux_w} aux_n={self.tape_aux_n}", flush=True)

        # ---- 四头 (07-20): 多辅助头挂最终token, 训练期专用, 三种投影模式通用 --
        # TSSM_AUX_HEADS=gist,landmark,act_hist (逗号多选)。narr 由 LLM CE
        # 承担, 三头分管 看过什么/路过什么/做过什么。权重经 rssm_loss 管路
        # ×0.1 汇入总损失: 默认 3.0/3.0/1.5 -> 有效 0.3/0.3/0.15。
        self.aux_heads = [h.strip() for h in
                          os.environ.get("TSSM_AUX_HEADS", "").split(",") if h.strip()]
        if "gist" in self.aux_heads:
            if getattr(self, "_gist_targets", None) is None:
                gp = os.environ.get("TSSM_GIST_TARGETS",
                    "/cluster/scratch/ruiqzhang/tssm_gist_targets_p9.pt")
                self._gist_targets = torch.load(gp, map_location="cpu",
                                                weights_only=True)
            self.x_aux_gist_head = nn.Linear(llm_dim, 9 * 1152)
            self.aux_w_gist = float(os.environ.get("TSSM_AUX_W_GIST", "3.0"))
        if "landmark" in self.aux_heads:
            lp = os.environ.get("TSSM_LANDMARK_LABELS",
                "/cluster/scratch/ruiqzhang/tssm_landmark_labels.pt")
            _blob = torch.load(lp, map_location="cpu", weights_only=False)
            self._aux_lm_first_seen = _blob["first_seen"]
            self._aux_lm_V = len(_blob["vocab"])
            self.x_aux_lm_head = nn.Linear(llm_dim, self._aux_lm_V)
            self.aux_w_lm = float(os.environ.get("TSSM_AUX_W_LANDMARK", "3.0"))
        if "act_hist" in self.aux_heads:
            ap = os.environ.get("TSSM_ACTION_SEQS",
                "/cluster/scratch/ruiqzhang/tssm_action_seqs.pt")
            self._act_seqs = torch.load(ap, map_location="cpu", weights_only=True)
            self.x_aux_act_head = nn.Linear(llm_dim, 3)   # fwd/left/right 族
            self.aux_w_act = float(os.environ.get("TSSM_AUX_W_ACTHIST", "1.5"))
            self.aux_act_win = int(os.environ.get("TSSM_AUX_ACT_WIN", "20"))
        if "gist_next" in self.aux_heads:
            # 向外推一步 (07-20 用户令, FutureNav future-gen 同款): 预测 t+1 帧
            # 的SigLIP九宫格, 动作无关的短时视觉演化先验。终点行掩掉。
            if getattr(self, "_gist_targets", None) is None:
                gp = os.environ.get("TSSM_GIST_TARGETS",
                    "/cluster/scratch/ruiqzhang/tssm_gist_targets_p9.pt")
                self._gist_targets = torch.load(gp, map_location="cpu",
                                                weights_only=True)
            self.x_aux_gistnext_head = nn.Linear(llm_dim, 9 * 1152)
            self.aux_w_gistnext = float(os.environ.get("TSSM_AUX_W_GISTNEXT", "1.5"))
        if "act_seq" in self.aux_heads:
            # 顺序敏感版 (07-20 用户令): 末L步动作逐位置9细类分类。
            # 直方图丢顺序("先转后走"="先走后转"), 序列头保顺序+幅度。
            ap = os.environ.get("TSSM_ACTION_SEQS",
                "/cluster/scratch/ruiqzhang/tssm_action_seqs.pt")
            if getattr(self, "_act_seqs", None) is None:
                self._act_seqs = torch.load(ap, map_location="cpu", weights_only=True)
            self.aux_actseq_len = int(os.environ.get("TSSM_AUX_ACTSEQ_LEN", "10"))  # 最近10步(07-20定): 更远的顺序无意义, 由状态自己记
            self.x_aux_actseq_head = nn.Linear(llm_dim, self.aux_actseq_len * 9)
            self.aux_w_actseq = float(os.environ.get("TSSM_AUX_W_ACTSEQ", "1.5"))
        # 次态对齐 (07-20, WoG式Query&Align移植): LoRA阶段教LLM预判下一步记忆。
        # 头读LLM隐状态(在llava_llama侧调用), 目标=冻结投影的s_{t+1} token。
        if os.environ.get("TSSM_ALIGN_AUX", "0") == "1":
            self.x_aux_align_head = nn.Linear(llm_dim, llm_dim)
            self.align_w = float(os.environ.get("TSSM_ALIGN_W", "1.0"))
            print(f"[TSSMMemoryAdapter] align-next aux ON w={self.align_w} (有效x0.1)",
                  flush=True)
        # 读出检索 (07-29, TSSM_READNCE门控): 两个查询头读LLM在记忆token位置的
        # 隐状态, 在冻结的ncev5钥匙空间(nce_k_head投影+L2归一)里做同集帧检索:
        #   retro头: 检索 t-k 帧 —— 逼LLM真把state里已证明存在的历史读出来;
        #   dyn头:   动作窗嵌入条件下检索 t+k 帧 —— FutureNav式动力学grounding。
        # 钥匙离线预算(build_readnce_keys.py), 同集其余帧天然是难负样本。
        if os.environ.get("TSSM_READNCE", "0") == "1":
            _kp = os.environ.get("TSSM_READNCE_KEYS",
                                 "/cluster/scratch/ruiqzhang/tssm_readnce_keys_ncev5.pt")
            _blob = torch.load(_kp, map_location="cpu", weights_only=True)
            self._readnce_keys = _blob["keys"] if isinstance(_blob, dict) and "keys" in _blob else _blob
            self.readnce_retro_lags = [int(x) for x in os.environ.get(
                "TSSM_READNCE_RETRO_LAGS", "3,5,10").split(",") if x.strip()]
            self.readnce_dyn_lags = [int(x) for x in os.environ.get(
                "TSSM_READNCE_DYN_LAGS", "1,3").split(",") if x.strip()]
            _dc = next(iter(self._readnce_keys.values())).shape[-1]
            self.readnce_lag_emb = nn.Embedding(max(1, len(self.readnce_retro_lags)), 64)
            self.readnce_dyn_lag_emb = nn.Embedding(max(1, len(self.readnce_dyn_lags)), 64)
            self.readnce_act_emb = nn.Embedding(10, 32)   # 动作9细类(1..9) + pad0
            # 动作窗编码 (07-31修): mean会丢顺序("先转后走"="先走后转", 07-20 actseq头
            # 同款教训); concat=固定k_max槽位保序拼接, pad用0号嵌入。默认concat。
            self.readnce_act_enc = os.environ.get("TSSM_READNCE_ACT_ENC", "concat")
            self.readnce_act_kmax = max(self.readnce_dyn_lags) if self.readnce_dyn_lags else 1
            _act_dim = 32 * self.readnce_act_kmax if self.readnce_act_enc == "concat" else 32
            self.readnce_q_head = nn.Sequential(
                nn.Linear(llm_dim + 64, 512), nn.GELU(), nn.Linear(512, _dc))
            self.readnce_dyn_head = nn.Sequential(
                nn.Linear(llm_dim + 64 + _act_dim, 512), nn.GELU(), nn.Linear(512, _dc))
            if getattr(self, "_act_seqs", None) is None:
                self._act_seqs = torch.load(
                    os.environ.get("TSSM_ACTION_SEQS",
                                   "/cluster/scratch/ruiqzhang/tssm_action_seqs.pt"),
                    map_location="cpu", weights_only=True)
            self.readnce_tau = float(os.environ.get("TSSM_READNCE_TAU", "0.1"))
            self.readnce_w = float(os.environ.get("TSSM_READNCE_W", "0.2"))
            self.readnce_dyn_w = float(os.environ.get("TSSM_READNCE_DYN_W", "0.2"))
            print(f"[TSSMMemoryAdapter] READNCE on: keys={len(self._readnce_keys)}eps "
                  f"dim={_dc} retro_lags={self.readnce_retro_lags} "
                  f"dyn_lags={self.readnce_dyn_lags} w={self.readnce_w}/{self.readnce_dyn_w} "
                  f"tau={self.readnce_tau}", flush=True)
        if self.aux_heads:
            print(f"[TSSMMemoryAdapter] aux heads={self.aux_heads} "
                  f"raw w=gist:{getattr(self,'aux_w_gist',None)} "
                  f"lm:{getattr(self,'aux_w_lm',None)} "
                  f"act:{getattr(self,'aux_w_act',None)} (有效值再x0.1)",
                  flush=True)
        # ---- Run C (week1 gates): oracle-future 注入, TSSM_ORACLE_FUT=1 门控 ----
        # 布局: [历史4token][<FUT>标记1][fut_h2][fut_h5][fut_h8] = 8 token。
        # 红线: 未来块不携带任何动作文本(答案泄漏); 标记/时距/类型全部是可学习
        # 嵌入, 不是词表token, llava_arch 无需改动。
        self.oracle_fut = os.environ.get("TSSM_ORACLE_FUT", "0") == "1"
        if self.oracle_fut:
            self.fut_marker = nn.Parameter(torch.zeros(1, llm_dim))
            self.fut_hor_emb = nn.Parameter(torch.zeros(3, llm_dim))   # h=2/5/8
            self.fut_type_emb = nn.Parameter(torch.zeros(1, llm_dim))  # imagined
            nn.init.normal_(self.fut_marker, std=0.02)
            nn.init.normal_(self.fut_hor_emb, std=0.02)
            nn.init.normal_(self.fut_type_emb, std=0.02)
            self.fut_proj_mode = os.environ.get("TSSM_FUT_PROJ", "shared")
            if self.fut_proj_mode == "separate":
                self.fut_proj = nn.Linear(self.state_dim, llm_dim)
            self._oracle_fut_cache = None
            self._online_fut_buf = None   # eval: navila_trainer 塞 [B,3,state_dim]
            print(f"[TSSMMemoryAdapter] ORACLE-FUT ON proj={self.fut_proj_mode} "
                  f"horizons=[2,5,8]", flush=True)

        # ---- LMK-Bank v1 (07-22, 门控 TSSM_LMK_BANK=1): 地标记忆银行 ----
        # 8槽×pooled SigLIP 1152; 训练=离线槽位表回放, 在线=OnlineBank 同一写入规则。
        self.lmk_bank_on = os.environ.get("TSSM_LMK_BANK", "0") == "1"
        if self.lmk_bank_on:
            import sys as _sys
            _here = os.path.dirname(os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            from lmk_bank import OnlineBank as _OB, L_SLOTS as _LS
            self.lmk_L = _LS
            self.lmk_ln = nn.LayerNorm(1152)
            self.lmk_proj = nn.Linear(1152, llm_dim)
            nn.init.normal_(self.lmk_proj.weight, std=0.02)
            nn.init.zeros_(self.lmk_proj.bias)
            self.lmk_slot_emb = nn.Parameter(torch.zeros(_LS, llm_dim))
            nn.init.normal_(self.lmk_slot_emb, std=0.02)
            self.lmk_null = nn.Parameter(torch.zeros(llm_dim))       # 空槽占位
            self.lmk_phase = nn.Linear(1, llm_dim, bias=False)       # 写入步相位
            nn.init.normal_(self.lmk_phase.weight, std=0.02)
            self.x_aux_lmk = nn.Linear(llm_dim, 512)                 # multi-hot 地标读出锚
            self.lmk_aux_w = float(os.environ.get("TSSM_LMK_AUX_W", "0.5"))
            self._lmk_pooled = None    # {key: fp16 [T,1152]} 懒加载
            self._lmk_slots = None     # {key: int16 [T,L]}
            self._lmk_labels = None    # first_seen v3
            self._lmk_online = _OB()
            print(f"[TSSMMemoryAdapter] LMK-BANK ON L={_LS} aux_w={self.lmk_aux_w}", flush=True)

        # ---- LMK-Bank v2 (07-23, 门控 TSSM_LMK_MODE=state): 指令名词驱动的 state 银行 ----
        # 与 v1(SigLIP payload) 独立并存。payload=TSSM state 快照(3072), 投影器对齐
        # F16H2K 主干 H=2048 两层去skip; 每槽1token; 词嵌入回归锚(靶表离线预建)。
        # 训练: ledger[ep] 的 (名词,触发步) → cache 取 state → 投影 → ≤L token。
        # 评测: navila_trainer 每步喂 lmk_online_push(state, step, noun_idx), 本类查表投影。
        self.lmk_mode = os.environ.get("TSSM_LMK_MODE", "")
        if self.lmk_mode == "state":
            _Hl = 2048
            self.lmk_L = int(os.environ.get("TSSM_LMK_SLOTS_N", "8"))
            self.lmk_ln = nn.LayerNorm(self.state_dim)
            self.lmk_l1 = nn.Linear(self.state_dim, self.state_dim)   # 3072→3072
            self.lmk_l2 = nn.Linear(self.state_dim, _Hl)             # 3072→2048
            self.lmk_out = nn.Linear(_Hl, llm_dim)                   # 2048→4096
            nn.init.normal_(self.lmk_out.weight, std=0.02)
            nn.init.zeros_(self.lmk_out.bias)
            self.lmk_slot_emb = nn.Parameter(torch.zeros(self.lmk_L, llm_dim))
            nn.init.normal_(self.lmk_slot_emb, std=0.02)
            self.lmk_null = nn.Parameter(torch.zeros(llm_dim))
            self.lmk_phase = nn.Linear(1, llm_dim, bias=False)       # 相对步差相位
            nn.init.normal_(self.lmk_phase.weight, std=0.02)
            self.x_aux_lmkw = nn.Linear(llm_dim, llm_dim)            # 词嵌入锚头 W (x_aux_→部署可弃)
            self.lmk_aux_w = float(os.environ.get("TSSM_LMK_AUX_W", "0.5"))
            self.lmk_phase_norm = float(os.environ.get("TSSM_LMK_PHASE_NORM", "64"))
            self._lmk_ledger = None      # {ep: {"nouns":[...], "fires":[(ni,step)]}}
            self._lmk_anchor = None      # {noun: [llm_dim] 冻结词嵌入靶}
            self._lmk_ob = []            # eval 在线银行: [(state[3072], step, noun)]
            self._lmk_dbg = 0
            # ---- v3 子门控 (07-23, TSSM_LMK_V3=1): 账本v3(槽位表) + 动作双头 ----
            # 数据源换成 build_lmk_ledger_v3 的 {ep:{slots[T,L], anchor{步:词|None}}}:
            # 因果律在建账时烤死(slots[t] 只含 ≤t 的写入), 前向零过滤。
            # 动作双头(用户07-23提议): 槽token → CE(prev_act)+CE(curr_act) 各10类,
            # 标签自专家动作序列 — 全页覆盖的密监督, 补词锚(45%页)的稀疏。
            self.lmk_v3 = os.environ.get("TSSM_LMK_V3", "0") == "1"
            if self.lmk_v3:
                self.x_aux_lmka = nn.Linear(llm_dim, 20)   # 双头合一: 前10=prev, 后10=curr
                self.lmk_act_w = float(os.environ.get("TSSM_LMK_ACT_W", "0.1"))
                self._lmk_v3_ledger = None   # lazy: {ep: {"slots","anchor"}}
                self._lmk_acts = None        # lazy: {key: 动作序列}
                print(f"[TSSMMemoryAdapter] LMK-v3(slots ledger) ON act_w={self.lmk_act_w}",
                      flush=True)
            print(f"[TSSMMemoryAdapter] LMK-BANK-v2(state) ON L={self.lmk_L} "
                  f"aux_w={self.lmk_aux_w}", flush=True)

        stage1_projector_path = stage1_projector_path or os.environ.get("TSSM_STAGE1_PROJECTOR_PATH")
        # nav3l 自建投影器, 从无 stage-1; 防御 env 泄漏 TSSM_STAGE1_PROJECTOR_PATH
        # (07-22: nanny :- 空串陷阱曾把 stage1_proj4.pt 漏进来 -> self.proj None 崩)
        if stage1_projector_path and self.proj_mode != "nav3l":
            self.load_stage1_projector(stage1_projector_path)
        # Load the state cache (lazy: only load if path provided)
        self.cache: dict = {}
        if cache_path is not None:
            self.load_cache(cache_path)

    def _load_feat_decoder(self, ckpt_path: str) -> None:
        """Load the FROZEN TSSM feat_decoder (full state 3072 -> SigLIP 1152)."""
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from navigation_tssm_feat import FeatDecoder, TSSMConfig  # noqa: E402
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = TSSMConfig(**{k: v for k, v in blob.get("config", {}).items()
                            if k in TSSMConfig.__dataclass_fields__})
        deter_dim = (cfg.n_layers * cfg.d_model if getattr(cfg, "deter_type", "last") == "concat_o"
                     else cfg.h_dim)
        dec = FeatDecoder(cfg, deter_dim=deter_dim)
        sd = {k[len("feat_decoder."):]: v for k, v in blob["model"].items()
              if k.startswith("feat_decoder.")}
        dec.load_state_dict(sd)
        # H2 (TSSM_TRAIN_DECODER=1): FOREWARN-style trainable translator. Keeps
        # the reconstruction init but lets task gradients re-purpose the
        # projection to smuggle non-visual state info (progress, action history)
        # into frame features — the info the frozen projection destroys
        # (probe: state 57% -> gist 33% progress readability).
        self.train_decoder = os.environ.get("TSSM_TRAIN_DECODER", "0") == "1"
        for p in dec.parameters():
            p.requires_grad_(self.train_decoder)
        if not self.train_decoder:
            dec.eval()
        self.feat_decoder = dec
        # v5 (lm2-p9): decoder may emit P patch blocks (feat_dim = P*1152) in a
        # feat_scale'd space; decode_gists reshapes to [B,K,P,1152] and undoes
        # the scale so gists land in raw SigLIP space.
        self._gist_scale = float(getattr(cfg, "feat_scale", 1.0))
        self._gist_feat_dim = int(cfg.feat_dim)
        print(f"[TSSMMemoryAdapter] pseudo-frame feat_decoder loaded from {ckpt_path} "
              f"(in={deter_dim + cfg.stoch * cfg.classes}, out={cfg.feat_dim}, "
              f"feat_scale={self._gist_scale})")

        # TEST-TIME ADAPTATION (TSSM_DECODER_ADAPT=1, eval only): the frozen
        # decoder is a train-scene artifact; in unseen buildings the online
        # state renders poorly (cos .95 -> .56, key-vs-lock verdict A).
        # Self-supervise a PER-EPISODE COPY on its native task — reconstruct the
        # current frame feat f_t from state s_t — using pairs the agent already
        # produces. GT-clean (only self-seen frames). Original stays frozen; the
        # copy is reset each episode. Off by default ->老线不受影响.
        self._dec_adapt = os.environ.get("TSSM_DECODER_ADAPT", "0") == "1"
        # 默认=扫参优胜组(adam 5e-4 ×2, 仿真后1/3 cos 0.934; SGD版只有0.63已弃)
        self._adapt_lr = float(os.environ.get("DECODER_ADAPT_LR", "5e-4"))
        self._adapt_warmup = int(os.environ.get("DECODER_ADAPT_WARMUP", "4"))
        self._adapt_bs = int(os.environ.get("DECODER_ADAPT_BS", "16"))
        self._adapt_steps = int(os.environ.get("DECODER_ADAPT_STEPS", "2"))
        self._adapt_dec = None            # per-episode trainable copy
        self._adapt_opt = None
        self._adapt_buf = []              # [(s_t 3072, f_t feat_dim)] this episode
        if self._dec_adapt:
            print(f"[DEC-ADAPT] on: lr={self._adapt_lr} warmup={self._adapt_warmup} "
                  f"bs={self._adapt_bs} steps/frame={self._adapt_steps}")

    def _adapt_reset_episode(self):
        """Fresh decoder copy + optimizer for a new episode (TTA)."""
        if not getattr(self, "_dec_adapt", False):
            return
        import copy
        self._adapt_dec = copy.deepcopy(self.feat_decoder)
        for p in self._adapt_dec.parameters():
            p.requires_grad_(True)
        self._adapt_dec.train()
        self._adapt_opt = torch.optim.Adam(self._adapt_dec.parameters(),
                                           lr=self._adapt_lr)
        self._adapt_buf = []

    def _adapt_step(self, state_vec, feat_vec):
        """Buffer the (state, current-frame-feat) pair and take a few SGD steps
        on the episode copy once past warmup. Called from step_online."""
        if not getattr(self, "_dec_adapt", False):
            return
        if self._adapt_dec is None:
            self._adapt_reset_episode()
        dev = next(self._adapt_dec.parameters()).device
        dt = next(self._adapt_dec.parameters()).dtype
        self._adapt_buf.append((state_vec.detach().flatten()[:self.cache_state_dim].to(dev, dt),
                                feat_vec.detach().flatten().to(dev, dt)))
        if len(self._adapt_buf) < self._adapt_warmup:
            return
        # 调用链在trainer的 no_grad 块里 —— TTA更新必须显式开grad
        with torch.enable_grad():
            for _ in range(self._adapt_steps):
                k = min(self._adapt_bs, len(self._adapt_buf))
                idx = torch.randint(0, len(self._adapt_buf), (k,))
                s = torch.stack([self._adapt_buf[i][0] for i in idx])
                f = torch.stack([self._adapt_buf[i][1] for i in idx])
                pred = self._adapt_dec(s) / self._gist_scale
                loss = torch.nn.functional.mse_loss(pred, f)
                self._adapt_opt.zero_grad(); loss.backward(); self._adapt_opt.step()

    @staticmethod
    def _to_patch_gists(gists: torch.Tensor) -> torch.Tensor:
        """Normalize decoder output to [B, K, P, 1152].

        P=1 for the legacy pooled decoder (out=1152); P=9 for the v5 lm2-p9
        decoder (out=10368 = 9*1152)."""
        B, K, out = gists.shape
        if out % 1152 == 0 and out // 1152 > 1:
            return gists.view(B, K, out // 1152, 1152)
        return gists.unsqueeze(2)

    @torch.no_grad()
    def decode_gists(self, episode_ids, steps, device) -> torch.Tensor:
        """[B, K, P, 1152] SigLIP-space gists of K history slots (K=pseudo_frames).
        P = spatial patch blocks per gist (1 = legacy pooled, 9 = v5 3x3 grid).
        Uses the online buffer when step_online() pre-filled it (eval), else the
        cache (training). Slots follow _history_indices (same as token mode)."""
        K = self.pseudo_frames
        if self.hist_decoder is not None:
            # ONE current state -> K past-view gists (single-state architecture).
            if self._online_state_buf is not None:
                st = self._online_state_buf.to(device).float()[:, -1, : self.cache_state_dim]
                self._online_state_buf = None
            else:
                old_h = self.history_states
                self.history_states = 1
                try:
                    st = self.lookup_state_sequence(episode_ids, steps, device)[:, 0, :]
                finally:
                    self.history_states = old_h
            dec = self.hist_decoder.to(device)
            d = next(dec.parameters()).dtype
            gists = dec(st.to(device=device, dtype=d)).view(
                st.shape[0], self._hist_K, self._hist_feat_dim)
            gists = self._to_patch_gists(gists)
            if os.environ.get("TSSM_ZERO_TOKENS", "0") == "1":
                gists = torch.zeros_like(gists)
            return gists
        if self._online_state_buf is not None:
            states = self._online_state_buf.to(device).float()          # [1, K', full]
            self._online_state_buf = None
            if states.shape[1] < K:                                     # pad left
                pad = states[:, :1].expand(-1, K - states.shape[1], -1)
                states = torch.cat([pad, states], dim=1)
            states = states[:, -K:, : self.cache_state_dim]
        else:
            old = self.history_states
            self.history_states = K
            try:
                states = self.lookup_state_sequence(episode_ids, steps, device)
            finally:
                self.history_states = old
        # TTA: render with the per-episode adapted copy when active.
        dec = (self._adapt_dec if getattr(self, "_dec_adapt", False)
               and self._adapt_dec is not None else self.feat_decoder).to(device)
        dec_dtype = next(dec.parameters()).dtype
        if getattr(self, "_dec_adapt", False) and self._adapt_dec is not None:
            with torch.no_grad():
                gists = dec(states.to(device=device, dtype=dec_dtype))
        elif getattr(self, "train_decoder", False) and self.training and not torch.is_grad_enabled():
            # decode_gists is @torch.no_grad for the frozen path; H2 needs grads.
            with torch.enable_grad():
                gists = dec(states.to(device=device, dtype=dec_dtype))
        else:
            gists = dec(states.to(device=device, dtype=dec_dtype))      # [B, K, feat_dim]
        if self._gist_scale != 1.0:
            gists = gists * self._gist_scale                            # back to raw space
        gists = self._to_patch_gists(gists)                             # [B, K, P, 1152]
        # Memory-attribution ablation (same env as token mode): zero the gists.
        if os.environ.get("TSSM_ZERO_TOKENS", "0") == "1":
            gists = torch.zeros_like(gists)
        return gists

    @staticmethod
    def _future_boundaries(T: int, n_seg: int, max_chunk: int):
        import math
        n_seg = max(1, min(n_seg, T - 1))
        bounds = sorted(set(round(k * (T - 1) / n_seg) for k in range(n_seg + 1)))
        out = [bounds[0]]
        for b in bounds[1:]:
            seg = b - out[-1]
            if seg > max_chunk:
                n_sub = int(math.ceil(seg / max_chunk))
                for j in range(1, n_sub):
                    out.append(out[-1] + seg // n_sub)
            out.append(b)
        return sorted(set(out))

    @torch.no_grad()
    def get_future_feats(self, episode_ids, steps, device) -> torch.Tensor:
        """[B, P, 1152] GT features of the next waypoint (oracle-future slot)."""
        if getattr(self, "_online_future_feat", None) is not None:
            f = self._online_future_feat
            self._online_future_feat = None             # consume once per step
            return f.unsqueeze(0).to(device)            # [1, P, 1152]
        outs = []
        for ep, st in zip(episode_ids, steps):
            ep = str(ep)
            tag, raw = (ep.split(":", 1) + [""])[:2] if ":" in ep else ("", ep)
            feats = self._fut_feats.get(tag, {}).get(raw)
            if feats is None:
                outs.append(torch.zeros(9, 1152))
                continue
            T = feats.shape[0]
            n_seg = self._fut_nseg.get(ep, 3)
            bounds = self._future_boundaries(T, n_seg, self._fut_maxchunk)
            t = max(0, min(int(st), T - 1))
            w_next = next((b for b in bounds if b > t), T - 1)
            f = feats[w_next].float()
            outs.append(f.reshape(-1, 1152) if f.dim() > 1 else f.view(1, 1152).expand(9, 1152).clone()
                        if f.numel() == 1152 else f.reshape(-1, 1152))
        return torch.stack(outs).to(device)

    # ------------------------------------------------------------------
    # C-line: instruction-conditioned projection (TSSM_PROJ_XATTN=1)
    # ------------------------------------------------------------------
    def _ensure_instr_tok(self):
        if self._instr_tok is None:
            from transformers import AutoTokenizer
            self._instr_tok = AutoTokenizer.from_pretrained(
                os.environ.get("TSSM_LLM_TOKENIZER",
                               "/cluster/project/rsl/ruiqzhang/codes/NaVILA/navila_8f/llm"),
                use_fast=True)

    def _ensure_instr_map(self):
        """episode_id -> instruction token ids, extracted once from the SFT
        annotations (the q field embeds the instruction). Keys stored both
        prefixed (r2r:EP) and bare, matching whichever the cache uses."""
        if self._instr_map_loaded:
            return
        self._instr_map_loaded = True
        self._ensure_instr_tok()
        import json as _json
        import re as _re
        # annotations.json q = the RAW instruction text (templates are applied
        # later by dataset.py); regexes only as fallback for templated files.
        pats = (_re.compile(r'task is: "(.*?)" (?:Analyze|Compare|Decide)', _re.S),
                _re.compile(r'navigation instruction is: "(.*?)"\n', _re.S),
                _re.compile(r'task is: "(.*?)"', _re.S))
        maxlen = int(os.environ.get("TSSM_INSTR_MAXLEN", "288"))
        for tag, envk, default in (
            ("r2r", "TSSM_INSTR_R2R",
             "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json"),
            ("rxr", "TSSM_INSTR_RXR",
             "/cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/annotations.json"),
        ):
            p = os.environ.get(envk, default)
            if not os.path.exists(p):
                continue
            seen: dict = {}
            for s in _json.load(open(p)):
                ep = str(s["video_id"]).rsplit("-", 1)[0]
                if ep in seen:
                    continue
                q = s.get("q", "")
                if 'task is: "' in q or 'instruction is: "' in q:
                    for pat in pats:
                        m = pat.search(q)
                        if m:
                            seen[ep] = m.group(1)
                            break
                else:                      # q IS the raw instruction text
                    txt = q.replace("<image>", "").strip()
                    if txt:
                        seen[ep] = txt
            n_tok = 0
            for ep, txt in seen.items():
                ids = self._instr_tok(txt, add_special_tokens=False, truncation=True,
                                      max_length=maxlen)["input_ids"]
                t = torch.tensor(ids, dtype=torch.long)
                self._instr_ids[f"{tag}:{ep}"] = t
                self._instr_ids.setdefault(ep, t)
                n_tok += len(ids)
            print(f"[TSSMMemoryAdapter] instr map {tag}: {len(seen)} eps, "
                  f"avg {n_tok / max(1, len(seen)):.0f} tok", flush=True)
        if len(self._instr_ids) < 1000:
            print("[TSSMMemoryAdapter] *** WARNING: instruction map nearly EMPTY "
                  f"({len(self._instr_ids)} keys) — xattn will degrade to "
                  "state-only readout. Check TSSM_INSTR_R2R/RXR paths. ***", flush=True)

    def set_online_instruction(self, text: Optional[str]):
        """Eval hook (navila_trainer, per episode): the current instruction for
        val episodes that are absent from the training annotation map."""
        if text is None:
            self._online_instr_ids = None
            self._online_instr_text = None
            return
        if getattr(self, "_online_instr_text", None) == text:
            return
        self._online_instr_text = text
        self._ensure_instr_tok()
        ids = self._instr_tok(text, add_special_tokens=False, truncation=True,
                              max_length=int(os.environ.get("TSSM_INSTR_MAXLEN", "288")))["input_ids"]
        self._online_instr_ids = torch.tensor(ids, dtype=torch.long)

    def _project_xattn(self, states: torch.Tensor, episode_ids) -> torch.Tensor:
        """states [B, K, state_dim] -> tokens [B, K*N, llm_dim]."""
        device = states.device
        emb_ref = getattr(self, "_embed_ref", None)
        assert emb_ref is not None, \
            "TSSM_PROJ_XATTN=1 needs llava_arch to set adapter._embed_ref"
        emb = emb_ref[0]
        B, K, _ = states.shape
        dtype = self.x_out.weight.dtype
        seqs = []
        if episode_ids is None:            # online eval: per-step instruction via
            episode_ids = [None] * B       # set_online_instruction, no cache lookup
        for ep in episode_ids:
            ids = self._instr_ids.get(str(ep))
            if ids is None:
                if self._online_instr_ids is not None:
                    ids = self._online_instr_ids
                elif not self._instr_map_loaded:
                    self._ensure_instr_map()
                    ids = self._instr_ids.get(str(ep))
            seqs.append(ids if ids is not None else torch.zeros(0, dtype=torch.long))
        L = max(1, max(len(s) for s in seqs))
        pad = torch.zeros(B, L, dtype=torch.long, device=device)
        kpad = torch.ones(B, L, dtype=torch.bool, device=device)   # True = masked
        for i, s in enumerate(seqs):
            if len(s):
                pad[i, : len(s)] = s.to(device)
                kpad[i, : len(s)] = False
        with torch.no_grad():                       # embeddings frozen constants
            instr_e = emb(pad)
        instr_h = self.x_instr_in(instr_e.to(dtype))                 # [B, L, H]
        instr_h = instr_h + self.x_pe[:L].to(dtype).unsqueeze(0)     # word order
        if hasattr(self, "x_instr_ctx"):
            instr_h = self.x_instr_ctx(instr_h, src_key_padding_mask=kpad)
        st_h = self.x_state_in(states.to(dtype).reshape(B * K, self.state_dim))
        instr_h = instr_h.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, L, -1)
        kpad = kpad.unsqueeze(1).expand(-1, K, -1).reshape(B * K, L)
        # Slot 0 = STATIC sink (graceful degradation without the black-hole
        # kinship of putting st_h itself in KV — state reaches the output via
        # the query residual, never through attention content).
        sink = self.x_sink.to(dtype).expand(B * K, 1, -1)
        kv = torch.cat([sink, instr_h], dim=1)                       # [B*K, 1+L, H]
        kpad = torch.cat([torch.zeros(B * K, 1, dtype=torch.bool, device=device), kpad], dim=1)
        drop_p = float(os.environ.get("TSSM_XATTN_SINK_DROP", "0.0"))
        if self.training and drop_p > 0:
            # train-only sink dropout: force excursions into the text tokens.
            # Never drop the sink for rows with no instruction (all-masked
            # rows make softmax NaN).
            has_instr = (~kpad[:, 1:]).any(dim=1)
            drop = (torch.rand(B * K, device=device) < drop_p) & has_instr
            kpad[:, 0] = kpad[:, 0] | drop

        mode = getattr(self, "xattn_mode", "progress_only")
        outs = []
        # ---- branch 1: progress (state asks the instruction) -------------
        if mode in ("progress_only", "dual", "prog_direct"):
            q = self.x_query[: self.n_prog].unsqueeze(0).to(dtype) + st_h.unsqueeze(1)
            for attn, lnq, lnkv, ffn in zip(self.x_attn, self.x_ln_q, self.x_ln_kv, self.x_ffn):
                a, _ = attn(lnq(q), lnkv(kv), lnkv(kv),
                            key_padding_mask=kpad, need_weights=False)
                q = q + a
                q = q + ffn(q)
            outs.append(q)                                           # [B*K, n_p, H]
        # ---- branch 2: world (instruction asks the state) -----------------
        if mode in ("world_only", "dual", "prog_direct"):
            # state -> N_s slots: REAL K/V set (two tokenizer kinds)
            flat = self.x_stok_ln_in(states.to(dtype).reshape(B * K, self.state_dim))
            if getattr(self, "x_stok_kind", "chunk") == "factorized_global":
                # every slot sees the FULL state through a 256-d bottleneck
                slot_flat = self.x_stok_g2(self.x_stok_g1(flat))
                raw = slot_flat.view(B * K, self.n_state_slots, -1)
            else:
                chunks = flat.view(B * K, self.n_state_slots, -1)
                raw = self.x_stok_mlp(chunks)
            slots = self.x_stok_ln(raw + self.x_stok_emb.to(dtype))
            if mode == "prog_direct":
                # baseline D: no attention — direct state slots as tokens
                outs.append(slots[:, : self.n_world])
            else:
                # instruction-conditioned queries: base + pooled-instr modulation
                denom = (~kpad[:, 1:]).sum(dim=1, keepdim=True).clamp_min(1).to(dtype)
                summ = (instr_h * (~kpad[:, 1:]).unsqueeze(-1).to(dtype)).sum(1) / denom
                qw0 = self.x_wq_base[: self.n_world].unsqueeze(0).to(dtype) + \
                    self.x_wq_proj(summ).unsqueeze(1)
                qw = qw0
                for attn, lnq, lnkv, ffn in zip(self.x_wattn, self.x_wln_q,
                                                self.x_wln_kv, self.x_wffn):
                    a, _ = attn(lnq(qw), lnkv(slots), lnkv(slots), need_weights=False)
                    if os.environ.get("TSSM_XATTN_DEBUG", "0") == "1" and \
                            getattr(self, "_xattn_dbg_n", 0) % 100 == 0:
                        print(f"[xattn-dbg] world attn_delta/carrier = "
                              f"{(a.norm() / qw.norm().clamp_min(1e-6)).item():.4f}",
                              flush=True)
                    qw = qw + a
                    qw = qw + ffn(qw)
                g = self.x_wgain if not torch.is_tensor(self.x_wgain) \
                    else self.x_wgain.to(dtype)
                if (torch.is_tensor(g) or g != 1.0):
                    qw = qw0 + g * (qw - qw0)     # carrier-dominance antidote
                # world-specific aux supervision: historical landmark recall
                if (self.world_aux == "landmark_recall" and self.training):
                    tgt = torch.zeros(B, self._lm_V, device=device, dtype=dtype)
                    hit = torch.zeros(B, dtype=torch.bool)
                    for i, (ep, st) in enumerate(zip(episode_ids, self._aux_steps or [0] * B)):
                        fs = self._lm_first_seen.get(str(ep))
                        if fs:
                            hit[i] = True
                            for j, t0 in fs.items():
                                if t0 <= int(st):
                                    tgt[i, j] = 1.0
                    if hit.any():
                        pooled = qw.view(B, K, -1, qw.shape[-1])[:, 0].mean(1)
                        logits = self.x_world_aux_head(pooled)
                        self._xattn_aux_loss = self.world_aux_w * F.binary_cross_entropy_with_logits(
                            logits[hit], tgt[hit].float())
                outs.append(qw)                                      # [B*K, n_w, H]
        # ---- branch dropout (train-only, 防支路塌缩的第一线) ---------------
        if self.training and len(outs) == 2:
            pd_ = float(os.environ.get("TSSM_PROGRESS_BRANCH_DROPOUT", "0"))
            wd_ = float(os.environ.get("TSSM_WORLD_BRANCH_DROPOUT", "0"))
            if pd_ > 0 or wd_ > 0:
                mp = (torch.rand(B * K, 1, 1, device=device) < pd_)
                mw = (torch.rand(B * K, 1, 1, device=device) < wd_)
                both = mp.squeeze() & mw.squeeze()
                mp[both] = False                       # never drop both
                outs[0] = outs[0].masked_fill(mp, 0.0)
                outs[1] = outs[1].masked_fill(mw, 0.0)
        # ---- branch-specific out-norm (可选) -------------------------------
        if hasattr(self, "x_prog_onorm") and len(outs) == 2:
            outs[0] = self.x_prog_onorm(outs[0])
            outs[1] = self.x_world_onorm(outs[1])
        self._xattn_dbg_n = getattr(self, "_xattn_dbg_n", 0) + 1
        # ---- fusion -------------------------------------------------------
        if len(outs) == 2 and getattr(self, "x_fusion", "concat") == "residual":
            h = outs[0] + self.x_fuse_alpha.to(dtype) * outs[1]      # needs n_p == n_w
        else:
            h = torch.cat(outs, dim=1)                               # [B*K, N_total, H]
        tokens = self.x_out(h)                                       # shared LLM-space head
        assert tokens.shape[1] == self.num_state_tokens, \
            f"budget mismatch: {tokens.shape[1]} != {self.num_state_tokens}"
        # ---- SigLIP gist salary, mode-agnostic (07-20) --------------------
        # Narration cannot express "what does this state SEE" (42.6% of RxR
        # 5-step B-course labels differ only by the progress %). The gist of the
        # frame ACTUALLY recorded at the supervised step is that missing channel,
        # and it is literally what the deploy prompt asks ("you will see:").
        # For B-course rows the target is the frame at t+H, not t.
        if (self.world_aux == "siglip_gist" and self.training
                and getattr(self, "_gist_targets", None) is not None):
            tgt = torch.zeros(B, 9, 1152, device=device, dtype=torch.float32)
            hit = torch.zeros(B, dtype=torch.bool)
            flags = getattr(self, "_imagined_flags", None)
            for i, (ep, st) in enumerate(zip(episode_ids or [None] * B,
                                             self._aux_steps or [0] * B)):
                row = self._gist_targets.get(str(ep))
                if row is None:
                    continue
                _hz = int(flags[i] or 0) if (flags and i < len(flags)) else 0
                t0 = int(st) + _hz          # B课: 靶子取冻结点那帧(变长)
                t0 = min(max(t0, 0), row.shape[0] - 1)
                r0 = row[t0]
                tgt[i] = (r0 if r0.dim() == 2 else r0.view(9, 1152)).to(device).float()
                hit[i] = True
            if hit.any():
                pooled = tokens.view(B, K, -1, tokens.shape[-1])[:, 0].mean(1)
                pred = self.x_world_aux_head(pooled).float().view(B, 9, 1152)
                self._xattn_aux_loss = self.world_aux_w * (
                    1 - F.cosine_similarity(pred[hit], tgt[hit], dim=2).mean())
        return tokens.reshape(B, K * self.num_state_tokens, self.llm_dim)

    def _lmk_slot_token(self, state_vec, k, dt_steps, device, wdt):
        """单槽: state(3072) → 1 token(llm_dim). state_vec 已在 device。"""
        h = torch.nn.functional.silu(self.lmk_l1(self.lmk_ln(state_vec)))
        h = torch.nn.functional.silu(self.lmk_l2(h))
        tok = self.lmk_out(h)                                        # [llm_dim]
        ph = torch.tensor([[dt_steps / self.lmk_phase_norm]], device=device, dtype=wdt)
        return tok + self.lmk_slot_emb[k].to(wdt) + self.lmk_phase(ph).squeeze(0)

    def lmk_v2_reset(self):
        self._lmk_ob = []

    def lmk_v2_push(self, state_vec, step, noun):
        """eval 每步: navila_trainer 匹配器触发时调用, 存 (state, step, noun)。
        ⚠ 在线 state 是完整 3072; h-only 臂(state_dim=2048)的槽位层按 2048 建,
        必须按 [state_start:state_start+state_dim] 切片后入库 —— 训练走缓存时
        _lmk_v2_forward 里有同款切片, 在线路径 07-27 前漏切导致 LayerNorm 3072≠2048 崩。"""
        sv = state_vec.detach().float().cpu().flatten()
        sv = sv[self.state_start:self.state_start + self.state_dim]
        self._lmk_ob.append((sv, int(step), noun))

    def _lmk_v2_forward(self, B, episode_ids, steps, device, dtype):
        """返回 ([B, L, llm_dim] 银行token, aux|None)。
        训练: ledger[ep] 的 (名词,触发步)≤t → cache 取 state → 投影;
        评测: episode_ids is None → 用 _lmk_ob(在线银行, B==1)。
        词嵌入锚: 每实槽 cos(W·tok, 冻结靶[名词]); 无靶批连图零(NCCL家法)。"""
        L = self.lmk_L
        wdt = self.lmk_l1.weight.dtype
        if episode_ids is None:                                      # ---- eval ----
            now = self._lmk_ob[-1][1] if self._lmk_ob else 0
            per = [[(s.to(device=device, dtype=wdt), now - st, nn)
                    for (s, st, nn) in self._lmk_ob[-L:]]]
        else:                                                       # ---- train ----
            if self._lmk_ledger is None:
                import glob
                led = {}
                for p in sorted(glob.glob(os.environ.get(
                        "TSSM_LMK_LEDGER_GLOB",
                        "/cluster/scratch/ruiqzhang/lmk_ledger_*.pt"))):
                    led.update(torch.load(p, map_location="cpu", weights_only=False))
                self._lmk_ledger = led
                self._lmk_anchor = torch.load(os.environ.get(
                    "TSSM_LMK_ANCHOR", "/cluster/scratch/ruiqzhang/lmk_anchor_targets.pt"),
                    map_location="cpu", weights_only=True)
                print(f"[lmk-v2] ledger {len(self._lmk_ledger)} eps, "
                      f"anchor {len(self._lmk_anchor)} nouns", flush=True)
            sts = [int(s) for s in (steps if steps is not None else [0] * B)]
            per = []
            for ep, t in zip(list(episode_ids), sts):
                led = self._lmk_ledger.get(str(ep))
                row = self.cache.get(str(ep))
                if led is None or row is None:
                    per.append([]); continue
                items = []
                for ni, st in led["fires"]:
                    if st <= t and st < row.shape[0]:
                        sv = row[st, self.state_start:self.state_start + self.state_dim]
                        items.append((sv.to(device=device, dtype=wdt), t - st,
                                      led["nouns"][ni]))
                per.append(items[:L])

        rows, anchor_pred, anchor_tgt = [], [], []
        for items in per:
            base = self.lmk_null.to(dtype).unsqueeze(0).expand(L, -1).clone()
            for k, (sv, dt, noun) in enumerate(items[:L]):
                tok = self._lmk_slot_token(sv, k, dt, device, wdt)
                base[k] = tok.to(dtype)
                if self.training and self._lmk_anchor is not None and noun in self._lmk_anchor:
                    anchor_pred.append(self.x_aux_lmkw(tok).float())
                    anchor_tgt.append(self._lmk_anchor[noun].to(device).float())
            rows.append(base)
        out = torch.stack(rows, 0)                                   # [B, L, llm_dim]

        aux = None
        if self.training and episode_ids is not None:
            if anchor_pred:
                P = torch.nn.functional.normalize(torch.stack(anchor_pred), dim=-1)
                T = torch.nn.functional.normalize(torch.stack(anchor_tgt), dim=-1)
                aux = self.lmk_aux_w * (1.0 - (P * T).sum(-1)).mean()
                aux = aux + out.sum() * 0.0                          # 连图保险
            else:
                aux = out.sum() * 0.0 + sum(p.sum() for p in self.x_aux_lmkw.parameters()) * 0.0
            if self._lmk_dbg < 3:
                self._lmk_dbg += 1
                nn_hit = sum(len(x) for x in per)
                print(f"[lmk-v2] hit slots={nn_hit}/{B} anchored={len(anchor_pred)} "
                      f"aux={float(aux):.4f}", flush=True)
        return out, aux

    def _lmk_v3_load(self):
        """lazy 加载 v3 资产(首个训练批): 账本(槽位表)+锚靶+动作序列。
        fail-fast: 缺谁当场炸, 别让臂空跑5000步才发现(07-18 五甲板静默空跑教训)。"""
        # 多账本: 冒号分隔 (r2r:rxr 两库对齐监督 — 07-23 用户家法)
        self._lmk_v3_ledger = {}
        led_blob = {}
        for _p in os.environ["TSSM_LMK_LEDGER_V3"].split(":"):
            if not _p:
                continue
            led_blob = torch.load(_p, map_location="cpu", weights_only=False)
            self._lmk_v3_ledger.update(led_blob["ledger"])
        self._lmk_anchor = torch.load(os.environ.get(
            "TSSM_LMK_ANCHOR", "/cluster/scratch/ruiqzhang/lmk_anchor_targets.pt"),
            map_location="cpu", weights_only=True)
        self._lmk_acts = torch.load(os.environ.get(
            "TSSM_ACTION_SEQS", "/cluster/scratch/ruiqzhang/tssm_action_seqs.pt"),
            weights_only=True)
        self._lmk_act_key = {k.split(":")[-1]: k for k in self._lmk_acts}
        assert len(self._lmk_v3_ledger) > 5000, "账本v3疑似残缺"
        assert len(self._lmk_anchor) > 500, "锚靶表疑似残缺"
        print(f"[lmk-v3] ledger {len(self._lmk_v3_ledger)} eps "
              f"(tau={led_blob.get('tau')} gap={led_blob.get('gap')}), "
              f"anchor {len(self._lmk_anchor)} nouns", flush=True)

    def _lmk_v3_forward(self, B, episode_ids, steps, device, dtype):
        """v3: 槽位表驱动 + 批量向量化投影 + 词锚/动作双头.
        返回 ([B, L, llm_dim], aux|None)。eval(episode_ids None)复用 _lmk_ob。
        与 v2 的关键差异: ① 因果律由账本承担(slots[t]即银行内容, 零运行时过滤);
        ② 全批实槽堆成一个 [N,3072] 一次过投影器(v2 是 B×L 次小GEMM);
        ③ 空页也是实槽(动作双头有标签), 词锚仅有名页。"""
        L = self.lmk_L
        wdt = self.lmk_l1.weight.dtype
        flat_sv, meta = [], []          # meta: (b, k, dt, noun|None, prev_a, curr_a)
        if episode_ids is None:                                       # ---- eval ----
            ob = self._lmk_ob[-L:]
            now = ob[-1][1] if ob else 0
            for k, (sv, st, _noun) in enumerate(ob):
                flat_sv.append(sv.to(device=device, dtype=wdt))
                meta.append((0, k, now - st, None, -1, -1))
            B = 1
        else:                                                         # ---- train ----
            if self._lmk_v3_ledger is None:
                self._lmk_v3_load()
            sts = [int(s) for s in (steps if steps is not None else [0] * B)]
            for b, (ep, t) in enumerate(zip(list(episode_ids), sts)):
                rec = self._lmk_v3_ledger.get(str(ep))
                row = self.cache.get(str(ep))
                # ep 可能带前缀(r2r:123, 与acts键同格式)也可能裸号 — 两路都试
                ak = str(ep) if str(ep) in self._lmk_acts else \
                    self._lmk_act_key.get(str(ep).split(":")[-1])
                if rec is None or row is None:
                    continue
                t_idx = min(t, rec["slots"].shape[0] - 1)
                acts = self._lmk_acts.get(ak) if ak else None
                for k, ws in enumerate(rec["slots"][t_idx].tolist()):
                    if ws < 0 or ws >= row.shape[0]:
                        continue                                     # 空槽
                    sv = row[ws, self.state_start:self.state_start + self.state_dim]
                    flat_sv.append(sv.to(device=device, dtype=wdt))
                    pa = int(acts[ws - 1]) if acts is not None and 0 < ws < len(acts) else -1
                    ca = int(acts[ws]) if acts is not None and ws < len(acts) else -1
                    meta.append((b, k, t - ws, rec["anchor"].get(ws), pa, ca))

        out = self.lmk_null.to(dtype).view(1, 1, -1).expand(B, L, -1).clone()
        aux = None
        if flat_sv:
            S = torch.stack(flat_sv)                                  # [N, 3072] 一次投影
            h = torch.nn.functional.silu(self.lmk_l1(self.lmk_ln(S)))
            toks = self.lmk_out(torch.nn.functional.silu(self.lmk_l2(h)))   # [N, llm]
            ks = torch.tensor([m[1] for m in meta], device=device)
            dts = torch.tensor([[m[2] / self.lmk_phase_norm] for m in meta],
                               device=device, dtype=wdt)
            toks = toks + self.lmk_slot_emb[ks].to(wdt) + self.lmk_phase(dts)
            for i, (b, k, *_rest) in enumerate(meta):
                out[b, k] = toks[i].to(dtype)

        if self.training and episode_ids is not None:
            # 哑元前向(07-23 二次NCCL验尸: 桶序分歧 REDUCE 32768 vs 86036):
            # 参数和×0只保证"有梯度", 不保证"梯度就绪顺序" — 空批rank的梯度在
            # 反向早段就绪, 有页rank经LLM主路径晚段就绪, ZeRO桶序错位死锁。
            # 根治: 每rank每步跑一次全零state的完整槽管线, 且结果广播加零进out
            # (走LLM主路径, 与真实页同位), 双头也各摸一次 → 计算图逐rank同构。
            _dS = torch.zeros(1, self.state_dim, device=device, dtype=wdt)
            _dh = torch.nn.functional.silu(self.lmk_l1(self.lmk_ln(_dS)))
            _dtok = self.lmk_out(torch.nn.functional.silu(self.lmk_l2(_dh)))
            _dtok = _dtok + self.lmk_slot_emb[0:1].to(wdt)                           + self.lmk_phase(torch.zeros(1, 1, device=device, dtype=wdt))
            out = out + (_dtok * 0.0).view(1, 1, -1).to(dtype)   # 主路径同位连图
            zero = self.x_aux_lmkw(_dtok).sum() * 0.0 + self.x_aux_lmka(_dtok).sum() * 0.0
            aux = zero
            _word_raw = _actp_raw = _actc_raw = None
            if flat_sv:
                # 词锚: 仅有名页
                idx = [i for i, m in enumerate(meta) if m[3]]
                if idx:
                    P = torch.nn.functional.normalize(
                        self.x_aux_lmkw(toks[idx]).float(), dim=-1)
                    Tg = torch.nn.functional.normalize(torch.stack(
                        [self._lmk_anchor[meta[i][3]] for i in idx]).to(device).float(), dim=-1)
                    _word_raw = (1.0 - (P * Tg).sum(-1)).mean()
                    aux = aux + self.lmk_aux_w * _word_raw
                # 动作双头: 全部有标签的槽 (prev/curr 各 10 类)
                aidx = [i for i, m in enumerate(meta) if m[4] >= 0 and m[5] >= 0]
                if aidx:
                    logits = self.x_aux_lmka(toks[aidx]).float()
                    lp, lc = logits.chunk(2, dim=-1)
                    tp = torch.tensor([meta[i][4] for i in aidx], device=device)
                    tc = torch.tensor([meta[i][5] for i in aidx], device=device)
                    _actp_raw = torch.nn.functional.cross_entropy(lp, tp)
                    _actc_raw = torch.nn.functional.cross_entropy(lc, tc)
                    aux = aux + self.lmk_act_w * 0.5 * (_actp_raw + _actc_raw)
            self._lmk_dbg += 1
            # 头3发sanity + 之后每200次前向周期打点(07-24: trainer段合并未生效, 分量直打日志)
            if self._lmk_dbg <= 3 or self._lmk_dbg % 200 == 0:
                nnn = sum(1 for m in meta if m[3])
                _w = f"{float(_word_raw):.4f}" if _word_raw is not None else "-1"
                _ap = f"{float(_actp_raw):.4f}" if _actp_raw is not None else "-1"
                _ac = f"{float(_actc_raw):.4f}" if _actc_raw is not None else "-1"
                print(f"[lmk-v3] n={self._lmk_dbg} slots={len(meta)}/{B*L} named={nnn} "
                      f"aux={float(aux):.4f} word={_w} actp={_ap} actc={_ac}", flush=True)
            # 分量日志(07-23 用户要求): raw各项 + 加权合计, trainer logs 段合并输出。
            # 只存 float(不进图), 供人判断 aux 相对 nav CE 的真实占比。
            self._lmk_log = {
                "lmk_word_raw": float(_word_raw) if _word_raw is not None else -1.0,
                "lmk_actp_raw": float(_actp_raw) if _actp_raw is not None else -1.0,
                "lmk_actc_raw": float(_actc_raw) if _actc_raw is not None else -1.0,
                "lmk_aux_wsum": float(aux),
            }
        return out, aux

    def _lmk_forward(self, B, episode_ids, steps, device, dtype):
        """LMK-Bank: 返回 ([B, L, llm_dim] 银行token, aux_loss|None).

        训练: (ep,step) 查离线槽位表+pooled缓存重建当步银行;
        在线: episode_ids is None → 用 OnlineBank 当前内容 (B==1)。
        空槽用 lmk_null 占位, token 数恒为 L (batch 无痛堆叠)。
        aux: mean(实槽token) → 512 multi-hot BCE(first_seen_v3, pos_weight=20);
        无标签批返回连图零 (07-22 NCCL 死锁家法)。
        """
        L = self.lmk_L
        wdt = self.lmk_proj.weight.dtype
        rows, aux_feats, aux_lbl = [], [], []
        if episode_ids is None:
            feats, wsteps = self._lmk_online.tensors()
            per = [(feats, wsteps)] * B          # eval B==1
        else:
            if self._lmk_pooled is None:
                self._lmk_pooled = torch.load(os.environ.get(
                    "TSSM_LMK_POOLED", "/cluster/scratch/ruiqzhang/lmk_pooled_r2r_rxr.pt"),
                    map_location="cpu", weights_only=True)
                self._lmk_slots = torch.load(os.environ.get(
                    "TSSM_LMK_SLOTS", "/cluster/scratch/ruiqzhang/lmk_slots_r2r_rxr.pt"),
                    map_location="cpu", weights_only=True)
                print(f"[lmk] pooled/slots loaded: {len(self._lmk_pooled)} eps", flush=True)
            per = []
            sts = [int(s) for s in (steps if steps is not None else [0] * B)]
            for ep, st in zip(list(episode_ids), sts):
                key = str(ep)
                pl, sl = self._lmk_pooled.get(key), self._lmk_slots.get(key)
                if pl is None or sl is None:
                    per.append((None, []))
                    continue
                st = min(st, sl.shape[0] - 1)
                ws = [int(x) for x in sl[st] if int(x) >= 0]
                per.append((pl[ws].float() if ws else None, ws))
        for feats, wsteps in per:
            base = self.lmk_null.to(dtype).unsqueeze(0).expand(L, -1).clone()
            if feats is not None and len(wsteps) > 0:
                n = min(len(wsteps), L)
                f = feats[:n].to(device=device, dtype=wdt)
                ph = (torch.tensor(wsteps[:n], device=device, dtype=wdt)
                      .clamp(0, 96).unsqueeze(-1) / 64.0)
                tok = (self.lmk_proj(self.lmk_ln(f)) + self.lmk_slot_emb[:n].to(wdt)
                       + self.lmk_phase(ph))
                base[:n] = tok.to(dtype)
                aux_feats.append(tok.mean(0))
            else:
                aux_feats.append(None)
            rows.append(base)
        out = torch.stack(rows, 0)                                   # [B, L, llm_dim]

        aux = None
        if self.training and episode_ids is not None:
            if self._lmk_labels is None:
                blob = torch.load(os.environ.get(
                    "TSSM_LMK_LABELS", "/cluster/scratch/ruiqzhang/tssm_landmark_labels_v3.pt"),
                    map_location="cpu", weights_only=True)
                self._lmk_labels = blob["first_seen"]
                print(f"[lmk] labels loaded: {len(self._lmk_labels)} eps", flush=True)
            logit_rows, ys = [], []
            for i, (ep, st) in enumerate(zip(list(episode_ids), sts)):
                fs = self._lmk_labels.get(str(ep))
                if fs is None or aux_feats[i] is None:
                    continue
                y = torch.zeros(512)
                for j, s0 in fs.items():
                    if int(s0) <= st:
                        y[int(j)] = 1.0
                logit_rows.append(self.x_aux_lmk(aux_feats[i]).float())
                ys.append(y)
            if logit_rows:
                lg = torch.stack(logit_rows)
                yy = torch.stack(ys).to(lg.device)
                aux = self.lmk_aux_w * F.binary_cross_entropy_with_logits(
                    lg, yy, pos_weight=torch.full((512,), 20.0, device=lg.device))
                aux = aux + out.sum() * 0.0                          # 连图保险
            else:
                aux = out.sum() * 0.0 + sum(p.sum() for p in self.x_aux_lmk.parameters()) * 0.0
        return out, aux

    def _nav3l_aux(self, tokens, hmid, episode_ids, steps):
        """NAV-3L v1 辅助损失: temporal(挂h,归一步号) + action(挂mean-pool
        token,最近1/2步动作,加权a2)。原始权重经 rssm_loss ×0.1 → 有效
        temporal 0.05 / a1 0.05 / a2 0.02。仅训练期。"""
        if not self.training or episode_ids is None:
            return None
        import torch.nn.functional as _F
        device = tokens.device
        B = tokens.shape[0]
        N = self.num_state_tokens
        eps = list(episode_ids)
        sts = [int(s) for s in (steps if steps is not None else [0] * B)]
        # 动作序列缓存 (与 act_seq 头同源, 懒加载)
        if getattr(self, "_act_seqs", None) is None:
            ap = os.environ.get("TSSM_ACTION_SEQS",
                                "/cluster/scratch/ruiqzhang/tssm_action_seqs.pt")
            self._act_seqs = torch.load(ap, map_location="cpu", weights_only=True)
        # --- temporal: 归一 elapsed-step 回归 (纯时钟, 不除T) ---
        step_t = torch.tensor([s / self.n3_step_norm for s in sts],
                              device=device, dtype=torch.float32).clamp(0, 1.5)
        h_cur = hmid.view(B, -1, self.n3_hid)[:, -1]     # 末槽=最新记忆 (hist8修正; K=1时与旧口径等价)
        l_t = _F.mse_loss(self.x_aux_temporal(h_cur).float().squeeze(-1), step_t)
        # --- action: mean-pool 整组当前记忆 token → 最近两步 ---
        pooled = tokens.view(B, -1, N, tokens.shape[-1])[:, -1].mean(1)    # 末槽=最新记忆 (hist8修正)
        tgt = torch.full((B, 2), -100, dtype=torch.long, device=device)
        for i, (ep, st) in enumerate(zip(eps, sts)):
            seq = self._act_seqs.get(str(ep))
            if seq is None:
                continue
            for k in (1, 2):
                if st - k >= 0 and st - k < len(seq):
                    a = int(seq[st - k])
                    if 1 <= a <= 9:
                        tgt[i, k - 1] = a - 1
        logit_a = self.x_aux_action(pooled).float().view(B, 2, 9)

        def _ce(lg, t):
            # 全 mask 时返回"连着计算图的零"(lg.sum()*0), 而非断开的 new_zeros:
            # 否则该 rank 的 x_aux_action 不参与 all-reduce -> NCCL desync 死锁
            # (07-22 血案: PER_DEVICE_BS=1 下某rank抽到episode首步即触发)。
            return (_F.cross_entropy(lg, t, ignore_index=-100)
                    if (t != -100).any() else lg.sum() * 0.0)
        l_a = _ce(logit_a[:, 0], tgt[:, 0]) + self.n3_a2w * _ce(logit_a[:, 1], tgt[:, 1])
        # NCCL同步保险 (07-22事故): 全mask批(st=0样本)若返回断图零, 该rank的
        # 动作头无梯度不进all-reduce, 与对端归约序列错位 -> REDUCE死锁10min自杀
        # (实锤: 卡住的桶=1.32M≈双辅助头参数量)。恒加连图零项, 保证每个
        # micro-batch双头都参与归约, 值恒为0不改损失。
        l_a = l_a + logit_a.sum() * 0.0
        # 键前缀哨兵 (07-14家法)
        if self._n3_dbg < 3:
            print(f"[nav3l] hit a1={int((tgt[:,0]>=0).sum())}/{B} "
                  f"a2={int((tgt[:,1]>=0).sum())}/{B} gate_mean={self._n3_gate_mean:.3f} "
                  f"l_t={float(l_t):.3f} l_a={float(l_a):.3f}", flush=True)
            self._n3_dbg += 1
        # 原始权重: temporal 0.5, action 0.5 → ×0.1 → 有效 0.05/0.05(a2内0.4→0.02)
        return 0.5 * l_t + 0.5 * l_a

    def _aux_heads_loss(self, tokens, episode_ids, steps):
        """四头损失 (07-20): 三辅助头读最终token(LLM看到什么就监督什么)。
        三种投影模式(MLP/slice/xattn)通用; 仅训练期, 头部署时剥离。
        gist: 冻结点那帧的SigLIP p9 (B课取 t+rolled); landmark: 截至冻结点
        已见地标多标签BCE; act_hist: 最近W步GT动作三族直方图(软标签CE, 永远
        用真实历史 —— 想象续步不改写"记忆里做过什么")。"""
        device = tokens.device
        B = tokens.shape[0]
        N = self.num_state_tokens
        # tokens [B, K*N, llm_dim] -> 只取第一组(当前记忆)的 N 个 token 平均
        pooled = tokens.view(B, -1, N, tokens.shape[-1])[:, 0].mean(1)
        flags = getattr(self, "_imagined_flags", None)
        eps = list(episode_ids) if episode_ids is not None else [None] * B
        sts = list(steps) if steps is not None else [0] * B
        hzs = [int(flags[i] or 0) if (flags is not None and i < len(flags)) else 0
               for i in range(B)]
        aux_total = None
        _dbg_hits = {}
        if "gist" in self.aux_heads:
            tgt = torch.zeros(B, 9, 1152, device=device, dtype=torch.float32)
            hit = torch.zeros(B, dtype=torch.bool)
            for i, (ep, st) in enumerate(zip(eps, sts)):
                row = self._gist_targets.get(str(ep))
                if row is None:
                    continue
                t0 = min(max(int(st) + hzs[i], 0), row.shape[0] - 1)
                r0 = row[t0]
                tgt[i] = (r0 if r0.dim() == 2 else r0.view(9, 1152)).to(device).float()
                hit[i] = True
            _dbg_hits["gist"] = int(hit.sum())
            if hit.any():
                pred = self.x_aux_gist_head(pooled).float().view(B, 9, 1152)
                l = self.aux_w_gist * (
                    1 - F.cosine_similarity(pred[hit], tgt[hit], dim=2).mean())
                aux_total = l if aux_total is None else aux_total + l
        if "gist_next" in self.aux_heads:
            tgt = torch.zeros(B, 9, 1152, device=device, dtype=torch.float32)
            hit = torch.zeros(B, dtype=torch.bool)
            for i, (ep, st) in enumerate(zip(eps, sts)):
                row = self._gist_targets.get(str(ep))
                if row is None:
                    continue
                t1 = int(st) + hzs[i] + 1
                if t1 > row.shape[0] - 1:      # 终点无"下一帧", 掩掉
                    continue
                r1 = row[t1]
                tgt[i] = (r1 if r1.dim() == 2 else r1.view(9, 1152)).to(device).float()
                hit[i] = True
            _dbg_hits["gist_next"] = int(hit.sum())
            if hit.any():
                pred = self.x_aux_gistnext_head(pooled).float().view(B, 9, 1152)
                l = self.aux_w_gistnext * (
                    1 - F.cosine_similarity(pred[hit], tgt[hit], dim=2).mean())
                aux_total = l if aux_total is None else aux_total + l
        if "landmark" in self.aux_heads:
            tgt = torch.zeros(B, self._aux_lm_V, device=device, dtype=torch.float32)
            hit = torch.zeros(B, dtype=torch.bool)
            for i, (ep, st) in enumerate(zip(eps, sts)):
                fs = self._aux_lm_first_seen.get(str(ep))
                if fs:
                    hit[i] = True
                    for j, t0 in fs.items():
                        if int(t0) <= int(st) + hzs[i]:
                            tgt[i, int(j)] = 1.0
            _dbg_hits["landmark"] = int(hit.sum())
            if hit.any():
                logits = self.x_aux_lm_head(pooled).float()
                l = self.aux_w_lm * F.binary_cross_entropy_with_logits(
                    logits[hit], tgt[hit])
                aux_total = l if aux_total is None else aux_total + l
        if "act_hist" in self.aux_heads:
            tgt = torch.zeros(B, 3, device=device, dtype=torch.float32)
            hit = torch.zeros(B, dtype=torch.bool)
            for i, (ep, st) in enumerate(zip(eps, sts)):
                seq = self._act_seqs.get(str(ep))
                if seq is None or int(st) < 1:
                    continue
                win = seq[max(0, int(st) - self.aux_act_win):int(st)]
                fam = torch.zeros(3)
                for a in win.tolist():
                    if 1 <= a <= 3:
                        fam[0] += 1
                    elif 4 <= a <= 6:
                        fam[1] += 1
                    elif 7 <= a <= 9:
                        fam[2] += 1
                if fam.sum() > 0:
                    tgt[i] = (fam / fam.sum()).to(device)
                    hit[i] = True
            _dbg_hits["act_hist"] = int(hit.sum())
            if hit.any():
                logp = F.log_softmax(self.x_aux_act_head(pooled).float(), dim=-1)
                l = self.aux_w_act * (-(tgt[hit] * logp[hit]).sum(-1).mean())
                aux_total = l if aux_total is None else aux_total + l
        if "act_seq" in self.aux_heads:
            L = self.aux_actseq_len
            tgt = torch.full((B, L), -100, dtype=torch.long, device=device)
            for i, (ep, st) in enumerate(zip(eps, sts)):
                seq = self._act_seqs.get(str(ep))
                if seq is None or int(st) < 1:
                    continue
                win = seq[max(0, int(st) - L):int(st)].tolist()
                # 右对齐: 位置L-1=最近一步(t-1), 前面不足处保持-100(掩掉)
                for k, a in enumerate(reversed(win)):
                    if 1 <= int(a) <= 9:
                        tgt[i, L - 1 - k] = int(a) - 1
            valid = tgt.ge(0)
            _dbg_hits["act_seq"] = int(valid.any(dim=1).sum())
            if valid.any():
                logits = self.x_aux_actseq_head(pooled).float().view(B, L, 9)
                l = self.aux_w_actseq * F.cross_entropy(
                    logits.view(B * L, 9), tgt.view(B * L), ignore_index=-100)
                aux_total = l if aux_total is None else aux_total + l
        if getattr(self, "_aux_dbg_n", 0) < 3:
            self._aux_dbg_n = getattr(self, "_aux_dbg_n", 0) + 1
            print(f"[aux-heads-dbg] batch{self._aux_dbg_n} B={B} hits={_dbg_hits} "
                  f"(某头恒0=键错, 立刻停)", flush=True)
        return aux_total

    def align_targets(self, device):
        """次态对齐目标: 冻结投影器翻译 s_{t+1} -> [B,N,llm_dim] + 有效mask。
        仅 MLP 投影模式; 终点步/缓存缺失行 mask 掉。"""
        assert self.proj is not None, "align aux 目前只支持 MLP 投影模式"
        eps, sts = getattr(self, "_align_keys", ([], []))
        B = len(eps)
        N = self.num_state_tokens
        tgt = torch.zeros(B, N, self.llm_dim, device=device, dtype=torch.float32)
        ok = torch.zeros(B, dtype=torch.bool)
        with torch.no_grad():
            for i, (ep, st) in enumerate(zip(eps, sts)):
                row = self.cache.get(str(ep))
                if row is None or int(st) + 1 > row.shape[0] - 1:
                    continue
                sv = row[int(st) + 1, self.state_start:self.state_start + self.state_dim]
                sv = sv.to(device=device, dtype=self.proj[1].weight.dtype).unsqueeze(0)
                tgt[i] = self.proj(sv).view(N, self.llm_dim).float()
                ok[i] = True
        return tgt, ok

    def load_stage1_projector(self, path: str) -> None:
        """Initialize the state-to-token MLP from Stage1 contrastive pretraining."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stage1 TSSM projector checkpoint not found: {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if getattr(self, "proj_xattn", False):
            sd = blob.get("proj", blob)
            xsd = {k: v for k, v in sd.items() if k.startswith("x_")}
            own = {k for k in self.state_dict() if k.startswith("x_")}
            # 四头(x_aux_*)是训练期专用读出头; 部署配置未开TSSM_AUX_HEADS时
            # 自动剥离, 不算模式错配。
            dropped_aux = sorted(k for k in (set(xsd) - own)
                                 if k.startswith("x_aux_"))
            if dropped_aux:
                xsd = {k: v for k, v in xsd.items() if k not in dropped_aux}
                print(f"[TSSMMemoryAdapter] dropped {len(dropped_aux)} "
                      f"training-only aux-head tensors (x_aux_*)", flush=True)
            unexpected = sorted(set(xsd) - own)
            if unexpected:
                raise ValueError(f"xattn projector load: ckpt has keys this "
                                 f"config lacks: {unexpected[:4]} (mode mismatch?)")
            missing = sorted(own - set(xsd))
            self.load_state_dict(xsd, strict=False)
            print(f"[TSSMMemoryAdapter] loaded Stage1 XATTN projector: {path} "
                  f"({len(xsd)} tensors; new-module keys left at init: "
                  f"{len(missing)})", flush=True)
            return
        sd = blob.get("proj", blob.get("adapter_proj", blob))
        if any(k.startswith("proj.") for k in sd.keys()):
            sd = {k[len("proj."):]: v for k, v in sd.items() if k.startswith("proj.")}
        expected = self.proj.state_dict()
        bad = []
        for k, v in sd.items():
            if k not in expected or tuple(v.shape) != tuple(expected[k].shape):
                bad.append((k, tuple(v.shape), tuple(expected[k].shape) if k in expected else None))
        if bad:
            raise ValueError(f"Stage1 projector shape mismatch for {path}: {bad[:4]}")
        missing, unexpected = self.proj.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"Could not load Stage1 projector cleanly from {path}: "
                f"missing={missing}, unexpected={unexpected}")
        print(
            f"[TSSMMemoryAdapter] loaded Stage1 projector: {path} "
            f"(state_dim={blob.get('state_dim', '?')}, slice={blob.get('slice_range', '?')})"
        )

    # ------------------------------------------------------------------
    # cache management
    # ------------------------------------------------------------------

    def load_cache(self, cache_path: str) -> None:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"TSSM state cache not found: {cache_path}\n"
                f"Run RSSM_v4/cache_tssm_states_v4.py first."
            )
        print(f"[TSSMMemoryAdapter] loading state cache: {cache_path}")
        d = torch.load(cache_path, map_location="cpu", weights_only=True)
        assert isinstance(d, dict), f"{cache_path} is not a dict-of-tensors"
        self.cache = d
        any_v = next(iter(d.values()))
        self.cache_state_dim = any_v.shape[-1]
        # HARD GUARD against silent truncation. The TSSM full state is
        # [h(2048), z(1024)] = 3072 under concat_o; using state_dim=1536 would
        # silently drop h_layer4 AND all of z (the observation latent). Any
        # sub-window (state_start>0 or state_dim<full) must be opted into.
        uses_subwindow = (self.state_start != 0) or (self.state_dim != self.cache_state_dim)
        if uses_subwindow and not self.allow_partial_state:
            raise ValueError(
                f"cache state dim {self.cache_state_dim} but adapter wants "
                f"[{self.state_start}:{self.state_start + self.state_dim}]. This "
                f"slices the TSSM state (e.g. drops z). Use full state "
                f"(state_start=0, state_dim={self.cache_state_dim}) or pass "
                f"allow_partial_state=True to intentionally take this window.")
        assert self.state_start + self.state_dim <= self.cache_state_dim, (
            f"slice [{self.state_start}:{self.state_start + self.state_dim}] exceeds "
            f"cache state dim {self.cache_state_dim}")
        print(
            f"[TSSMMemoryAdapter] cache loaded: {len(d)} episodes; "
            f"using slice [{self.state_start}:{self.state_start + self.state_dim}] "
            f"of {self.cache_state_dim} dims"
        )
        # B-course (future-aware projector): imagine(s_t, a_t) per (ep, step).
        self._imagined_cache = None
        self._imagined_flags = None
        ip = os.environ.get("TSSM_IMAGINED_CACHE", "")
        if ip:
            blob = {}
            for p in ip.split(","):
                p = p.strip()
                if p:
                    blob.update(torch.load(p, map_location="cpu", weights_only=True))
            self._imagined_cache = blob
            print(f"[TSSMMemoryAdapter] imagined-next cache: {len(blob)} episodes",
                  flush=True)

    # ------------------------------------------------------------------
    # state lookup
    # ------------------------------------------------------------------

    def lookup_states(self, episode_ids: List[str], steps: List[int],
                      device: torch.device) -> torch.Tensor:
        """
        Return [B, state_dim] float32 tensor.
        从 cache 里取每个样本对应 step 的 TSSM state。

        如果 episode 不在 cache 里，或 step 越界，用 0 兜底（不会让训练崩）。
        Falls back to zeros if (episode_id, step) is missing — so a slightly
        outdated cache won't crash training, just degrade those samples.
        """
        B = len(episode_ids)
        out = torch.zeros(B, self.state_dim, dtype=torch.float32, device=device)
        for i, (ep, st) in enumerate(zip(episode_ids, steps)):
            ep = str(ep)                     # in case it's bytes/np.str_
            if ep not in self.cache:
                continue
            states = self.cache[ep]          # [T_ep, cache_state_dim] fp16
            # clamp step to valid range; sometimes step indexing is off-by-one
            # at episode boundaries. 边界用 clamp 防越界。
            t = int(st)
            t = max(0, min(t, states.shape[0] - 1))
            s0 = self.state_start
            out[i] = states[t, s0 : s0 + self.state_dim].float().to(device)
        return out

    def _tape_forward(self, z, a, pad_mask, want_aux=False):
        """z [B,L,zdim] float, a [B,L] long, pad_mask [B,L] bool(True=pad) -> [B,K,llm]
        want_aux=True 时额外返回 aux query 的输出 [B,aux_n,sem_dim](训练期回归用)。"""
        dt = self.tp_z_in.weight.dtype
        L = z.shape[1]
        x = self.tp_z_in(z.to(dt)) + self.tp_a_in(self.tp_a_emb(a).to(dt)) \
            + self.tp_pos[:L].unsqueeze(0)
        x = self.tp_lnk1(x)
        q = self.tp_q.unsqueeze(0).expand(z.shape[0], -1, -1)
        if want_aux and self.tape_aux_w > 0:
            qa = self.tp_aux_q.unsqueeze(0).expand(z.shape[0], -1, -1)
            q = torch.cat([q, qa], dim=1)                               # [B, K+aux_n, d]
        o, _ = self.tp_attn1(self.tp_lnq1(q), x, x, key_padding_mask=pad_mask)
        q = q + o
        q = q + self.tp_ffn1(q)
        o, _ = self.tp_attn2(self.tp_lnq2(q), x, x, key_padding_mask=pad_mask)
        q = q + o
        q = q + self.tp_ffn2(q)
        if want_aux and self.tape_aux_w > 0:
            main, aux = q[:, :self.tape_k], q[:, self.tape_k:]
            return self.tp_out(main), self.tp_aux_out(aux)              # [B,K,llm], [B,aux_n,sem]
        return self.tp_out(q)                                          # [B,K,llm]

    def _tape_tokens_cache(self, episode_ids, steps, device):
        """训练/缓存路径: 磁带 = cache[ep][:t+1] 的 z 段 + 动作序列。
        若 tape_aux_w>0: 额外在 [0,t] 上均匀抽 tape_aux_n 个历史时刻, 用该时刻原始
        SigLIP段特征(不经编码器/TSSM)当回归靶, 存入 self._tape_aux_loss(training时)。"""
        if not getattr(self, "_act_seqs", None):
            ap = os.environ.get("TSSM_ACTION_SEQS",
                 "/cluster/scratch/ruiqzhang/tssm_action_seqs.pt")
            self._act_seqs = torch.load(ap, map_location="cpu", weights_only=True)
        want_aux = self.training and self.tape_aux_w > 0
        if want_aux and not getattr(self, "_tape_aux_feats", None):
            fp = os.environ.get("TSSM_TAPE_AUX_FEATS",
                 "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/feat/siglip2_pool/features.pt")
            self._tape_aux_feats = torch.load(fp, map_location="cpu", weights_only=True)
        B = len(episode_ids)
        zs, acts, lens, aux_tgt, aux_mask = [], [], [], [], []
        for ep, st in zip(map(str, episode_ids), steps):
            rows = self.cache.get(ep)
            a = self._act_seqs.get(ep)
            if a is None:
                a = self._act_seqs.get(ep.split(":", 1)[-1])
            if rows is None:
                zs.append(torch.zeros(1, self.tape_zdim)); acts.append(torch.zeros(1, dtype=torch.long)); lens.append(1)
                if want_aux:
                    aux_tgt.append(torch.zeros(self.tape_aux_n, self.tape_sem_dim)); aux_mask.append(False)
                continue
            t = max(0, min(int(st), rows.shape[0] - 1))
            lo = max(0, t + 1 - self.tape_max)
            z = rows[lo:t + 1, self.tape_zoff:self.tape_zoff + self.tape_zdim].float()
            if a is None:
                av = torch.zeros(z.shape[0], dtype=torch.long)
            else:
                # 入向对齐(07-28 tapex尸检): 评测时第k步只可能知道"进入k的动作"
                # a_{k-1}(last_action), 出向动作尚未决策。训练磁带必须同构:
                # z_k 配 a_{k-1}, 首行补0。原实现配出向 act_seqs[k] = 训评错位一格。
                full_a = torch.as_tensor(a, dtype=torch.long)
                inc = torch.cat([torch.zeros(1, dtype=torch.long), full_a[:-1]])
                av = inc[lo:t + 1]
                if av.shape[0] < z.shape[0]:
                    av = torch.cat([av, torch.zeros(z.shape[0] - av.shape[0], dtype=torch.long)])
            zs.append(z); acts.append(av[:z.shape[0]]); lens.append(z.shape[0])
            if want_aux:
                ep_key = ep.split(":", 1)[-1]
                sf = self._tape_aux_feats.get(ep)
                if sf is None:
                    sf = self._tape_aux_feats.get(ep_key)
                if sf is None or t < 1:
                    aux_tgt.append(torch.zeros(self.tape_aux_n, self.tape_sem_dim)); aux_mask.append(False)
                else:
                    idx = torch.linspace(0, t, self.tape_aux_n).round().long().clamp(max=sf.shape[0] - 1)
                    aux_tgt.append(sf[idx, :self.tape_sem_dim].float()); aux_mask.append(True)
        Lm = max(lens)
        zp = torch.zeros(B, Lm, self.tape_zdim, device=device)
        apd = torch.zeros(B, Lm, dtype=torch.long, device=device)
        pm = torch.ones(B, Lm, dtype=torch.bool, device=device)
        for i, (z, a, L) in enumerate(zip(zs, acts, lens)):
            zp[i, :L] = z.to(device); apd[i, :L] = a.to(device); pm[i, :L] = False
        if want_aux:
            main, aux_pred = self._tape_forward(zp, apd, pm, want_aux=True)
            tgt = torch.stack(aux_tgt).to(device)                       # [B,aux_n,sem]
            m = torch.tensor(aux_mask, device=device).float().view(-1, 1, 1)
            cos = torch.nn.functional.cosine_similarity(aux_pred, tgt, dim=-1)  # [B,aux_n]
            self._tape_aux_loss = self.tape_aux_w * (
                (1 - cos) * m.squeeze(-1)).sum() / m.sum().clamp(min=1e-8)
            return main
        self._tape_aux_loss = None
        return self._tape_forward(zp, apd, pm)

    def _history_indices(self, t: int) -> List[int]:
        """K slot indices into [0..t] per self.history_span. Shared by the cache
        path (lookup_state_sequence) and the online path (step_online) so train
        and eval always see identically-sampled history slots.
          history_states=1      -> [t]
          span="recent"         -> last K steps, left-padded with the oldest
          span="full" (default) -> FOREWARN-style: K slots linspace over [0, t]
        """
        K = max(1, self.history_states)
        if K == 1:
            return [t]
        if self.history_span == "navila":
            # 08-18 用户令: 镜像 NaVILA sample_and_pad_images —— 7槽 linspace(0,t,K-1,
            # endpoint=False) 均匀历史 + 显式最新槽[t]; 不足K步左侧补 -1(零state=黑帧等价)
            if t + 1 < K:
                return [-1] * (K - (t + 1)) + list(range(0, t + 1))
            import numpy as _np
            hist = _np.linspace(0, t, num=K - 1, endpoint=False).astype(int).tolist()
            return hist + [t]
        if self.history_span == "recent":
            lo = max(0, t - K + 1)
            idx = list(range(lo, t + 1))
            if len(idx) < K:
                idx = [idx[0]] * (K - len(idx)) + idx
            return idx
        return torch.linspace(0, t, K).round().long().tolist()

    def _ensure_joint(self, device):
        """v12 联合训练 (07-27): state 不再读冻结缓存, 而是训练循环内在线跑
        **可训的** NavigationTSSM(吃冻结特征缓存)。构造期建(KV-prefix 教训:
        懒建的参数进不了优化器)。同 batch 全部 episode 拼成**一次** forward
        (ZeRO 教训: 同一 Linear 在 python 循环里多次调用会 "gradient computed twice")。
        同一次 forward 顺带产出 TSSM 自监督损失(recon/KL/retro), 作防遗忘锚。"""
        if getattr(self, "_joint_tssm", None) is not None:
            return
        import sys as _s
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in _s.path:
            _s.path.insert(0, here)
        from navigation_tssm_feat import NavigationTSSM, TSSMConfig
        ck = os.environ.get("TSSM_JOINT_CKPT",
             "/cluster/scratch/ruiqzhang/tssm_v8_retro_ckpt/checkpoint-85000.pt")
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        cd = blob.get("config", {})
        valid = {k: v for k, v in cd.items() if k in TSSMConfig.__dataclass_fields__}
        tssm = NavigationTSSM(TSSMConfig(**valid))
        tssm.load_state_dict(blob["model"])
        tssm.train()
        # 注册为子模块 → requires_grad/optimizer/保存 全自动覆盖
        self.tssm_joint = tssm.to(device)
        self._joint_tssm = self.tssm_joint
        fp = os.environ.get("TSSM_JOINT_FEATS",
             "/cluster/scratch/ruiqzhang/feats_siglip2pool_r2r_rxr.pt")
        self._joint_feats = torch.load(fp, map_location="cpu", weights_only=True)
        self._joint_anchor_w = float(os.environ.get("TSSM_JOINT_ANCHOR_W", "1.0"))
        if not getattr(self, "_act_seqs", None):
            ap = os.environ.get("TSSM_ACT_SEQS",
                 "/cluster/scratch/ruiqzhang/tssm_action_seqs.pt")
            self._act_seqs = torch.load(ap, map_location="cpu", weights_only=True)
            print(f"[joint] act_seqs {len(self._act_seqs)}", flush=True)
        print(f"[TSSMMemoryAdapter] JOINT mode: tssm={os.path.basename(ck)} "
              f"feats={len(self._joint_feats)} anchor_w={self._joint_anchor_w}", flush=True)

    def joint_state_sequence(self, episode_ids: List[str], steps: List[int],
                             device: torch.device) -> torch.Tensor:
        """v12: 梯度在线版 lookup_state_sequence。返回 [B,K,state_dim](带梯度),
        并把 anchor_w * TSSM自损失 存入 self._tssm_joint_loss。"""
        self._ensure_joint(device)
        B = len(episode_ids)
        K = self.history_states
        uniq = []
        for ep in map(str, episode_ids):
            if ep not in uniq:
                uniq.append(ep)
        feats, acts, lens = [], [], []
        for ep in uniq:
            f = self._joint_feats.get(ep)
            a = None
            if getattr(self, "_act_seqs", None):
                a = self._act_seqs.get(ep)
                if a is None:
                    a = self._act_seqs.get(ep.split(":", 1)[-1])
            if f is None:
                f = torch.zeros(1, self.tssm_joint.config.feat_dim)
            T = f.shape[0]
            if a is None:
                a = torch.zeros(T, dtype=torch.long)
            feats.append(f.float()); acts.append(a[:T].long()); lens.append(T)
        Tm = max(lens)
        fpad = torch.zeros(len(uniq), Tm, feats[0].shape[-1], device=device)
        apad = torch.zeros(len(uniq), Tm, dtype=torch.long, device=device)
        mpad = torch.zeros(len(uniq), Tm, device=device)
        for i, (f, a, L) in enumerate(zip(feats, acts, lens)):
            fpad[i, :L] = f.to(device); apad[i, :a.shape[0]] = a.to(device); mpad[i, :L] = 1
        # dtype 跟随权重(07-27 二修): DeepSpeed bf16 会把 tssm_joint 权重整体转 bf16,
        # 首个 LN 即撞 fp32 输入("expected Float but found BFloat16" = 输入Float/权重BF16)。
        # 一修的"fp32孤岛"方向读反了(对抗 DeepSpeed 无解)。正解: 输入跟权重 dtype,
        # TSSM 全程 bf16 + 关 autocast(内部 dtype 均匀)。23M 小模型 bf16 数值可接受。
        _pdt = next(self.tssm_joint.parameters()).dtype
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.tssm_joint(fpad.to(_pdt), apad, mpad.to(_pdt))  # 单次调用(ZeRO 安全)
        full = torch.stack([st.full for st in out.states], dim=1)    # [E, T, 3072]
        self._tssm_joint_loss = self._joint_anchor_w * out.loss
        res = torch.zeros(B, K, self.state_dim, device=device, dtype=full.dtype)
        s0 = self.state_start
        for i, (ep, st) in enumerate(zip(map(str, episode_ids), steps)):
            e = uniq.index(ep)
            t = max(0, min(int(st), lens[e] - 1))
            idx = self._history_indices(t)
            _v = [j for j, ix in enumerate(idx) if ix >= 0]
            if _v:
                res[i, _v] = full[e, [idx[j] for j in _v], s0:s0 + self.state_dim]
        return res

    def lookup_state_sequence(self, episode_ids: List[str], steps: List[int],
                              device: torch.device) -> torch.Tensor:
        """Return [B, K, state_dim] history sequence ending at the requested step."""
        K = self.history_states
        B = len(episode_ids)
        out = torch.zeros(B, K, self.state_dim, dtype=torch.float32, device=device)
        imag = getattr(self, "_imagined_cache", None)
        for i, (ep, st) in enumerate(zip(episode_ids, steps)):
            ep = str(ep)
            if ep not in self.cache:
                continue
            states = self.cache[ep]
            t = max(0, min(int(st), states.shape[0] - 1))
            idx = self._history_indices(t)
            s0 = self.state_start
            _v = [j for j, ix in enumerate(idx) if ix >= 0]
            if _v:
                out[i, _v] = states[[idx[j] for j in _v],
                                    s0 : s0 + self.state_dim].float().to(device)
            # B-course: swap the LAST slot for imagine(s_t, a_t) so the
            # projector learns to read imagined states (label = real t+1 narr).
            if imag is not None and self._imagined_flags and int(self._imagined_flags[i] or 0) > 0:
                row = imag.get(ep)
                if row is not None:
                    ti = max(0, min(int(st), row.shape[0] - 1))
                    out[i, -1] = row[ti, s0 : s0 + self.state_dim].float().to(device)
        return out

    # ------------------------------------------------------------------
    # forward (matches v3 RSSM signature so existing scaffolding works)
    # ------------------------------------------------------------------

    def forward(
        self,
        frame_vec: torch.Tensor,                       # [B, T, 1152] — ignored
        actions: Optional[torch.Tensor] = None,         # ignored
        collisions: Optional[torch.Tensor] = None,      # ignored
        episode_ids: Optional[List[str]] = None,        # required (cache mode)
        steps: Optional[List[int]] = None,              # required (cache mode)
    ):
        device = frame_vec.device

        # Online (eval) path: caller pre-populated `_online_state_buf` via
        # step_online(); we use it once and clear it.
        if self._online_state_buf is not None:
            s0 = self.state_start
            if self._online_state_buf.dim() == 3:
                states = self._online_state_buf[:, :, s0 : s0 + self.state_dim].to(device)
            else:
                states = self._online_state_buf[:, s0 : s0 + self.state_dim].to(device).unsqueeze(1)
            self._online_state_buf = None
            B = states.shape[0]
        else:
            # Training / cache path
            if episode_ids is None or steps is None:
                raise ValueError(
                    "TSSMMemoryAdapter requires episode_ids and steps in cache "
                    "mode. Either populate them via the dataloader, or call "
                    "step_online() before model.forward() in eval mode."
                )
            B = len(episode_ids)
            if os.environ.get("TSSM_JOINT", "0") == "1" and self.training:
                states = self.joint_state_sequence(episode_ids, steps, device)  # [B,K,D] 带梯度
            else:
                states = self.lookup_state_sequence(episode_ids, steps, device)    # [B, K, state_dim]
            # F2-lite TRAIN (07-18): append the 3 pregenerated k=1 imagined
            # futures -> [B, 1+3, state_dim]; groups match eval's
            # imagine_future_branches() ordering [now, fwd, left, right].
            if os.environ.get("TSSM_FUTURE_TRAIN", "0") == "1":
                if not hasattr(self, "_futures_cache"):
                    fp = os.environ.get("TSSM_FUTURES_CACHE", "")
                    self._futures_cache = (
                        torch.load(fp, map_location="cpu", weights_only=True) if fp and os.path.exists(fp) else None)
                    if self._futures_cache is not None:
                        print(f"[TSSMMemoryAdapter] futures cache loaded: {len(self._futures_cache)} eps", flush=True)
                if self._futures_cache is not None:
                    s0 = self.state_start
                    futs = []
                    for i, (ep, st) in enumerate(zip(episode_ids, steps)):
                        F = self._futures_cache.get(str(ep))
                        if F is None:
                            futs.append(states[i, -1:, :].repeat(3, 1))     # fallback: copy now
                        else:
                            t = max(0, min(int(st), F.shape[0] - 1))
                            futs.append(F[t, :, s0:s0 + self.state_dim].to(device).float())
                    states = torch.cat([states[:, -1:, :], torch.stack(futs, 0)], dim=1)  # [B,4,D]

        # project to LLaMA token space
        # cast to projector's dtype to play well with bf16 autocast
        K = states.shape[1]
        if os.environ.get("TSSM_ALIGN_AUX", "0") == "1" and episode_ids is not None:
            self._align_keys = (list(episode_ids), list(steps) if steps else [0] * B)
        if getattr(self, "proj_mode", "") == "nav3l":
            s = states.reshape(B * K, self.state_dim).to(self.n3_l1.weight.dtype)
            s_ln = self.n3_ln(s)
            hmid = torch.nn.functional.silu(
                self.n3_l2(torch.nn.functional.silu(self.n3_l1(s_ln))))   # [BK,1024]
            s_proj = self.n3_skip(s_ln)                                   # [BK,1024]
            g = torch.sigmoid(self.n3_gate(torch.cat([hmid, s_proj], -1)))
            f = g * hmid + (1 - g) * s_proj
            if getattr(self, "n3_slot_emb", None) is not None and K > 1:
                f = f + self.n3_slot_emb[:K].to(f.dtype).repeat(B, 1)   # 行序=(b0槽0..槽K-1,b1..)
            tokens = self.n3_out(f).view(B, K * self.num_state_tokens, self.llm_dim)
            self._n3_h = hmid                            # 供 temporal 头
            self._n3_gate_mean = float(g.mean())         # 诊断
            self._xattn_aux_loss = self._nav3l_aux(tokens, hmid, episode_ids, steps)
        elif getattr(self, "proj_xattn", False):
            self._aux_steps = steps                     # for world aux labels
            self._xattn_aux_loss = None
            tokens = self._project_xattn(states, episode_ids)          # [B, K*N, llm_dim]
        elif self.proj_slice:
            states = states.to(self.slice_mlp[0].weight.dtype)
            flat_states = states.reshape(B * K, self.state_dim)
            N = self.num_state_tokens
            chunks = self.slice_norm(flat_states).view(B * K, N, self.state_dim // N)
            tokens = self.slice_mlp(chunks) + self.slice_tok_emb       # [B*K, N, llm_dim]
            tokens = tokens.view(B, K * N, self.llm_dim)
        else:
            states  = states.to(self.proj[1].weight.dtype)
            flat_states = states.reshape(B * K, self.state_dim)
            tokens  = self.proj(flat_states)                               # [B*K, N*llm_dim]
            tokens  = tokens.view(B, K * self.num_state_tokens, self.llm_dim)  # [B, K*N, llm_dim]

        # ---- xattn-tape: 追加情景记忆 token (训练=缓存磁带; 评测=在线磁带) ----
        if getattr(self, "tape_xattn", False):
            self._tape_aux_loss = None      # 防上一train batch残留漏进eval/rssm_loss
            if episode_ids is not None and steps is not None and self.training:
                tape_tok = self._tape_tokens_cache(episode_ids, steps, device)
            elif getattr(self, "_online_history", None):
                hist = torch.stack(self._online_history, 0)            # [L, full_dim]
                lo = max(0, hist.shape[0] - self.tape_max)
                z = hist[lo:, self.tape_zoff:self.tape_zoff + self.tape_zdim].float().unsqueeze(0).to(device)
                a = torch.tensor(self._online_actions[lo:], dtype=torch.long,
                                 device=device).unsqueeze(0)
                pm = torch.zeros(1, z.shape[1], dtype=torch.bool, device=device)
                tape_tok = self._tape_forward(z, a, pm)
            else:
                tape_tok = None
            if tape_tok is not None:
                tokens = torch.cat([tokens, tape_tok.to(tokens.dtype)], dim=1)
            _tal = getattr(self, "_tape_aux_loss", None)
            if _tal is not None:
                _prev = getattr(self, "_xattn_aux_loss", None)
                self._xattn_aux_loss = _tal if _prev is None else _prev + _tal

        # ---- 四头辅助损失 (07-20): MLP/slice/xattn 三种投影模式通用 --------
        if self.training and getattr(self, "aux_heads", None):
            if not getattr(self, "proj_xattn", False):
                self._xattn_aux_loss = None      # xattn 分支已在上面自行重置
            _aux = self._aux_heads_loss(tokens, episode_ids, steps)
            if _aux is not None:
                prev = getattr(self, "_xattn_aux_loss", None)
                self._xattn_aux_loss = _aux if prev is None else prev + _aux

        # F2-v3 candidate (P0 verdict 07-19): branch identity SURVIVES the
        # frozen projector (85% linear probe on token deltas) but at 7.6%
        # amplitude — too quiet for SFT gradients to discover. Amplify the
        # token-space deltas, norm-matched so magnitudes stay in-distribution:
        #   T'_b = ||T_b||/||u_b|| * (T_now + amp*(T_b - T_now))
        # Gated by TSSM_FUTURE_AMP (float, 0=off); applies only to the 4-group
        # future layout; train and eval share this exact path (H1 law).
        # ---- Run C: oracle-future 块追加 [<FUT>][h2][h5][h8] (TSSM_ORACLE_FUT=1) ----
        if getattr(self, "oracle_fut", False):
            if self._online_fut_buf is not None:
                fut_states = self._online_fut_buf.to(tokens.device).float()
                self._online_fut_buf = None
            else:
                if self._oracle_fut_cache is None:
                    fp = os.environ.get("TSSM_ORACLE_FUT_CACHE",
                        "/cluster/scratch/ruiqzhang/tssm_futures_oracle258_merged.pt")
                    self._oracle_fut_cache = torch.load(fp, map_location="cpu",
                                                        weights_only=True)
                    print(f"[TSSMMemoryAdapter] oracle-fut cache loaded: "
                          f"{len(self._oracle_fut_cache)} eps", flush=True)
                s0 = self.state_start
                rows, _hit = [], 0
                for ep, st_i in zip(episode_ids, steps):
                    Fm = self._oracle_fut_cache.get(str(ep))
                    if Fm is None:
                        rows.append(torch.zeros(3, self.state_dim))
                    else:
                        t = max(0, min(int(st_i), Fm.shape[0] - 1)); _hit += 1
                        rows.append(Fm[t, :, s0:s0 + self.state_dim].float())
                fut_states = torch.stack(rows, 0).to(tokens.device)
                # 键前缀哨兵 (07-14家法: 某头恒0=键错立刻停)
                self._ofut_dbg = getattr(self, "_ofut_dbg", 0)
                if self._ofut_dbg < 3:
                    print(f"[oracle-fut] batch hit {_hit}/{len(episode_ids)}",
                          flush=True)
                    self._ofut_dbg += 1
            if self.fut_proj_mode == "separate":
                ft = self.fut_proj(fut_states.to(self.fut_proj.weight.dtype))
            else:
                _fs = fut_states.reshape(-1, self.state_dim).to(self.proj[1].weight.dtype)
                ft = self.proj(_fs).view(B, 3, self.num_state_tokens,
                                         self.llm_dim).mean(2)          # [B,3,llm]
            ft = ft + self.fut_hor_emb.unsqueeze(0).to(ft.dtype) \
                    + self.fut_type_emb.unsqueeze(0).to(ft.dtype)
            mk = self.fut_marker.unsqueeze(0).expand(B, -1, -1).to(ft.dtype)
            tokens = torch.cat([tokens.to(ft.dtype), mk, ft], dim=1)    # [B,K*N+4,llm]

        _amp = float(os.environ.get("TSSM_FUTURE_AMP", "0"))
        if _amp > 0 and K == 4:
            g = tokens.view(B, K, self.num_state_tokens, self.llm_dim)
            now, fut = g[:, :1], g[:, 1:]
            u = now + _amp * (fut - now)
            u = u * (fut.norm(dim=-1, keepdim=True) / u.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            tokens = torch.cat([now, u], dim=1).view(B, K * self.num_state_tokens, self.llm_dim)
        env_scale = float(os.environ.get("TSSM_TOKEN_SCALE", "1.0"))
        if env_scale != 1.0:
            tokens = tokens * env_scale
        if os.environ.get("TSSM_SHUFFLE_TOKENS", "0") == "1" and B > 1:
            tokens = tokens[torch.randperm(B, device=tokens.device)]

        # Diagnostic ablation: zero out the TSSM memory tokens at eval to test
        # whether the model actually relies on the state content. Gated by env
        # so normal train/eval are unaffected.
        if os.environ.get("TSSM_ZERO_TOKENS", "0") == "1":
            tokens = torch.zeros_like(tokens)

        # ---- LMK-Bank 银行 token 追加在 state tokens 之后 (置于 ZERO_TOKENS
        # 之后 → 状态清零消融时银行保留, 可做"纯相册"诊断) ----
        if getattr(self, "lmk_bank_on", False):            # v1: SigLIP payload
            _lmk_tok, _lmk_aux = self._lmk_forward(B, episode_ids, steps,
                                                   tokens.device, tokens.dtype)
            tokens = torch.cat([tokens, _lmk_tok], dim=1)
            if _lmk_aux is not None:
                _prev = getattr(self, "_xattn_aux_loss", None)
                self._xattn_aux_loss = _lmk_aux if _prev is None else _prev + _lmk_aux
        elif getattr(self, "lmk_mode", "") == "state":     # v2/v3: state payload
            _fwd = self._lmk_v3_forward if getattr(self, "lmk_v3", False) else self._lmk_v2_forward
            _lmk_tok, _lmk_aux = _fwd(B, episode_ids, steps,
                                      tokens.device, tokens.dtype)
            tokens = torch.cat([tokens, _lmk_tok], dim=1)
            if _lmk_aux is not None:
                _prev = getattr(self, "_xattn_aux_loss", None)
                self._xattn_aux_loss = _lmk_aux if _prev is None else _prev + _lmk_aux

        # reshape to v3-compatible [B, T=1, N, llm_dim]
        # T=1 because the (cached or online) state already summarizes the full
        # trajectory up to the current step (no per-frame state needed).
        state_tokens = tokens.unsqueeze(1)                             # [B, 1, N, llm_dim]

        rssm_loss = torch.tensor(0.0, device=device)
        if getattr(self, "_xattn_aux_loss", None) is not None:
            # world-branch aux (landmark recall) rides the existing rssm_loss
            # plumbing: llava_llama adds tssm_loss_weight(0.1) * this to total.
            rssm_loss = rssm_loss + self._xattn_aux_loss
        return state_tokens, rssm_loss

    # ------------------------------------------------------------------
    # Online (Habitat-sim eval) mode
    # ------------------------------------------------------------------

    def _ensure_online_loaded(self, device: torch.device):
        """Lazy-load TSSM core + SigLIP scene encoder for online rollout."""
        # v12: 联合训练后 TSSM 已漂移(权重存于 ckpt 的 rssm.tssm_joint.*)。评测在线
        # 路径必须用**训练后的** tssm_joint, 而非 TSSM_ONLINE_CKPT 的原始冻结版 ——
        # 否则重演 07-26 的训评错配(在线 state 与 LoRA 所学分布正交)。
        if self._online_tssm is None and getattr(self, "tssm_joint", None) is not None:
            self._online_tssm = self.tssm_joint.to(device).eval()
            self._online_tssm_cfg = self.tssm_joint.config
            for p_ in self._online_tssm.parameters():
                p_.requires_grad_(False)
            print("[TSSMMemoryAdapter] online TSSM <- tssm_joint (联合训练后权重)", flush=True)
        if self._online_tssm is None:
            if self._online_ckpt_path is None:
                raise RuntimeError(
                    "online TSSM mode used but no checkpoint configured. "
                    "Pass online_tssm_ckpt= when constructing TSSMMemoryAdapter, "
                    "or call set_online_paths() before step_online()."
                )
            # Import kept local so cache-mode training has no extra deps.
            import sys
            here = os.path.dirname(os.path.abspath(__file__))
            if here not in sys.path:
                sys.path.insert(0, here)
            from navigation_tssm_feat import NavigationTSSM, TSSMConfig  # noqa: E402

            blob = torch.load(self._online_ckpt_path, map_location="cpu",
                              weights_only=False)
            cfg_dict = blob.get("config", {})
            valid = {k: v for k, v in cfg_dict.items()
                     if k in TSSMConfig.__dataclass_fields__}
            z_dim = int(valid.get("stoch", TSSMConfig.stoch)) * int(
                valid.get("classes", TSSMConfig.classes)
            )
            last_state_dim = int(valid.get("h_dim", TSSMConfig.h_dim)) + z_dim
            concat_state_dim = (
                int(valid.get("n_layers", TSSMConfig.n_layers))
                * int(valid.get("d_model", TSSMConfig.d_model))
                + z_dim
            )
            want_state_end = self.state_start + self.state_dim
            if want_state_end > last_state_dim and want_state_end <= concat_state_dim:
                if valid.get("deter_type") != "concat_o":
                    print(
                        "[TSSMMemoryAdapter] forcing online TSSM "
                        "deter_type=concat_o because adapter expects "
                        f"state slice end={want_state_end}, "
                        f"last_state_dim={last_state_dim}, "
                        f"concat_state_dim={concat_state_dim}"
                    )
                    valid["deter_type"] = "concat_o"
            cfg = TSSMConfig(**valid)
            tssm = NavigationTSSM(cfg).to(device).eval()
            tssm.load_state_dict(blob["model"])
            online_state_dim = int(getattr(tssm, "state_dim", last_state_dim))
            if want_state_end > online_state_dim:
                raise RuntimeError(
                    "online TSSM state is too small for adapter slice: "
                    f"online_state_dim={online_state_dim}, "
                    f"slice=[{self.state_start}:{want_state_end}], "
                    f"deter_type={cfg.deter_type}, "
                    f"checkpoint={self._online_ckpt_path}"
                )
            for p in tssm.parameters():
                p.requires_grad_(False)
            self._online_tssm = tssm
            self._online_tssm_cfg = cfg
            print(f"[TSSMMemoryAdapter] online TSSM loaded "
                  f"(step={blob.get('step', '?')}, feat_dim={cfg.feat_dim}, "
                  f"deter_type={cfg.deter_type}, state_dim={online_state_dim})")

        if self._online_siglip is None:
            from transformers import (SiglipVisionModel,  # noqa: E402
                                      SiglipImageProcessor)
            path = self._online_siglip_path
            if path is None:
                raise RuntimeError(
                    "online SigLIP path not configured. "
                    "Set online_siglip_path= or call set_online_paths()."
                )
            sig = SiglipVisionModel.from_pretrained(
                path, torch_dtype=torch.float32).to(device).eval()
            for p in sig.parameters():
                p.requires_grad_(False)
            proc = SiglipImageProcessor.from_pretrained(path)
            mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(device)
            std  = torch.tensor(proc.image_std ).view(1, 3, 1, 1).to(device)
            size = (proc.size.get("height")
                    or proc.size.get("shortest_edge") or 384)
            self._online_siglip = (sig, mean, std, size)
            print(f"[TSSMMemoryAdapter] online SigLIP loaded "
                  f"(input={size}x{size})")

    def _ensure_online_dualtower(self, device: torch.device) -> None:
        """WAM线双塔 (08-02): 惰性加载 DepthAnything-V2-S 的 DINOv2-S 骨干。
        eval 环境 transformers=4.37 没有 DepthAnything 类, 但 Dinov2Model 有;
        DA 检查点里骨干权重带 'backbone.' 前缀, 剥掉即可原位加载(层名同构)。
        深度头(DPT)不加载 — 我们只要骨干 patch 嵌入。"""
        if getattr(self, "_online_da", None) is not None:
            return
        import json as _json
        from transformers import Dinov2Config, Dinov2Model
        da_path = os.environ.get(
            "TSSM_ONLINE_DA",
            "/cluster/scratch/ruiqzhang/hf_cache/hub/"
            "models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/"
            "5426e4f0f36572d16453bbda7a8389317b1bef99")
        with open(os.path.join(da_path, "config.json")) as f:
            bb = _json.load(f)["backbone_config"]
        cfg = Dinov2Config(**{k: v for k, v in bb.items()
                              if k in Dinov2Config().to_dict()})
        da = Dinov2Model(cfg)
        try:
            from safetensors.torch import load_file as _lf
            sd = _lf(os.path.join(da_path, "model.safetensors"))
        except (ImportError, FileNotFoundError):
            sd = torch.load(os.path.join(da_path, "pytorch_model.bin"),
                            map_location="cpu", weights_only=True)
        bb_sd = {k[len("backbone."):]: v for k, v in sd.items()
                 if k.startswith("backbone.")}
        missing, unexpected = da.load_state_dict(bb_sd, strict=False)
        real_missing = [m for m in missing if not m.startswith("pooler")]
        assert not real_missing, f"DA backbone missing keys: {real_missing[:5]}"
        da = da.to(device).eval()
        for p in da.parameters():
            p.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self._online_da = (da, mean, std)
        print(f"[TSSMMemoryAdapter] online DA-V2-S backbone loaded "
              f"({sum(p.numel() for p in da.parameters())/1e6:.0f}M, "
              f"unexpected={len(unexpected)})", flush=True)

    def set_online_paths(self, tssm_ckpt: str, siglip_path: str) -> None:
        """Configure the paths after construction (used by eval scripts that
        load the model first then attach the online weights)."""
        self._online_ckpt_path = tssm_ckpt
        self._online_siglip_path = siglip_path

    def reset_online(self) -> None:
        """Reset the agent's TSSM state. Call this on every Habitat episode
        boundary (envs.reset_at(...) or initial envs.reset())."""
        self._online_state = None
        self._online_state_buf = None
        self._online_history = []
        self._online_actions = []
        if getattr(self, "lmk_bank_on", False):
            self._lmk_online.reset()         # LMK-Bank v1: 每集清库
        if getattr(self, "lmk_mode", "") == "state":
            self.lmk_v2_reset()              # LMK-Bank v2: 每集清库
        self._adapt_reset_episode()          # fresh decoder copy per episode (TTA)

    @torch.no_grad()
    @torch.no_grad()
    def _encode_online_frame(self, rgb_image, device) -> torch.Tensor:
        """Encode ONE RGB view into the online TSSM's feature space ([1, feat_dim]).
        Does NOT touch self._last_online_feat (beacon reads that as the CURRENT
        view) — callers stash what they need themselves."""
        from PIL import Image
        import numpy as np

        sig, mean, std, size = self._online_siglip

        # Normalize input to PIL.Image RGB
        if not isinstance(rgb_image, Image.Image):
            rgb_image = Image.fromarray(np.asarray(rgb_image)).convert("RGB")
        else:
            rgb_image = rgb_image.convert("RGB")

        # Feature mode follows the ONLINE TSSM's feat_dim so online features
        # always match the cache the LoRA was trained on:
        #   1152  -> legacy: BICUBIC resize, final-layer mean-pool
        #   P*1152-> v5 lm2-p9: BILINEAR resize (matches precompute_features_patch.py),
        #            hidden_states[-2] (VLM tower layer, no post_layernorm),
        #            gxg block-average, flatten. Requires the navila_8f tower
        #            as online SigLIP — see the guard below.
        online_feat_dim = int(self._online_tssm_cfg.feat_dim)

        # ⚠ 1152 有**两种**互不相容的来源, 维度一样所以不会报错, 但空间正交:
        #   (a) legacy: navila 塔 last_hidden_state 在 patch 上 mean-pool
        #   (b) v8/siglip2: google/siglip2-so400m 的**原生 MAP 池化头** pooler_output
        # v8 起的 TSSM 训练缓存走 (b) (见 precompute_pooled_siglip2.py:36), 配方 =
        # BILINEAR 384 + (x-0.5)/0.5 + pooler_output。若在线误走 (a), state 与训练
        # 缓存近正交 -> 导航塌 (07-26 实测踩坑, v8down ck250/500 冒烟因此作废)。
        # 判定: 显式 TSSM_ONLINE_FEAT=siglip2_pool 优先, 否则按塔路径含 "siglip2" 自动识别。
        _feat_mode = os.environ.get("TSSM_ONLINE_FEAT", "").strip().lower()
        _is_sig2 = (_feat_mode == "siglip2_pool" or
                    (_feat_mode == "" and "siglip2" in str(self._online_siglip_path).lower()))
        if _feat_mode == "dualtower_p9":
            # WAM线双塔 (08-02): SigLIP2-B/16 [9,768] ⊕ DA-V2-S骨干 [9,384] -> [1,9*1152]。
            # 维度=10368 与 lm2-p9 撞车, 故必须显式 env 选择; 配方与
            # precompute_dualtower.py 逐位一致 (384 BILINEAR / (x-.5)/.5 | 378+ImageNet)。
            self._ensure_online_dualtower(device)
            rgb = rgb_image.resize((384, 384), Image.BILINEAR)
            x = (torch.from_numpy(np.asarray(rgb).copy()).float()
                 .permute(2, 0, 1).unsqueeze(0).to(device) / 255.0)     # [1,3,384,384] in [0,1]
            s_out = sig(pixel_values=(x - 0.5) / 0.5).last_hidden_state  # [1,576,768]
            s_grid = (s_out.view(1, 24, 24, 768).view(1, 3, 8, 3, 8, 768)
                      .mean(dim=(2, 4)).reshape(1, 9, 768))
            import torch.nn.functional as _F
            da, da_mean, da_std = self._online_da
            xd = _F.interpolate(x, size=(378, 378), mode="bilinear", align_corners=False)
            hs = da(pixel_values=(xd - da_mean) / da_std).last_hidden_state  # [1,730,384] 含CLS
            hs = hs[:, 1:]                                              # [1,729,384]
            d_grid = (hs.view(1, 27, 27, 384).view(1, 3, 9, 3, 9, 384)
                      .mean(dim=(2, 4)).reshape(1, 9, 384))
            feat = torch.cat([s_grid, d_grid], dim=-1).reshape(1, 9 * 1152).float()
        elif online_feat_dim == 1152 and _is_sig2:
            rgb = rgb_image.resize((size, size), Image.BILINEAR)
            arr = torch.from_numpy(np.asarray(rgb).copy()).float().permute(2, 0, 1) / 255.0
            arr = ((arr.unsqueeze(0).to(device) - 0.5) / 0.5)          # 与 precompute 逐位一致
            feat = sig(pixel_values=arr).pooler_output.float()          # [1, 1152] MAP 头
        elif online_feat_dim % 1152 == 0 and online_feat_dim // 1152 > 1:
            Pg = online_feat_dim // 1152
            g = int(round(Pg ** 0.5))
            assert g * g == Pg and 27 % g == 0, f"unsupported gist grid {Pg}"
            if "navila_8f" not in str(self._online_siglip_path):
                print("[TSSMMemoryAdapter] WARNING: online TSSM expects lm2-p9 "
                      f"features but online SigLIP path is {self._online_siglip_path} "
                      "— v5 caches were built with navila_8f/vision_tower; a wrong "
                      "tower gives near-orthogonal states (cos~0.33).")
            rgb = rgb_image.resize((size, size), Image.BILINEAR)
            arr = torch.from_numpy(np.asarray(rgb)).float().permute(2, 0, 1) / 255.0
            arr = arr.unsqueeze(0).to(device)                          # [1,3,H,W]
            arr = (arr - mean) / std
            hs = sig(pixel_values=arr, output_hidden_states=True
                     ).hidden_states[-2]                               # [1,729,1152] raw
            blk = 27 // g
            feat = (hs.view(1, 27, 27, 1152)
                      .view(1, g, blk, g, blk, 1152)
                      .mean(dim=(2, 4))
                      .reshape(1, online_feat_dim))                    # [1, P*1152]
            feat = feat.float()
        else:
            # SigLIP preprocessing: resize to (size, size), to tensor, normalize.
            # Match precompute_features.py exactly: bicubic resize, /255, mean/std.
            rgb = rgb_image.resize((size, size), Image.BICUBIC)
            arr = torch.from_numpy(np.asarray(rgb)).float().permute(2, 0, 1) / 255.0
            arr = arr.unsqueeze(0).to(device)                          # [1,3,H,W]
            arr = (arr - mean) / std

            # Mean-pool last_hidden_state over patch tokens (no CLS for SigLIP).
            feat = sig(pixel_values=arr).last_hidden_state.mean(dim=1) # [1, feat_dim]
            feat = feat.float()
        return feat

    @torch.no_grad()
    def encode_future_view(self, rgb_image, device: Optional[torch.device] = None) -> None:
        """ORACLE-FUTURE eval sensor: encode a sim-rendered next-waypoint view
        into the SAME lm2-p9 space the future slot was trained on, and stash it;
        the next get_future_feats() consumes it (train/eval interface identity)."""
        if device is None:
            if getattr(self, "proj_mode", "") == "nav3l":
                device = self.n3_l1.weight.device   # nav3l: proj=None 且无 slice_mlp
            else:
                device = (self.x_out.weight.device if getattr(self, "proj_xattn", False)
                          else (self.slice_mlp[0] if self.proj is None else self.proj[1]).weight.device)
        self._ensure_online_loaded(device)
        feat = self._encode_online_frame(rgb_image, device)             # [1, P*1152]
        self._online_future_feat = feat.view(-1, 1152).detach()         # [P, 1152]

    def step_online(self,
                    rgb_image,                       # PIL.Image OR np.ndarray uint8 HxWx3
                    last_action: int = 0,
                    device: Optional[torch.device] = None) -> torch.Tensor:
        """Roll the TSSM one step on the latest observation, store the state
        in the buffer, and return it for inspection/logging.

        rgb_image     : current camera observation.
        last_action   : NaVILA action ID taken to land here (0 at episode start).
        device        : where to run TSSM/SigLIP; defaults to projector device.

        After this call, the next forward() will project this state into
        LLM tokens regardless of cache lookup.
        """
        from PIL import Image
        import numpy as np

        if device is None:
            if getattr(self, "proj_mode", "") == "nav3l":
                device = self.n3_l1.weight.device   # nav3l: proj=None 且无 slice_mlp
            else:
                device = (self.x_out.weight.device if getattr(self, "proj_xattn", False)
                          else (self.slice_mlp[0] if self.proj is None else self.proj[1]).weight.device)
        self._ensure_online_loaded(device)

        feat = self._encode_online_frame(rgb_image, device)

        # Stash the current-frame feature for read-only consumers (e.g. the
        # HWM stop-beacon dump) so they don't re-run SigLIP.
        self._last_online_feat = feat.detach()

        # First step of an episode: get a fresh prior and ignore last_action.
        if self._online_state is None:
            self._online_state = self._online_tssm.get_initial_state(
                1, device, dtype=torch.float32)
            a_prev = torch.zeros(1, dtype=torch.long, device=device)
        else:
            a_prev = torch.tensor([int(last_action)], dtype=torch.long,
                                  device=device)

        self._online_state = self._online_tssm.inference_step(
            self._online_state, a_prev, feat)
        state_vec = self._online_state.full                             # [1, full_dim]

        # Accumulate full history this episode, then expose the SAME K-slot
        # sequence the cache path feeds during training (see _history_indices).
        self._online_history.append(state_vec.squeeze(0))               # [full_dim]
        if getattr(self, "tape_xattn", False):
            if not hasattr(self, "_online_actions") or self._online_actions is None:
                self._online_actions = []
            self._online_actions.append(int(a_prev))
        # TTA: self-supervise the decoder copy on (state_t, current-frame feat).
        self._adapt_step(state_vec.squeeze(0), feat)
        idx = self._history_indices(len(self._online_history) - 1)
        _z = torch.zeros_like(self._online_history[0])
        seq = torch.stack([self._online_history[j] if j >= 0 else _z for j in idx], dim=0)  # [K, full_dim]
        # [1, K, full_dim]; forward() slices [state_start:state_start+state_dim]
        self._online_state_buf = seq.unsqueeze(0)

        # ---- LMK-Bank: 在线银行观察当前帧 (与离线 replay 同一 write_rule) ----
        if getattr(self, "lmk_bank_on", False):
            f = feat.squeeze(0).float().cpu()
            pooled = f.view(-1, 1152).mean(0) if f.numel() > 1152 else f  # p9→池化/legacy原样
            self._lmk_online.observe(pooled, int(a_prev),
                                     len(self._online_history) - 1)
        return state_vec

    @torch.no_grad()
    def imagine_future_branches(self, k: int = 3):
        """FUTURE-ZS (zero-shot F2 probe, 07-17): after step_online, replace the
        1-state buffer with FOUR state groups [now, fwd, left, right] — each
        future = imagine k steps open-loop (branch-fixed first action: fwd50/
        left30/right30; continuation via the BC actor). forward() then projects
        16 tokens (4 groups x 4) through the frozen stage-1 projector.
        Gated by the caller (env TSSM_FUTURE_ZS); additive, touches nothing else.
        Prior fidelity @k=3: cos 0.986 / actor-agree 92.9% (probe_prior_fidelity)."""
        import torch.nn as nn
        if self._online_state is None or self._online_tssm is None:
            return
        dev = self._online_state.h.device
        if not hasattr(self, "_zs_actor"):
            path = os.environ.get("TSSM_BC_ACTOR",
                                  "/cluster/scratch/ruiqzhang/bc_action_head_v5.pt")
            blob = torch.load(path, map_location="cpu", weights_only=False)
            m = nn.Sequential(nn.LayerNorm(self.state_dim), nn.Linear(self.state_dim, 512),
                              nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 10))
            m.load_state_dict(blob["model"])
            self._zs_actor = m.to(dev).eval()
        # TSSM_FUTURE_MACRO=1 (07-20): deterministic H-step programs instead of
        # BC-actor continuation. The BC actor's "typicality" collapses branch
        # identity (85/51/24% decodable at H=1/2/3); fixed programs keep the
        # branches apart by construction, and the world model itself holds up
        # (given-action rollout cos 0.978 @k=5). Training twin: FUTURE_H in
        # scripts/data/build_future_course.py uses the same H.
        macro = os.environ.get("TSSM_FUTURE_MACRO", "0") == "1"
        groups = [self._online_state.full.squeeze(0)]
        for first in (2, 5, 8):   # fwd50 / left30 / right30
            s = self._online_tssm.imagine_step(
                self._online_state, torch.tensor([first], dtype=torch.long, device=dev))
            for _ in range(k - 1):
                if macro:
                    a = torch.tensor([2], dtype=torch.long, device=dev)  # commit forward
                else:
                    a = self._zs_actor(s.full).argmax(-1)
                s = self._online_tssm.imagine_step(s, a)
            groups.append(s.full.squeeze(0))
        if os.environ.get("TSSM_FUTURE_SHUFFLE", "0") == "1":
            # swap probe: rotate the 3 futures so every group is mislabeled
            # ([now,fwd,left,right] -> [now,left,right,fwd]). SR unchanged vs
            # the unshuffled run  =>  the LLM never reads the future groups.
            groups = [groups[0], groups[2], groups[3], groups[1]]
        self._online_state_buf = torch.stack(groups, 0).unsqueeze(0)   # [1,4,full]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

RSSMMemoryAdapter = TSSMMemoryAdapter


if __name__ == "__main__":
    torch.manual_seed(0)
    adapter = TSSMMemoryAdapter(
        state_dim=512, llm_dim=4096, num_state_tokens=4,
        cache_path=None,        # test without cache → fallback zeros path
    )
    print(f"projector params: "
          f"{sum(p.numel() for p in adapter.proj.parameters())/1e6:.2f}M")

    B, T = 3, 8
    frame_vec = torch.randn(B, T, 1152)
    eps = ["ep_42", "ep_43", "ep_99"]
    steps = [3, 7, 0]
    tokens, loss = adapter(frame_vec, episode_ids=eps, steps=steps)
    print(f"output state_tokens shape: {tokens.shape}  loss: {loss.item()}")
    assert tokens.shape == (B, 1, 4, 4096)
    print("OK")

#!/usr/bin/env python3
# ============================================================
# Stage-2:Qwen 全量微调 —— 教大模型"边看边记"地导航
# ============================================================
# 【这份文件从头到尾在干一件事】把一个通用视觉语言大模型(Qwen3.5 4B/9B)
# 微调成导航员:每一步喂它"一张当前画面 + 一段指令 + 16个记忆token",
# 让它输出下一个动作(前进25cm/左转15度/停...)。
#
# 【全文地图】(按执行顺序, 每段正文前还有细注释)
#   第0段  Blackwell显卡补丁(和算法无关, 纯粹让新卡能跑)
#   第1段  读环境变量 = 实验的全部旋钮(哪份数据/哪个模型/学习率...)
#   main():
#     ①模型加载  ——载入Qwen, 解冻语言部分, 冻结视觉部分
#     ②记忆投影器——把TSSM的3072维状态向量翻译成16个"词",塞进句子里
#     ③READNCE头 ——两个小考官:逼模型的脑内状态能"回忆过去/预想未来"
#     ④DeepSpeed  ——把9B模型的优化器账本切片摊到4张卡(不然单卡装不下)
#     ⑤数据       ——R2R+RxR的(画面,指令,答案)三元组, 各卡分片互不重复
#     ⑥训练循环   ——前向算损失(动作预测+两个考官) → 反传 → 攒够64条更新一次
#     ⑦存档       ——每SAVE_STEPS步存两份:可续训的分片版+可直接评测的整模型
#
# 消融开关:
#   NO_MEMORY=1  → 无记忆对照臂(记忆token恒为空白)——判"记忆价值"的另一半
# ============================================================
import io
import json
import os
import random

# Blackwell(sm_120) 修复: fla 大量 kernel 的 autotune 配置含 num_warps=8, 在
# Triton 3.2-3.6 上编出非法代码(fla#734 同族; 我们实测 misaligned address)。
# 设 TRITON_MAX_WARPS=4 时全局封顶, 必须在任何 fla/模型 import 之前生效。
if os.environ.get("TRITON_MAX_WARPS"):
    import triton as _tr
    _MAXW = int(os.environ["TRITON_MAX_WARPS"])
    _OrigConfig = _tr.Config
    class _CappedConfig(_OrigConfig):
        def __init__(self, *args, **kwargs):
            if kwargs.get("num_warps") and kwargs["num_warps"] > _MAXW:
                kwargs["num_warps"] = _MAXW
            super().__init__(*args, **kwargs)
    _tr.Config = _CappedConfig
    print(f"[blackwell-fix] triton.Config num_warps capped at {_MAXW}", flush=True)
import sqlite3
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

# 复用 Stage-1 的零件:数据集、projector、占位符逻辑(保证两阶段完全一致)
from train_stage1 import SYS, NavDataset, StateProjector, pick_placeholder  # noqa

# ---- 第1段: 实验旋钮 ----------------------------------------------------
# 全部用 os.environ.get("名字", 默认值) 的写法: 发射脚本(sbatch)里 export 什么,
# 这里就读到什么; 不 export 就用默认值。好处是改实验不用改代码, 而且 sbatch
# 文件本身就成了这次实验的完整"配置存档"。
MODEL_DIR = os.environ.get("QWEN_DIR", "/cluster/scratch/ruiqzhang/models/Qwen3.5-9B")
STAGE1_PROJ = os.environ.get("STAGE1_PROJ", "/cluster/scratch/ruiqzhang/qwen35_stage1/projector_step4000.pt")
ANN = os.environ.get("ANN", "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json")
SQLITE = os.environ.get("SQLITE", "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/train.sqlite")
STATE_CACHE = os.environ.get("STATE_CACHE", "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/tssm_states_v5_lm2p9.pt")
OUT = os.environ.get("OUT", "/cluster/scratch/ruiqzhang/qwen35_stage2")
DS_CONFIG = os.environ.get("DS_CONFIG",
                           "/cluster/project/rsl/ruiqzhang/codes/qwen35-vln/configs/zero3.json")
LR = float(os.environ.get("LR", "1e-5"))              # [S2-③]
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5500"))  # ≈ R2R一个epoch(354k÷64)
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "250"))
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(448 * 448)))
NO_MEMORY = os.environ.get("NO_MEMORY", "0") == "1"   # 消融:无记忆对照
# ---- RxR 多源 (07-30): 设了 ANN_RXR 即启用双源, state键切到 "dom:ep" 口径 ----
ANN_RXR = os.environ.get("ANN_RXR", "")
SQLITE_RXR = os.environ.get("SQLITE_RXR", "/cluster/scratch/ruiqzhang/NaVILA-Dataset/RxR/train.sqlite")
# ---- RxR扩容第三源 (08-02, full6): 13334新集, 键同"rxr:ep"口径(与老库集合零冲突已验) ----
ANN_RXR2 = os.environ.get("ANN_RXR2", "")
SQLITE_RXR2 = os.environ.get("SQLITE_RXR2", "/cluster/scratch/ruiqzhang/rxr_expand/expand.sqlite")
# ---- 读出检索 rnceans (07-30, 与NaVILA臂同构): 答案位隐状态查冻结ncev5钥匙空间 ----
READNCE = os.environ.get("READNCE", "0") == "1"
READNCE_KEYS = os.environ.get("READNCE_KEYS", "/cluster/scratch/ruiqzhang/tssm_readnce_keys_ncev5.pt")
READNCE_ACTS = os.environ.get("READNCE_ACTS", "/cluster/scratch/ruiqzhang/tssm_action_seqs.pt")
READNCE_W = float(os.environ.get("READNCE_W", "0.2"))
READNCE_DYN_W = float(os.environ.get("READNCE_DYN_W", "0.2"))
READNCE_TAU = float(os.environ.get("READNCE_TAU", "0.1"))
READNCE_RETRO_LAGS = [int(x) for x in os.environ.get("READNCE_RETRO_LAGS", "3,5,10").split(",") if x.strip()]
READNCE_DYN_LAGS = [int(x) for x in os.environ.get("READNCE_DYN_LAGS", "1,3").split(",") if x.strip()]
SEED = 42
# 全局batch固定64:4卡 × 每卡1条 × 16累积(和NaVILA全部实验同口径)
ACCUM = int(os.environ.get("ACCUM", "16"))


def main():
    import deepspeed
    from transformers import AutoProcessor, AutoModelForImageTextToText

    # ---- 多卡起手式:torchrun 给每个进程发了编号(rank) ----
    # 4张卡=4个一模一样的python进程同时跑这份文件。rank是"我是几号"(0~3),
    # world是"一共几个"。它们各算各的梯度, DeepSpeed负责在幕后把梯度求平均,
    # 所以4个进程的模型永远保持一致。rank=0号进程额外负责打日志和存档。
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    torch.manual_seed(SEED + rank)

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    os.makedirs(OUT, exist_ok=True)
    log(f"Stage-2 启动: world={world}, LR={LR}, NO_MEMORY={NO_MEMORY}")

    # ---- 1. 模型 + 处理器 ----
    processor = AutoProcessor.from_pretrained(MODEL_DIR, max_pixels=MAX_PIXELS)
    tok = processor.tokenizer
    tok.padding_side = "right"   # 批量labels掩码假定[prompt|answer|pad]布局, 左padding会错位
    # 直接联训 (07-27 用户令): STAGE1_PROJ=none → 无热启动, projector 随机初始化
    # 与全量FT同步学。全量域梯度直达一切, 无 LoRA 域"投影器被饿死"之虞(毒药案是
    # 冻结LLM域的病); FutureNav 全新头从零联训先例。⚠盯 loss: 前几百步应单调降。
    if str(STAGE1_PROJ).strip().lower() in ("", "none", "0"):
        blob = None
        ph, ph_id = pick_placeholder(tok)
        cfg = {"placeholder": ph,
               "n_mem": int(os.environ.get("N_MEM", "16")),
               "d_model": None,     # 模型载入后按 hidden_size 回填(4B/9B 通用)
               "d_slice": int(os.environ.get("D_SLICE", "64")),
               "hidden": int(os.environ.get("PROJ_HIDDEN", "1024"))}
        log(f"直接联训: 无stage1热启动, placeholder={ph}, n_mem={cfg['n_mem']}")
    else:
        blob = torch.load(STAGE1_PROJ, map_location="cpu")
        cfg = blob["cfg"]
        ph = cfg["placeholder"]
        ph_id = tok.encode(ph, add_special_tokens=False)[0]

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

    # [S2-①] 解冻LLM、冻结视觉编码器
    # "冻结"=不给这部分参数算梯度=训练中永不改动。视觉编码器(把图变向量的
    # 部分)是预训练见过海量图片才练成的, 我们的导航数据量级远不够重教它,
    # 硬教只会教坏(灾难性遗忘)。所以只训语言部分和新加的小模块。
    model.requires_grad_(True)
    frozen = 0
    for name, p in model.named_parameters():
        if "visual" in name or "vision" in name:   # Qwen的视觉模块名带visual
            p.requires_grad_(False)
            frozen += p.numel()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    log(f"冻结视觉参数: {frozen/1e6:.0f}M; 其余全量可训")

    # [S2-②] projector 热启动
    if cfg["d_model"] is None:
        cfg["d_model"] = getattr(model.config, "hidden_size", None) \
            or model.config.text_config.hidden_size
    proj = StateProjector(n_tok=cfg["n_mem"], d_model=cfg["d_model"],
                          d_slice=cfg["d_slice"], hidden=cfg["hidden"]).float()
    if blob is not None:
        proj.load_state_dict(blob["proj"])
        log(f"projector 热启动自 {STAGE1_PROJ}")
    else:
        log(f"projector 随机初始化 (n_tok={cfg['n_mem']}, d_model={cfg['d_model']}) — 直接联训")

    # 【投影器是什么】TSSM世界模型把"走到现在看过的一切"压缩成一个3072维
    # 向量(state); 但LLM只认"词向量"。StateProjector就是翻译官: 把3072维
    # 切成16片, 每片过一个共享小网络抬升到LLM的词向量维度 → 得到16个
    # "记忆词", 稍后会顶替句子里的16个占位符。
    # 把projector挂成模型的子模块,DeepSpeed就会把它一起管起来
    model.state_projector = proj

    # ---- 1.5 读出检索头 (07-30 rnceans移植): 必须在deepspeed.initialize前建好 ----
    # 【READNCE是什么】给LLM加两门"附加考试", 逼它真的去读记忆token而不是无视:
    #   retro头(回忆考): 拿LLM决策时刻的脑内状态, 去一排帧指纹里认出"3/5/10步
    #           之前我看到的是哪一帧"。答不对=没在用记忆 → 罚分(损失)。
    #   dyn头(预想考): 告诉它接下来要执行的动作串, 让它认出"1/3/5步之后会看到
    #           哪一帧"。逼它在脑内装一个"往前推演"的小模型。
    # 帧指纹(keys)来自冻结的TSSM塔, 训练中不动; 考试只在训练时进行, 推理时零开销。
    readnce_keys, readnce_acts = None, None
    if READNCE:
        _blob = torch.load(READNCE_KEYS, map_location="cpu", weights_only=True)
        readnce_keys = _blob["keys"] if isinstance(_blob, dict) and "keys" in _blob else _blob
        readnce_acts = torch.load(READNCE_ACTS, map_location="cpu", weights_only=True)
        _dc = next(iter(readnce_keys.values())).shape[-1]
        _dm = cfg["d_model"]
        model.readnce_lag_emb = nn.Embedding(max(1, len(READNCE_RETRO_LAGS)), 64)
        model.readnce_dyn_lag_emb = nn.Embedding(max(1, len(READNCE_DYN_LAGS)), 64)
        model.readnce_act_emb = nn.Embedding(10, 32)
        # 动作窗保序拼接 (07-31修, 与NaVILA侧同款): mean丢顺序, concat固定槽位+pad0
        _act_kmax = max(READNCE_DYN_LAGS) if READNCE_DYN_LAGS else 1
        model.readnce_q_head = nn.Sequential(
            nn.Linear(_dm + 64, 512), nn.GELU(), nn.Linear(512, _dc))
        model.readnce_dyn_head = nn.Sequential(
            nn.Linear(_dm + 64 + 32 * _act_kmax, 512), nn.GELU(), nn.Linear(512, _dc))
        log(f"READNCE on(answer锚): keys={len(readnce_keys)}eps dim={_dc} "
            f"retro={READNCE_RETRO_LAGS} dyn={READNCE_DYN_LAGS} "
            f"w={READNCE_W}/{READNCE_DYN_W} tau={READNCE_TAU}")

    # ---- 2. 记忆注入钩子(与Stage-1/探针同一套) ----
    # 【钩子(hook)是什么】pytorch允许在某一层"埋伏"一个函数, 每次该层算完都
    # 会被叫一次, 可以偷看/篡改它的输出。我们埋伏在"词嵌入层"(词id→词向量):
    # 句子里预先写了16个占位符词, 嵌入层把它们变成向量后, 钩子把这些位置
    # 整体换成投影器产出的16个记忆向量——记忆就这样"混"进了句子。
    # holder是个传纸条的小盒子: 训练循环每个批次把新记忆放进去, 钩子取用。
    holder = {"mem": None}

    def hook(module, inputs, output):
        if holder["mem"] is None:
            return output
        mask = inputs[0] == ph_id
        if not bool(mask.any()):
            return output
        out = output.clone()
        out[mask] = holder["mem"].reshape(-1, out.shape[-1]).to(out.dtype)
        return out

    model.get_input_embeddings().register_forward_hook(hook)

    # ---- 2.5 READNCE: 钩子截取考卷用的隐状态 ----
    # 考官需要"LLM脑内状态"当考卷。锚点两选(08-07 用户拍板换早中层):
    #   READNCE_ANCHOR_LAYER=N → 截第N层decoder块的输出(REPA/ROCKET/VLA同行
    #     共识: 表征类辅助损失挂早中层; 9B首选11=34%深度的全注意力层——
    #     记忆token的按址读取靠全注意力, 线性注意力层只有压缩状态不擅检索);
    #   不设 → 老口径final norm(最后层, 4B冠军配方位置, 留作对照)。
    # 双锚支持(08-07): retro与dyn可各挂一层(READNCE_DYN_ANCHOR_LAYER不设或
    # 与retro同层时共用一份考卷)。当前拍板: retro@11(表征早中层), dyn@23
    # (72%深度全注意力, 预想考更靠近决策端)。
    _h_holder = {"h": None, "hd": None}
    if READNCE:
        def _find_layer(spec, fallback_final):
            if spec:
                cands = [(n, m) for n, m in model.named_modules()
                         if n.endswith(f".layers.{spec}")
                         and "visual" not in n and "vision" not in n]
                assert cands, f"READNCE: 找不到语言模型第{spec}层"
                return cands[-1]
            norms = [(n, m) for n, m in model.named_modules()
                     if n.endswith(".norm") and ".layers." not in n
                     and "visual" not in n and "vision" not in n]
            assert norms, "READNCE: 找不到语言模型final norm"
            return norms[-1]

        _anchor = os.environ.get("READNCE_ANCHOR_LAYER", "").strip()
        _danchor = os.environ.get("READNCE_DYN_ANCHOR_LAYER", "").strip()
        _nname, _nmod = _find_layer(_anchor, True)

        def _cap(_m, _i, out):
            # decoder块的输出是元组(隐状态, ...), final norm输出是裸张量, 两吃
            _h_holder["h"] = out[0] if isinstance(out, tuple) else out
            return out
        _nmod.register_forward_hook(_cap)
        if _danchor and _danchor != _anchor:
            _dname, _dmod = _find_layer(_danchor, True)

            def _capd(_m, _i, out):
                _h_holder["hd"] = out[0] if isinstance(out, tuple) else out
                return out
            _dmod.register_forward_hook(_capd)
            log(f"READNCE hooks: retro@{_nname} dyn@{_dname}")
        else:
            log(f"READNCE hidden hook on: {_nname} (retro+dyn共用)")

    # ---- 3. [S2-④] DeepSpeed接管:优化器/分片/累积全在zero3.json里 ----
    # 【为什么需要DeepSpeed】9B参数全量训练时, Adam优化器要为每个参数记
    # 一份动量+一份方差+fp32主本, 账本远比模型本身大, 单卡96G根本装不下。
    # ZeRO的思路: 4张卡各存账本的1/4, 用到谁的部分就临时通信取——空间换通信。
    ds_cfg = json.load(open(DS_CONFIG))
    ds_cfg["train_micro_batch_size_per_gpu"] = int(os.environ.get("PER_DEVICE_BS", "1"))
    ds_cfg["gradient_accumulation_steps"] = ACCUM
    ds_cfg["optimizer"]["params"]["lr"] = LR
    # WarmupDecayLR的峰值LR独立于optimizer.lr; 不同步改会把实际LR封顶在json旧值
    if "warmup_max_lr" in ds_cfg.get("scheduler", {}).get("params", {}):
        ds_cfg["scheduler"]["params"]["warmup_max_lr"] = LR
    ds_cfg["scheduler"]["params"]["total_num_steps"] = MAX_STEPS
    ds_cfg["scheduler"]["params"]["warmup_num_steps"] = max(1, MAX_STEPS // 33)
    # 08-06 分组LR: 新初始化模块(projector+READNCE头)无stage1热启动, 按惯例给5×;
    # WarmupDecayLR的warmup_max_lr吃列表(逐参数组), 否则每步会把组间差覆盖回同值。
    PROJ_LR = float(os.environ.get("PROJ_LR", str(LR * 5)))
    _fresh, _base = [], []
    for _n, _p in model.named_parameters():
        if not _p.requires_grad:
            continue
        (_fresh if ("state_projector" in _n or "readnce_" in _n) else _base).append(_p)
    log(f"分组LR: base={sum(p.numel() for p in _base)/1e6:.0f}M @ {LR} | "
        f"fresh(proj+readnce)={sum(p.numel() for p in _fresh)/1e6:.1f}M @ {PROJ_LR}")
    ds_cfg["scheduler"]["params"]["warmup_max_lr"] = [LR, PROJ_LR]
    engine, _, _, _ = deepspeed.initialize(
        model=model, config=ds_cfg,
        model_parameters=[{"params": _base, "lr": LR},
                          {"params": _fresh, "lr": PROJ_LR}])

    # ---- 3.5 断点续训:gpupr只有24h墙,5500步大概率要跨墙 ----
    # 注意:必须同样的GPU数resume(ZeRO分片按world size切,换卡数必崩)
    resume_step = 0
    tags = sorted((int(d.name[7:]) for d in os.scandir(OUT)
                   if d.is_dir() and d.name.startswith("ds_step")), reverse=True) \
        if os.path.isdir(OUT) else []
    # 08-11: OOM可把存档杀成半截尸体(10000残档实弹)——烂档跳过, 逐个回退到最近健康档
    for _tag in tags:
        try:
            path, _ = engine.load_checkpoint(OUT, tag=f"ds_step{_tag}")
        except Exception as _e:
            log(f"ds_step{_tag} 加载失败({type(_e).__name__}), 疑似半截档, 回退上一档")
            continue
        if path is not None:
            resume_step = _tag
            log(f"resumed from ds_step{_tag}")
            break
    # 有档在场却一个都加载不上 → 宁死不从零重训(静默清零=灾难, 08-11家法)
    if tags and resume_step == 0:
        raise RuntimeError(f"{len(tags)}个ds_step存档全部加载失败, 拒绝fresh start, 人工检查")
    # 08-13 竞态损伤复活方案: ds档全灭时, 允许从健康hf快照热启(QWEN_DIR指hf_stepN)
    # 并把步位/LR调度快进到位。一旦新ds档落地, tags非空→本分支自动失效(自解除)。
    _force = int(os.environ.get("RESUME_STEP_FORCE", "0"))
    if _force and resume_step == 0:
        resume_step = _force
        for _ in range(_force):
            engine.lr_scheduler.step()
        log(f"RESUME_STEP_FORCE={_force}: hf权重热启(优化器全新), LR调度已快进对位, "
            f"当前lr={engine.lr_scheduler.get_last_lr()}")

    # ---- 4. 数据(每个rank吃自己那份:按样本编号 % world) ----
    # states = 预先算好的TSSM状态字典 {"r2r:集号-步号": 3072维向量}。
    # 训练时不现跑TSSM(太慢), 直接查表——这就是"state缓存"。
    # 数据分片: 0号卡拿第0,4,8...条, 1号卡拿第1,5,9...条, 保证不重不漏。
    states = torch.load(STATE_CACHE, map_location="cpu")
    if ANN_RXR:
        # 双源/三源: state缓存须为 "r2r:ep"/"rxr:ep" 键口径
        srcs = [(ANN, SQLITE, "r2r"), (ANN_RXR, SQLITE_RXR, "rxr")]
        if ANN_RXR2:
            srcs.append((ANN_RXR2, SQLITE_RXR2, "rxr"))   # 扩容集, 同域前缀
        ds = NavDataset(None, states, sources=srcs)
    else:
        ds = NavDataset(ANN, states)
    my_idx = list(range(rank, len(ds), world))          # rank间无重叠
    BS = int(os.environ.get("PER_DEVICE_BS", "1"))
    dl = DataLoader(torch.utils.data.Subset(ds, my_idx), batch_size=BS,
                    shuffle=True, num_workers=2, collate_fn=lambda b: b,
                    drop_last=True)

    def build_batch(ss):
        """多样本拼批 (07-31 打满手术): 逐条造prompt, processor批量padding;
        labels按每行prompt长度掩码, padding位一并-100。

        【拼批在做什么】把几条样本拼成模型吃的一锅:
          1. 每条造出完整对话文本: 系统提示 + <memory>16个占位符</memory>
             + 当前画面 + 指令 + 标准答案(如 "Move forward 25 cm")。
          2. processor把文本变token id、图片变像素张量, 短句尾部补padding对齐。
          3. labels = 监督信号: 只有"答案"那几个token参与算损失; prompt部分
             和padding部分全部填-100(-100是pytorch约定的"此位不计损失")。
             ——模型只为"下一步动作说什么"挨打, 不为复读题目挨打。"""
        # 08-18 无记忆消融(用户令): prompt整体去记忆——不留<memory>块, 系统提示也换无记忆版
        if NO_MEMORY:
            mem_span = ""
            _sys = ("You are a robot navigating an indoor environment. Use the current "
                    "observation and the instruction to decide your next action.")
        else:
            mem_span = "<memory>" + ph * cfg["n_mem"] + "</memory>"
            _sys = SYS
        fulls, prompts, imgs = [], [], []
        for s in ss:
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": _sys}]},
                {"role": "user", "content": [
                    {"type": "text", "text": mem_span + "\nCurrent observation: "},
                    {"type": "image"},
                    {"type": "text",
                     "text": f"\nInstruction: \"{s['q'].strip()}\"\nWhat is your next action?"}]},
            ]
            prompt = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompts.append(prompt)
            fulls.append(prompt + s["a"].strip() + tok.eos_token)
            imgs.append(s["img"])
        ef = processor(text=fulls, images=imgs, return_tensors="pt", padding=True)
        ep_ = processor(text=prompts, images=imgs, return_tensors="pt", padding=True)
        labels = ef["input_ids"].clone()
        plen = ep_["attention_mask"].sum(dim=1)            # 每行prompt真实长度
        for bi in range(labels.shape[0]):
            labels[bi, : int(plen[bi])] = -100
        labels[ef["attention_mask"] == 0] = -100           # padding位不监督
        ef["labels"] = labels
        return {k: v.to(engine.device) for k, v in ef.items()}

    # ---- 5. 训练循环 ----
    # 【micro与step】显存只装得下每卡2条样本(micro batch), 但我们想要64条
    # 的稳定梯度 → 攒: 每卡前向反传8次只攒梯度不更新(micro+=1), 攒满
    # ACCUM次engine.step()才真正更新一次参数(step+=1)。
    # 2条/卡 × 4卡 × 8次累积 = 全局批64。日志和存档都按step计。
    step = resume_step
    micro = 0
    run_loss = 0.0
    rn_stat = {"rce": 0.0, "rn": 0, "dce": 0.0, "dn": 0, "rhit": 0.0, "dhit": 0.0}
    t0 = time.time()
    while step < MAX_STEPS:
        for ss in dl:
            batch = build_batch(ss)
            if batch["input_ids"].shape[1] > 4096:
                continue
            if NO_MEMORY:
                holder["mem"] = None                    # 消融:便条空白
            else:
                # dtype修(07-20): bf16引擎里投影器参数是bf16, 输入必须同型。
                # 批量: [B,3072] → [B,n_mem,d], 钩子按行主序回填B×n_mem个占位符
                _st = torch.stack([s["state"] for s in ss])
                holder["mem"] = engine.module.state_projector(
                    _st.to(engine.device, dtype=torch.bfloat16))
                # 安全断言: 占位符数必须=B*n_mem(防pad token与占位符撞id的静默错位)
                _npl = int((batch["input_ids"] == ph_id).sum())
                assert _npl == len(ss) * cfg["n_mem"], \
                    f"placeholder数{_npl} != {len(ss)}x{cfg['n_mem']}"
            loss = engine(**batch).loss
            # ---- READNCE (07-30, rnceans同构): 答案预测位隐状态池化→查冻结钥匙空间。
            # retro头认t-k帧(无动作), dyn头给动作窗认t+k帧。0*Σp恒挂图防ZeRO桶错位。
            # 【考试流程人话版】对批里每条样本:
            #   1. 从钩子抄来的隐状态里, 取"即将说出第一个答案词"那个位置的向量
            #      (那一刻=决策瞬间, 而且看不到答案, 无作弊可能);
            #   2. retro考: 向量+滞后档标签 → 小头 → 和该集全部帧指纹算相似度
            #      → 交叉熵逼"t-k那帧"得分最高(同集其他帧都是干扰项);
            #   3. dyn考: 同上, 但额外拼上未来动作串的嵌入(保序), 目标换成t+k帧;
            #   4. 两门考试的平均罚分乘小权重(0.2)加进总损失。
            if READNCE:
                # 【0×Σp是什么黑魔法】某些批次可能一场考试都考不成(比如样本
                # 都在集子开头, 没有"3步前"可回忆), 这时考官参数没梯度。但
                # DeepSpeed要求所有可训参数每步都出现在计算图里, 否则梯度桶
                # 对不齐直接崩。0.0×(所有考官参数之和)= 数值上加了个0,
                # 但让每个参数都"挂"在图上, 拿到一份合法的零梯度。
                _zs = None
                for _mod in (engine.module.readnce_q_head, engine.module.readnce_dyn_head,
                             engine.module.readnce_lag_emb, engine.module.readnce_dyn_lag_emb,
                             engine.module.readnce_act_emb):
                    for _p in _mod.parameters():
                        _zs = _p.sum() if _zs is None else _zs + _p.sum()
                loss = loss + 0.0 * _zs.to(loss.dtype)
                h = _h_holder["h"]; _h_holder["h"] = None
                hd = _h_holder["hd"]; _h_holder["hd"] = None
                if hd is None:
                    hd = h                     # 未设双锚时dyn与retro共用考卷
                _r_all, _d_all = [], []
                if h is not None:
                    for bi, s in enumerate(ss):
                        K = readnce_keys.get(s["ep"])
                        t = int(s["step"]); T = 0 if K is None else K.shape[0]
                        if K is None or t >= T:
                            continue
                        lab = batch["labels"][bi]
                        ans = (lab != -100).nonzero(as_tuple=True)[0]
                        if ans.numel() == 0:
                            continue
                        # 查询锚点 (07-31 GPT审计): teacher forcing下第2+答案位的隐状态
                        # 看得见真实动作前缀(泄漏捷径, 泄压逼读) → 默认只取首答案token
                        # 的预测位(=prompt末位, 零泄漏, 且正是推理时的决策位)。
                        if os.environ.get("READNCE_QANCHOR", "first") == "first":
                            pred_pos = (ans[:1] - 1).clamp(min=0)
                        else:                                # all: 冠军口径(rncenp实测夺冠)
                            pred_pos = (ans - 1).clamp(min=0)
                        # 逐位置查询 (08-01 用户令, 对齐rncenp冠军配方): 不池化,
                        # 每个答案位各自为查询, CE按位置平均, 命中按多数位置计
                        Hq = h[bi, pred_pos, :]              # [P, d]  retro考卷(第11层)
                        Hqd = hd[bi, pred_pos, :]            # [P, d]  dyn考卷(第23层)
                        Kd = K.to(h.device, torch.float32)

                        def _rn_ce_multi(Hx, head, extras, tgt):
                            P = Hx.shape[0]
                            ex = [e.to(Hx.dtype).unsqueeze(0).expand(P, -1) for e in extras]
                            q = F.normalize(head(torch.cat([Hx] + ex, dim=-1)).float(), dim=-1)
                            lg = (q @ Kd.T) / READNCE_TAU    # [P, T]
                            ce = F.cross_entropy(lg, torch.full((P,), tgt,
                                                 device=h.device, dtype=torch.long))
                            hit = float((lg.argmax(-1) == tgt).float().mean() >= 0.5)
                            return ce, hit

                        for li, k in enumerate(READNCE_RETRO_LAGS):
                            if t - k < 0:
                                continue
                            le = engine.module.readnce_lag_emb(
                                torch.tensor(li, device=h.device))
                            _r_all.append(_rn_ce_multi(
                                Hq, engine.module.readnce_q_head, [le], t - k))
                        acts = readnce_acts.get(s["ep"])
                        if acts is None:
                            continue
                        for li, k in enumerate(READNCE_DYN_LAGS):
                            if t + k >= T or t + k + 1 > len(acts):
                                continue
                            win = acts[t + 1: t + k + 1].long().to(h.device)
                            if win.numel() == 0:
                                continue
                            _kmax = max(READNCE_DYN_LAGS)
                            _slots = torch.zeros(_kmax, dtype=torch.long, device=h.device)
                            _slots[: win.numel()] = win
                            ae = engine.module.readnce_act_emb(_slots).flatten()  # 保序
                            dle = engine.module.readnce_dyn_lag_emb(
                                torch.tensor(li, device=h.device))
                            _d_all.append(_rn_ce_multi(
                                Hqd, engine.module.readnce_dyn_head, [dle, ae], t + k))
                if _r_all:
                    loss = loss + (READNCE_W * torch.stack([x for x, _ in _r_all]).mean()).to(loss.dtype)
                if _d_all:
                    loss = loss + (READNCE_DYN_W * torch.stack([x for x, _ in _d_all]).mean()).to(loss.dtype)
                rn_stat["rce"] += sum(float(x) for x, _ in _r_all); rn_stat["rn"] += len(_r_all)
                rn_stat["dce"] += sum(float(x) for x, _ in _d_all); rn_stat["dn"] += len(_d_all)
                rn_stat["rhit"] += sum(hh for _, hh in _r_all)
                rn_stat["dhit"] += sum(hh for _, hh in _d_all)
            engine.backward(loss)                       # ZeRO-3的梯度回传
            engine.step()                               # 累积够了才真更新
            holder["mem"] = None
            run_loss += loss.item()
            micro += 1
            if micro % ACCUM == 0:
                step += 1
                if step % 10 == 0 and rank == 0:
                    _rnmsg = ""
                    if READNCE and (rn_stat["rn"] or rn_stat["dn"]):
                        _rnmsg = (f" | rnce r_ce={rn_stat['rce']/max(1,rn_stat['rn']):.3f}"
                                  f" acc={rn_stat['rhit']/max(1,rn_stat['rn']):.2f}({rn_stat['rn']})"
                                  f" d_ce={rn_stat['dce']/max(1,rn_stat['dn']):.3f}"
                                  f" acc={rn_stat['dhit']/max(1,rn_stat['dn']):.2f}({rn_stat['dn']})")
                        rn_stat.update({"rce": 0.0, "rn": 0, "dce": 0.0, "dn": 0,
                                        "rhit": 0.0, "dhit": 0.0})
                    log(f"step {step}/{MAX_STEPS} loss={run_loss/(10*ACCUM):.4f} "
                        f"{(time.time()-t0)/max(step-resume_step,1):.1f}s/step "
                        f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G" + _rnmsg)
                    run_loss = 0.0
                if step % SAVE_STEPS == 0 or step >= MAX_STEPS:
                    # 存两份:①DeepSpeed原生(能续训,分片) ②16bit整模型(能直接eval)
                    # ①含优化器账本, 巨大(9B约百G), 由修剪器只保最近两份;
                    # ②hf_stepN目录=普通HuggingFace模型+projector.pt, 冒烟保姆
                    #   看到它落地就自动发评测。
                    engine.save_checkpoint(OUT, tag=f"ds_step{step}")
                    engine.save_16bit_model(f"{OUT}/hf_step{step}")
                    if rank == 0:
                        processor.save_pretrained(f"{OUT}/hf_step{step}")
                        torch.save({"proj": {k: v.cpu() for k, v in
                                             engine.module.state_projector.state_dict().items()},
                                    "cfg": cfg}, f"{OUT}/hf_step{step}/projector.pt")
                        log(f"saved step{step} (ds分片 + hf整模型)")
                if step >= MAX_STEPS:
                    break
    log("DONE")


if __name__ == "__main__":
    main()

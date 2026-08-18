#!/usr/bin/env python3
# ============================================================
# 评测桥·客户端 —— habitat 里开着模拟器,问对面的 Qwen 怎么走
# ============================================================
# 跑在旧环境(navila-eval,有habitat);Qwen在隔壁进程(qwen_server.py)。
# 每一步干四件事:
#   1) 拿当前相机画面
#   2) 喂给在线TSSM,滚出记忆state(和NaVILA线同一个世界模型)
#   3) 把 画面+指令+state 发给Qwen服务,拿回动作文本
#   4) 把"前进75cm"翻译成模拟器按键,执行
# 结果落盘成 NaVILA 同款 json,decompose_smoke.py 直接能算分。
#
# 用法(见 scripts/eval_bridge.sbatch,一般不手动跑):
#   SERVER=http://127.0.0.1:7860 TAG=qwen_stage1 SMOKE_PER_SCENE=3 \
#     python eval_qwen_vln.py
# ============================================================
import base64
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np
import torch
from PIL import Image

SERVER = os.environ.get("SERVER", "http://127.0.0.1:7860")
TAG = os.environ.get("TAG", "qwen_bridge_test")
OUT_DIR = os.environ.get("OUT_DIR",
                         f"/cluster/project/rsl/ruiqzhang/codes/qwen35-vln/eval_out/{TAG}")
MAX_STEPS_PER_EP = int(os.environ.get("MAX_STEPS_PER_EP", "195"))
TSSM_CKPT = os.environ.get("TSSM_ONLINE_CKPT",
                           "/cluster/scratch/ruiqzhang/tssm_v5_lm2p9_kb09_act01_pose01_free10.pt")
SIGLIP_DIR = os.environ.get("TSSM_ONLINE_SIGLIP",
                            "/cluster/project/rsl/ruiqzhang/codes/NaVILA/navila_pretrain/vision_tower")

NAVILA = "/cluster/project/rsl/ruiqzhang/codes/NaVILA"
sys.path.insert(0, f"{NAVILA}/RSSM_v4")


# ---------- 零件1:在线TSSM(照抄NaVILA线step_online的v5路径) ----------
class OnlineTSSM:
    """当前画面流进来,信念state滚出去。每集开始要 reset()。"""

    def __init__(self, device="cuda"):
        from navigation_tssm_feat import NavigationTSSM, TSSMConfig
        from transformers import SiglipVisionModel, SiglipImageProcessor
        blob = torch.load(TSSM_CKPT, map_location="cpu", weights_only=False)
        cfg = TSSMConfig(**{k: v for k, v in blob["config"].items()
                            if k in TSSMConfig.__dataclass_fields__})
        self.tssm = NavigationTSSM(cfg)
        self.tssm.load_state_dict(blob["model"])
        self.tssm.eval().to(device)
        self.feat_dim = int(cfg.feat_dim)          # v5=10368(9块×1152); v8retro=1152(pool)
        self.pooled = (self.feat_dim == 1152)      # siglip2_pool 口径(stage2全量线)
        # 双塔口径 (08-03, full8): 维度10368与lm2p9撞车, 必须显式env选择
        self.dualtower = (os.environ.get("TSSM_ONLINE_FEAT", "").strip().lower()
                          == "dualtower_p9")
        if not self.pooled:
            g = int(round((self.feat_dim // 1152) ** 0.5))
            self.g, self.blk = g, 27 // g
        self.sig = SiglipVisionModel.from_pretrained(SIGLIP_DIR).eval().to(device)
        ip = SiglipImageProcessor.from_pretrained(SIGLIP_DIR)
        self.mean = torch.tensor(ip.image_mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(ip.image_std).view(1, 3, 1, 1).to(device)
        self.size = ip.size["height"]
        if self.dualtower:
            # DA-V2-S骨干经Dinov2前缀剥离加载(tf4.37无DepthAnything类; 对拍cos=0.99999已验)
            import json as _json
            from transformers import Dinov2Config, Dinov2Model
            from safetensors.torch import load_file as _lf
            da_path = os.environ.get(
                "TSSM_ONLINE_DA",
                "/cluster/scratch/ruiqzhang/hf_cache/hub/"
                "models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/"
                "5426e4f0f36572d16453bbda7a8389317b1bef99")
            with open(os.path.join(da_path, "config.json")) as f:
                bb = _json.load(f)["backbone_config"]
            dcfg = Dinov2Config(**{k: v for k, v in bb.items()
                                   if k in Dinov2Config().to_dict()})
            self.da = Dinov2Model(dcfg)
            sd = _lf(os.path.join(da_path, "model.safetensors"))
            self.da.load_state_dict({k[len("backbone."):]: v for k, v in sd.items()
                                     if k.startswith("backbone.")}, strict=False)
            self.da.eval().to(device)
            self.da_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            self.da_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.dev = device
        self.state = None
        print(f"[tssm] ready (feat_dim={self.feat_dim}, dualtower={self.dualtower})", flush=True)

    def reset(self):
        self.state = None

    @torch.no_grad()
    def step(self, rgb_np, last_action: int):
        img = Image.fromarray(np.uint8(rgb_np)).convert("RGB")
        if self.dualtower:
            # 与precompute_dualtower.py逐位一致: 384 BILINEAR/(x-.5)/.5 ‖ 378+ImageNet
            import torch.nn.functional as F
            rgb = img.resize((384, 384), Image.BILINEAR)
            x = (torch.from_numpy(np.asarray(rgb).copy()).float()
                 .permute(2, 0, 1).unsqueeze(0).to(self.dev) / 255.0)
            s_out = self.sig(pixel_values=(x - 0.5) / 0.5).last_hidden_state   # [1,576,768]
            s_grid = (s_out.view(1, 24, 24, 768).view(1, 3, 8, 3, 8, 768)
                      .mean(dim=(2, 4)).reshape(1, 9, 768))
            xd = F.interpolate(x, size=(378, 378), mode="bilinear", align_corners=False)
            hs = self.da(pixel_values=(xd - self.da_mean) / self.da_std).last_hidden_state[:, 1:]
            d_grid = (hs.view(1, 27, 27, 384).view(1, 3, 9, 3, 9, 384)
                      .mean(dim=(2, 4)).reshape(1, 9, 384))
            feat = torch.cat([s_grid, d_grid], dim=-1).reshape(1, 9 * 1152).float()
            if self.state is None:
                self.state = self.tssm.get_initial_state(1, self.dev, dtype=torch.float32)
                a = torch.zeros(1, dtype=torch.long, device=self.dev)
            else:
                a = torch.tensor([last_action], dtype=torch.long, device=self.dev)
            self.state = self.tssm.inference_step(self.state, a, feat)
            h = self.state.h if hasattr(self.state, "h") else self.state[0]
            z = self.state.z if hasattr(self.state, "z") else self.state[1]
            return torch.cat([h.flatten(), z.flatten()])[:3072].cpu().tolist()
        rgb = img.resize((self.size, self.size), Image.BILINEAR)   # 两口径都是BILINEAR
        arr = torch.from_numpy(np.asarray(rgb)).float().permute(2, 0, 1) / 255.0
        if self.pooled:
            # siglip2_pool 口径, 与 rssm_memory_adapter 在线路径逐位一致: (x-0.5)/0.5 + MAP头
            arr = ((arr.unsqueeze(0).to(self.dev)) - 0.5) / 0.5
            feat = self.sig(pixel_values=arr).pooler_output.float()  # [1,1152]
        else:
            arr = ((arr.unsqueeze(0).to(self.dev)) - self.mean) / self.std
            hs = self.sig(pixel_values=arr, output_hidden_states=True).hidden_states[-2]
            feat = (hs.view(1, 27, 27, 1152)                        # 27×27patch
                      .view(1, self.g, self.blk, self.g, self.blk, 1152)
                      .mean(dim=(2, 4))                             # 3×3块平均
                      .reshape(1, self.feat_dim)).float()
        if self.state is None:
            self.state = self.tssm.get_initial_state(1, self.dev, dtype=torch.float32)
            a = torch.zeros(1, dtype=torch.long, device=self.dev)
        else:
            a = torch.tensor([last_action], dtype=torch.long, device=self.dev)
        self.state = self.tssm.inference_step(self.state, a, feat)
        # 拼成LoRA线同款3072向量: h(2048) ‖ z(1024)
        h = self.state.h if hasattr(self.state, "h") else self.state[0]
        z = self.state.z if hasattr(self.state, "z") else self.state[1]
        return torch.cat([h.flatten(), z.flatten()])[:3072].cpu().tolist()


# ---------- 零件2:动作文本 → 模拟器按键 ----------
# 模拟器一次只走25cm/转15°,所以"前进75cm"=按3下前进。
def parse_action(text: str):
    t = text.lower()
    import re
    if "stop" in t:
        return [0]
    m = re.search(r"forward\s+(\d+)", t)
    if m:
        return [1] * max(1, round(int(m.group(1)) / 25))
    m = re.search(r"(left|right)\s+(\d+)", t)
    if m:
        key = 2 if m.group(1) == "left" else 3
        return [key] * max(1, round(int(m.group(2)) / 15))
    return [1]                                    # 看不懂就往前挪一步,别卡死


# ---------- 零件3:问Qwen ----------
def ask_qwen(rgb_np, instruction, state):
    buf = io.BytesIO()
    Image.fromarray(np.uint8(rgb_np)).convert("RGB").save(buf, format="JPEG", quality=90)
    req = urllib.request.Request(
        f"{SERVER}/act", method="POST",
        data=json.dumps({"image_b64": base64.b64encode(buf.getvalue()).decode(),
                         "instruction": instruction, "state": state}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["action"]


# ---------- 主流程:标准VLN-CE评测循环 ----------
def main():
    os.chdir(f"{NAVILA}/evaluation")
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
    from habitat_baselines.common.environments import get_env_class

    config = get_config("vlnce_baselines/config/r2r_baselines/navila.yaml", [])
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    # 独立脚本没走trainer,得自己把EVAL.SPLIT抄进TASK_CONFIG(否则默认train split,数据不存在)
    split = config.EVAL.SPLIT
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.DATASET.ROLES = ["guide"]
    config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    # task.py的分片补丁字段(trainer里由num_chunks/chunk_idx抄入);评测桥不分片
    config.TASK_CONFIG.DATASET.NUM_CHUNKS = 1
    config.TASK_CONFIG.DATASET.CHUNK_IDX = 0
    # 冒烟口径: SCENES="s1,s2" 限场景; PER_SCENE=N 每场景配额(超额episode快速跳过)
    _sc = os.environ.get("SCENES", "").strip()
    if _sc:
        config.TASK_CONFIG.DATASET.CONTENT_SCENES = _sc.split(",")
    config.freeze()
    per_scene = int(os.environ.get("PER_SCENE", "0"))       # 0 = 不限
    scene_count = {}
    envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
    _nomem = os.environ.get("NO_MEMORY", "0") == "1"
    tssm = None if _nomem else OnlineTSSM()   # 消融: 不装在线塔
    os.makedirs(OUT_DIR, exist_ok=True)

    num_eps = sum(envs.number_of_episodes)
    print(f"[eval] episodes to run: {num_eps}", flush=True)
    results = {}
    _lat_tssm, _lat_llm, _lat_prims = [], [], []
    obs = envs.reset()

    skipped = 0
    while len(results) + skipped < num_eps:
        ep = envs.current_episodes()[0]
        sc = str(getattr(ep, "scene_id", "?")).split("/")[-1].split(".")[0]
        if per_scene and scene_count.get(sc, 0) >= per_scene:
            skipped += 1                                  # 配额满, 快速跳过
            if len(results) + skipped < num_eps:
                obs = envs.reset_at(0)
            continue
        scene_count[sc] = scene_count.get(sc, 0) + 1
        instruction = ep.instruction.instruction_text
        if tssm is not None:
            tssm.reset()
        last_sem_action = 0        # TSSM 词表(0-9 语义动作), 非 habitat 原语号!
        steps = 0
        queue = []
        done = False
        while not done and steps < MAX_STEPS_PER_EP:
            rgb = obs[0]["rgb"]
            if not queue:                                # 队列空了才问Qwen
                _t0 = time.time()
                state = None if tssm is None else tssm.step(rgb, last_sem_action)
                _t1 = time.time()
                text = ask_qwen(rgb, instruction, state)
                _t2 = time.time()
                queue = parse_action(text)
                # 08-12 用户令: 一步推理延迟账(tssm编码 / LLM往返 / 本次决策覆盖几个原语)
                _lat_tssm.append(_t1 - _t0)
                _lat_llm.append(_t2 - _t1)
                _lat_prims.append(max(1, len(queue)))
                # 训练同源词表(data_navila.ACTION_VOCAB); 解析失败按 forward 25 处置
                try:
                    from data_navila import parse_action as _sem_parse
                    last_sem_action = _sem_parse(text)
                except Exception:
                    last_sem_action = 1
            act = queue.pop(0)
            outputs = envs.step([act])
            obs = [o for (o, r, d, i) in [outputs[0]]] if isinstance(outputs[0], tuple) else outputs[0]
            # habitat VectorEnv.step 返回 [(obs, reward, done, info)]
            o, r, d, info = outputs[0]
            obs = [o]
            # (语义动作已在问询时记录; habitat原语号不喂TSSM)
            steps += 1
            done = d or act == 0
            if done:
                results[str(ep.episode_id)] = {
                    "distance_to_goal": info["distance_to_goal"],
                    "success": info["success"],
                    "spl": info["spl"],
                    "ndtw": info.get("ndtw", 0.0),
                    "path_length": info.get("path_length", 0.0),
                    "oracle_success": info.get("oracle_success", 0.0),
                    "steps_taken": steps,
                }
                print(f"[{len(results)}/{num_eps}] ep{ep.episode_id} "
                      f"SR={info['success']:.0f} steps={steps}", flush=True)
                if len(results) + skipped < num_eps:
                    obs = envs.reset_at(0)               # 换下一集
        # 超步数没done:强制记0分
        if not done:
            results[str(ep.episode_id)] = {
                "distance_to_goal": 99.0, "success": 0.0, "spl": 0.0,
                "ndtw": 0.0, "path_length": 0.0, "oracle_success": 0.0,
                "steps_taken": steps}
            if len(results) + skipped < num_eps:
                obs = envs.reset_at(0)

    out = f"{OUT_DIR}/val_unseen_1-0.json"
    json.dump(results, open(out, "w"))
    n = len(results)
    sr = 100 * sum(v["success"] for v in results.values()) / n
    print(f"\n===== {TAG}: n={n} SR={sr:.1f} -> {out} =====", flush=True)
    if _lat_llm:
        _a = np.array(_lat_tssm) * 1000
        _b = np.array(_lat_llm) * 1000
        _q = (_a + _b)
        _pp = _q / np.array(_lat_prims)          # 折算到每个25cm/15°原语
        print(f"[bench] 决策次数={len(_b)} | tssm编码 {_a.mean():.0f}ms | "
              f"LLM往返 {_b.mean():.0f}ms (p50 {np.percentile(_b,50):.0f} / p95 {np.percentile(_b,95):.0f}) | "
              f"一步决策合计 {_q.mean():.0f}ms | 折算每原语 {_pp.mean():.0f}ms | "
              f"平均每决策原语数 {np.mean(_lat_prims):.2f}", flush=True)


if __name__ == "__main__":
    main()

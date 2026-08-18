#!/usr/bin/env python3
# ============================================================
# 评测桥·服务端 —— 把 Qwen 变成一个"问了就答"的本地网站
# ============================================================
# 为什么要拆两个进程?
#   habitat 模拟器住在旧环境(navila-eval),Qwen3.5 需要新 transformers
#   (vllm环境),两者装不进同一个 Python → 各开一个进程,用 localhost
#   HTTP 说话。本文件 = Qwen 那半边(vllm 环境跑)。
#
# 接口就一个:
#   POST /act   收: {image_b64, instruction, state(3072个float数组,可空)}
#               回: {"action": "The next action is move forward 75 cm."}
#   GET  /health 活着就回 ok(客户端用它等启动)
#
# 启动(评测sbatch会自动拉起,不用手动):
#   MODEL=... PROJ=... PORT=7860 python qwen_server.py
# ============================================================
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image

MODEL_DIR = os.environ.get("MODEL", "/cluster/scratch/ruiqzhang/models/Qwen3.5-9B")
PROJ_PATH = os.environ.get("PROJ", "/cluster/scratch/ruiqzhang/qwen35_stage1/projector_step4000.pt")
PORT = int(os.environ.get("PORT", "7860"))
# Stage-2 的 hf_stepXXXX 目录整模型:MODEL 指向它、PROJ 指向它里面的 projector.pt 即可

from train_stage1 import SYS, StateProjector  # noqa: E402


def load():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    blob = torch.load(PROJ_PATH, map_location="cpu")
    cfg = blob["cfg"]
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, max_pixels=cfg.get("max_pixels", 448 * 448))  # stage2 cfg无此键, 用训练时默认
    tok = processor.tokenizer
    ph = cfg["placeholder"]
    ph_id = tok.encode(ph, add_special_tokens=False)[0]
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True, device_map="cuda")   # 跳过CPU随机初始化+直接落卡, 启动10min→~2min
    model.eval().requires_grad_(False)
    proj = StateProjector(n_tok=cfg["n_mem"], d_model=cfg["d_model"],
                          d_slice=cfg["d_slice"], hidden=cfg["hidden"]).cuda().float()
    proj.load_state_dict(blob["proj"])
    proj.eval()

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
    print(f"[server] 模型就绪: {MODEL_DIR}", flush=True)
    return processor, tok, model, proj, cfg, ph, holder


PROCESSOR, TOK, MODEL, PROJ, CFG, PH, HOLDER = load()


NO_MEMORY = os.environ.get("NO_MEMORY", "0") == "1"   # 08-18 无记忆消融评测口径
SYS_NOMEM = ("You are a robot navigating an indoor environment. Use the current "
             "observation and the instruction to decide your next action.")


@torch.no_grad()
def decide(image_b64: str, instruction: str, state):
    import time as _t; _t0 = _t.time()
    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    # 与train_stage2无记忆分支逐字同款: 无<memory>块+无记忆系统提示
    mem_span = "" if NO_MEMORY else ("<memory>" + PH * CFG["n_mem"] + "</memory>")
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYS_NOMEM if NO_MEMORY else SYS}]},
        {"role": "user", "content": [
            {"type": "text", "text": mem_span + "\nCurrent observation: "},
            {"type": "image"},
            {"type": "text",
             "text": f"\nInstruction: \"{instruction.strip()}\"\nWhat is your next action?"}]},
    ]
    prompt = PROCESSOR.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = PROCESSOR(text=[prompt], images=[img], return_tensors="pt")
    enc = {k: v.cuda() for k, v in enc.items()}
    if NO_MEMORY:
        HOLDER["mem"] = None                     # 消融: 无占位符可注入
    elif state is not None:
        HOLDER["mem"] = PROJ(torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0))
    out = MODEL.generate(**enc,
                         max_new_tokens=int(os.environ.get("GEN_TOKENS", "12")),
                         do_sample=False,
                         pad_token_id=TOK.eos_token_id)
    HOLDER["mem"] = None
    text = TOK.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[timing] {_t.time()-_t0:.2f}s prompt={enc['input_ids'].shape[1]} -> {text.strip()[:40]!r}", flush=True)
    return text.strip()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):                       # 关掉每请求一行的噪音日志
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {})

    def do_POST(self):
        if self.path != "/act":
            self._send(404, {})
            return
        n = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(n))
        try:
            action = decide(req["image_b64"], req["instruction"], req.get("state"))
            print(f"[act] {action}", flush=True)   # 桥测时肉眼审模型吐的动作文本
            self._send(200, {"action": action})
        except Exception as e:                        # 单集失败不拖死整场评测
            self._send(500, {"error": str(e)})


if __name__ == "__main__":
    print(f"[server] listening on 127.0.0.1:{PORT}", flush=True)
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()

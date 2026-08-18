#!/usr/bin/env python3
"""Stage-1: Qwen3.5-9B (FROZEN) + trainable StateProjector.

TSSM state (3072) -> N memory-token embeddings, injected at placeholder
positions via an embedding-layer forward hook (keeps the model's own
vision merge path fully intact). Loss = CE on the action answer only.

This is the projector-alignment stage of the qwen35-vln line (PLAN.md P1);
Stage-2 (full FT, ZeRO-3) reuses the same data/model wiring.

Data: NaVILA R2R annotations.json (video_id "ep-step", q, a, frames[]),
frames from train.sqlite (key -> JPEG blob), states from the per-dataset
v5 cache tssm_states_v5_lm2p9.pt (ep -> [T,3072]). All GT, nothing synthetic.
"""
import io
import json
import os
import random
import sqlite3
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MODEL_DIR = os.environ.get("QWEN_DIR", "/cluster/scratch/ruiqzhang/models/Qwen3.5-9B")
ANN = os.environ.get("ANN", "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/annotations.json")
SQLITE = os.environ.get("SQLITE", "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/train.sqlite")
STATE_CACHE = os.environ.get("STATE_CACHE", "/cluster/scratch/ruiqzhang/NaVILA-Dataset/R2R/tssm_states_v5_lm2p9.pt")
OUT = os.environ.get("OUT", "/cluster/scratch/ruiqzhang/qwen35_stage1")
N_MEM = int(os.environ.get("N_MEM", "256"))       # 用户令: 8*32,记忆要占足注意力版面
LR = float(os.environ.get("LR", "5e-4"))
ACCUM = int(os.environ.get("ACCUM", "16"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "4000"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "250"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))          # 0 = all
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(448 * 448)))  # cap vision tokens
SEED = 42

SYS = ("You are a robot navigating an indoor environment. A world-model memory "
       "summarizing everything you have seen so far is given between <memory> "
       "and </memory>. Use the memory, the current observation and the "
       "instruction to decide your next action.")


def pick_placeholder(tok):
    """One-token placeholder that is NOT part of the vision pipeline."""
    for c in ["<|fim_pad|>", "<|fim_prefix|>", "<|repo_name|>", "<|file_sep|>",
              "衢", "龘"]:
        try:
            ids = tok.encode(c, add_special_tokens=False)
        except Exception:
            continue
        if len(ids) == 1:
            return c, ids[0]
    raise RuntimeError("no single-token placeholder found")


class NavDataset(Dataset):
    """单源(旧接口, ann_path+全局SQLITE)或多源(sources=[(ann,sqlite,dom),...])。

    多源模式(07-30, 补RxR): state/钥匙/动作缓存统一用 "dom:ep" 键(ncev5混合缓存
    口径); 单源模式保持裸 ep 键, 老臂/老脚本行为逐位不变。
    """

    def __init__(self, ann_path, states, sources=None):
        self.samples = []
        self.states = states
        if sources is None:
            sources = [(ann_path, SQLITE, None)]
        self._sqlite_paths = [sq for _, sq, _ in sources]
        self._cons = [None] * len(sources)
        for si, (ann, _sq, dom) in enumerate(sources):
            raw = json.load(open(ann))
            skip = 0
            n0 = len(self.samples)
            for s in raw:
                ep, step = s["video_id"].rsplit("-", 1)
                key = f"{dom}:{ep}" if dom else ep
                st = states.get(key)
                if st is None or int(step) >= st.shape[0] or not s.get("frames"):
                    skip += 1
                    continue
                self.samples.append((si, key, int(step), s["q"], s["a"], s["frames"][-1]))
            print(f"dataset[{dom or 'single'}]: {len(self.samples)-n0} usable, "
                  f"{skip} skipped", flush=True)
            del raw
        random.Random(SEED).shuffle(self.samples)
        if MAX_SAMPLES:
            self.samples = self.samples[:MAX_SAMPLES]
        print(f"dataset total: {len(self.samples)}", flush=True)

    def __len__(self):
        return len(self.samples)

    def _sql(self, si=0):
        if self._cons[si] is None:                 # lazy per-worker connection
            self._cons[si] = sqlite3.connect(
                f"file:{self._sqlite_paths[si]}?mode=ro", uri=True,
                check_same_thread=False)
        return self._cons[si]

    def __getitem__(self, i):
        si, key, step, q, a, frame_key = self.samples[i]
        row = self._sql(si).execute(
            "SELECT image FROM frames WHERE key=?", (frame_key,)).fetchone()
        if row is None:
            return self[(i + 1) % len(self)]
        img = Image.open(io.BytesIO(row[0])).convert("RGB")
        return {"img": img, "q": q, "a": a, "ep": key, "step": step,
                "state": self.states[key][step].float()}


class StateProjector(nn.Module):
    """state -> N memory tokens, slice + SHARED up-projection.

    A naive Linear(hidden, N*4096) head explodes to ~4B params at N=256;
    instead each token gets a d_slice-dim slice of the state projection and
    a shared MLP lifts slices to d_model, plus a learned per-token embedding
    so tokens are distinguishable. ~55M params at N=256."""

    def __init__(self, state_dim=3072, n_tok=256, d_model=4096,
                 d_slice=64, hidden=1024):
        super().__init__()
        self.n_tok, self.d_model, self.d_slice = n_tok, d_model, d_slice
        self.slicer = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, n_tok * d_slice))
        self.up = nn.Sequential(
            nn.Linear(d_slice, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.tok_emb = nn.Parameter(torch.zeros(n_tok, d_model))

    def forward(self, s):                                    # [B,3072] fp32
        x = self.slicer(s).view(-1, self.n_tok, self.d_slice)
        return self.up(x) + self.tok_emb                      # [B,N,d_model]


def main():
    from transformers import AutoProcessor
    torch.manual_seed(SEED)
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, max_pixels=MAX_PIXELS)
    tok = processor.tokenizer
    ph, ph_id = pick_placeholder(tok)
    print(f"placeholder: {ph!r} id={ph_id}", flush=True)

    try:
        from transformers import AutoModelForImageTextToText as AM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AM
    model = AM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16,
                               attn_implementation="sdpa")
    model.to(dev)
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()                                   # frozen; dropout off

    proj = StateProjector(n_tok=N_MEM,
                          d_model=model.get_input_embeddings().embedding_dim
                          ).to(dev).float()
    print(f"trainable projector params: {sum(p.numel() for p in proj.parameters())/1e6:.1f}M",
          flush=True)

    # ---- memory injection: replace placeholder rows after embedding lookup ----
    holder = {"mem": None}
    emb_layer = model.get_input_embeddings()

    def hook(module, inputs, output):
        if holder["mem"] is None:
            return output
        ids = inputs[0]
        mask = ids == ph_id
        if not bool(mask.any()):
            return output
        out = output.clone()
        out[mask] = holder["mem"].reshape(-1, out.shape[-1]).to(out.dtype)
        return out

    emb_layer.register_forward_hook(hook)

    states = torch.load(STATE_CACHE, map_location="cpu")
    ds = NavDataset(ANN, states)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2,
                    collate_fn=lambda b: b[0])

    opt = torch.optim.AdamW(proj.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEPS)

    def build_batch(s):
        mem_span = "<memory>" + ph * N_MEM + "</memory>"
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [
                {"type": "text", "text": mem_span + "\nCurrent observation: "},
                {"type": "image"},
                {"type": "text",
                 "text": f"\nInstruction: \"{s['q'].strip()}\"\nWhat is your next action?"}]},
        ]
        prompt = processor.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
        full = prompt + s["a"].strip() + tok.eos_token
        ef = processor(text=[full], images=[s["img"]], return_tensors="pt")
        ep_ = processor(text=[prompt], images=[s["img"]], return_tensors="pt")
        labels = ef["input_ids"].clone()
        labels[:, : ep_["input_ids"].shape[1]] = -100
        labels[ef["input_ids"] == ph_id] = -100
        ef["labels"] = labels
        return {k: v.to(dev) for k, v in ef.items()}

    step = 0
    micro = 0
    t0 = time.time()
    run_loss = 0.0
    while step < MAX_STEPS:
        for s in dl:
            batch = build_batch(s)
            if batch["input_ids"].shape[1] > 4096:
                continue
            holder["mem"] = proj(s["state"].unsqueeze(0).to(dev))
            out = model(**batch)
            (out.loss / ACCUM).backward()
            holder["mem"] = None
            run_loss += out.loss.item()
            micro += 1
            if micro % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 10 == 0:
                    print(f"step {step}/{MAX_STEPS} loss={run_loss/ (10*ACCUM):.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} "
                          f"{(time.time()-t0)/max(step,1):.1f}s/step "
                          f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}G",
                          flush=True)
                    run_loss = 0.0
                if step % SAVE_STEPS == 0 or step == MAX_STEPS:
                    torch.save({"proj": proj.state_dict(),
                                "cfg": {"n_mem": N_MEM, "state_dim": 3072,
                                        "d_model": proj.d_model,
                                        "d_slice": proj.d_slice, "hidden": 1024,
                                        "placeholder": ph, "sys": SYS,
                                        "max_pixels": MAX_PIXELS}},
                               f"{OUT}/projector_step{step}.pt")
                    print(f"saved {OUT}/projector_step{step}.pt", flush=True)
                if step >= MAX_STEPS:
                    break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

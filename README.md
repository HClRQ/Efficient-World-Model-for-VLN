# Efficient World Model for VLN

Memory tokens from a compact world model (TSSM) replace multi-frame history in
Vision-and-Language Navigation, making a 9B VLM navigator both faster (~2.4x per
action vs. 8-frame baseline) and competitive in success rate.

## Layout

| dir | contents |
|---|---|
| `tssm/` | World-model core: TSSM (transformer-core state-space model) & RSSM (GRU-core) — model, trainer, launchers, feature extraction (SigLIP p9), state-cache / retrieval-key builders, and the LLM-side memory adapter |
| `probes/` | Science probes: seen-vs-unseen open-loop imagination (Dreamer-style), history retrieval exams, causal history-swap/shuffle interventions |
| `finetune_qwen/` | Full fine-tuning line: Qwen3.5-9B from public base + 16 memory tokens + deep supervision (READNCE dual mid-layer anchors), plus inference server & Habitat eval bridge |
| `finetune_navila_lora/` | LoRA line on NaVILA-8f: drop-in modified `llava_llama.py` / `llava_arch.py` (mid-layer anchored deep supervision, multi-slot history memory) and arm launch scripts incl. `hist8` (7 uniform history states + latest, NaVILA-mirrored sampling) |

## Key findings (probes/)

- **Memory generalizes, imagination does not**: history retrieval from the frozen
  state transfers to unseen scenes nearly losslessly (0.95-0.98 for the
  transformer core), while open-loop imagination collapses below a freeze-frame
  baseline in unseen scenes (n=1716, both cores, both towers).
- Causal interventions (history swap / shuffle) confirm the state is an ordered,
  content-determined archive rather than scene-prior guessing.

Paths in scripts reference our SLURM cluster layout; adapt `feat/state cache`
locations before use. Datasets (R2R / RxR / StreamVLN-processed RxR) and
checkpoints are not included.

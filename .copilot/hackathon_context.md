# Hackathon Context — Google DeepMind × InstaLILY On-Device AI Hackathon

## Event Details
- **Organizer:** INSTALILY.AI × Google DeepMind
- **Duration:** 10 AM – 6 PM (8 hours)
- **Team:** 2 developers + AI coding agents

---

## The Challenge

Build something that combines:
- ✅ A **fine-tuned on-device model** adapted to your task
- ✅ **Agentic behavior** — your system decides and acts autonomously
- ✅ **Visual input** — camera, video, or screen feeding into the system
- ✅ A **genuine reason** this runs on-device
- ⭐ **Optional bonus:** voice input and/or output

---

## Available Models

| Model | Size | Capabilities | Our Use |
|---|---|---|---|
| **Gemma 3** | 4B, 12B | Most capable, text + image reasoning | Scribe + Architect agents |
| **Gemma 3n** | E4B | On-device optimized, text + image + audio + video natively | Perceiver (Agent 1) |
| **Gemma 3 270M** | 270M | Tiny, fine-tunable | Potential fine-tune target |
| **FunctionGemma** | 270M | Designed for function calling | Analyst (fine-tuned for action decisions) |
| **PaliGemma 2** | 3B | Dedicated vision model (detection, segmentation, OCR) | Stretch: whiteboard OCR |

### Key Model Details
- **Gemma 3n E4B handles images AND video natively** with MobileNet-V5 encoder
- **Gemma 3n includes native audio** — 160ms chunks, voice + vision in single model without separate STT/TTS
- **Gemma 3 native function calling from 1B up** works well for agentic loops
- **FunctionGemma** — define API surface, fine-tune on examples → outputs right function call
  - Output format: `<start_function_call>call:func_name{param:<escape>value<escape>}<end_function_call>`
  - System prompt: `"You are a model that can do function calling with the following functions"`

---

## Fine-Tuning

### Frameworks (All Supported)
- **Hugging Face** (Transformers + TRL)
- **Unsloth** (fast LoRA)
- **Keras**
- **NeMo**
- **MLX** (mlx_lm.lora) ← WE'RE USING THIS

### Deployment Options (All Supported)
- **LM Studio**
- **Ollama**
- **llama.cpp**
- **LiteRT-LM**
- **MLX** ← WE'RE USING THIS

### Key Fact
> "LoRA fine-tuning on smaller Gemma models can finish in under an hour."

---

## What Judges Want

### Great Submission
- A real reason it runs on-device
- Components that work together as a **system** (not checklist)
- Fine-tuning that makes project **noticeably better** at its specific task
- **A live demo that actually runs**
- If voice is included, it feels natural and adds value

### What Impresses
- Projects where on-device **genuinely matters**
- Systems where **components connect to each other** rather than sitting side by side
- Ambition matters, but **a demo that runs** beats a complex idea explained with slides
- Voice is a bonus when it works well

### Being Realistic About 8 Hours
- Build a **focused prototype**, not a polished product
- Spend middle of day **wiring up agent loop and vision pipeline**
- Save last couple hours for **integration and testing with real inputs**
- "A focused prototype that works live will always be more compelling than something ambitious explained with slides"

---

## Documentation & Resources

| Resource | URL |
|---|---|
| Gemma models & docs | ai.google.dev/gemma |
| All models on HuggingFace | huggingface.co/google |
| Gemma Cookbook | github.com/google-gemini/gemma-cookbook |
| Google AI Edge / LiteRT-LM | github.com/google-ai-edge/LiteRT-LM |
| Fine-tuning guide | ai.google.dev/gemma/docs/tune |
| FunctionGemma | huggingface.co/google/functiongemma-270m-it |
| Gemma 3n developer guide | developers.googleblog.com/en/introducing-gemma-3n-developer-guide |
| Google AI Studio | aistudio.google.com |

---

## GPU VM Credentials

| Field | Value |
|---|---|
| IP | 34.30.177.139 |
| User | hackathon |
| Password | 23973796 |
| VM Name | hackathon-vm-hack-team03 |
| SSH | `ssh hackathon@34.30.177.139` |

### VM Specs (Verified)
- **GPU:** NVIDIA RTX PRO 6000 Blackwell — **98GB VRAM**
- **RAM:** 176GB
- **Disk:** 170GB available
- **CUDA:** 12.8
- **Python:** 3.10
- **⚠️ No internet access** (DNS can't resolve external hosts)
- **⚠️ No sudo access** (password required)

---

## Example Projects From Organizers

1. **Surgical instrument tracker** — Camera watches procedure tray, flags missing/out-of-sequence instruments. Agency = procedure-aware state machine.
2. **Industrial anomaly detector** — Camera monitors robotic arm, spots drift/defects in real-time. Agency = continuous monitor-assess-act loop.
3. **Workspace inventory tracker** — Camera watches physical space, tracks items, maintains running inventory. Agency = persistent model of what should be where.

# MeetMind — Complete Task Plan

## Team Roles
- **You (Dev 1):** Agent 2 (Scribe + Analyst + Architect), Fine-tuning, Dashboard
- **Friend (Dev 2):** Agent 1 (Perceiver — camera + mic), Integration wiring

---

## 🟢 DONE

### Agent 2 Core (Dev 1)
- [x] Project structure (`agents/` + `core/` + `main.py`)
- [x] Data contracts with dedup keys, context trimming, validation
- [x] `LLMProvider` Protocol with `generate_json()` in contract
- [x] MLX Gemma provider: retry logic, model load recovery, string-aware brace matching, truncation recovery
- [x] Scribe agent: fuzzy dedup (0.7 threshold), field validation, length caps, consecutive failure tracking
- [x] Analyst agent: 7 actions, dedup prevention, multi-format parser (FunctionGemma + Gemma 3 + bare names)
- [x] Architect agent: 5 domain templates
- [x] SQLite persistence (events + state)
- [x] Main orchestrator: per-cycle timing, health checks, error propagation
- [x] **Tested with real Gemma 3 4B** — 6.33s avg, dedup verified ✅
- [x] `.copilot/` agent memory (product.md, tasks.md, hackathon_context.md, implementation_tracker.md)

### Fine-Tuning (Dev 1) ✅
- [x] Data generation script: 122 examples across 7 actions, 5 domains, 14 edge cases
- [x] Edge cases: ambiguity, contradictions, sarcasm, decision reversals, implicit tasks, multi-person
- [x] MLX LoRA fine-tuning script: stratified splits, --mask-prompt, gradient accumulation, evaluation
- [x] **Trained on Mac in 1 minute** (Gemma 3 1B, 100 iters, 2.4 GB peak)
- [x] **Fused model at `outputs/analyst-fused` (0.77 GB)**
- [x] **Eval: 81.2% accuracy (13/16)**
  - extract_action_item: 100% ✅
  - log_decision: 100% ✅
  - flag_gap: 100% ✅
  - request_artifact: 100% ✅
  - suggest_next_step: 100% ✅
  - continue_observing: 50% ⚠️
  - provide_insight: 0% ❌ (needs more examples)

---

## 🔵 IN PROGRESS

### Fine-Tuning (Dev 1) ✅
- [x] Data generation script: 165 examples across 7 actions, 5 domains, 30+ edge cases
- [x] Contrastive pairs teaching insight vs gap vs suggestion distinctions
- [x] MLX LoRA fine-tuning: stratified splits, --mask-prompt, gradient accumulation
- [x] **v1:** 100 iters → 81.2% (provide_insight 0%)
- [x] **v2:** 300 iters → **85.7% (18/21)**, all 7 actions working
- [x] Fused model at `outputs/analyst-fused` (0.77 GB, 2.8 min training)
- [x] `provide_insight` 0% → 100%, `continue_observing` 50% → 100%

---

## 🔵 IN PROGRESS

### Dashboard (Dev 1)
- [x] FastAPI server with SSE endpoint (localhost-first)
- [x] Dashboard HTML — dark premium theme (v1 shell)
- [x] Real-time panels: Timeline, Action Items, Decisions, Gaps, Suggestions
- [x] Artifact viewer panel
- [x] Whiteboard state preview
- [x] Electron desktop shell (local backend + dashboard embedded)
- [x] Consolidated UI under `ui/` and backend runtime under `backend/`

### Agent 1: Perceiver (Dev 2 — Friend)
- [ ] Camera capture module (OpenCV, 3-5s intervals)
- [ ] Audio capture module (sounddevice, continuous)
- [ ] Gemma 3n E4B multimodal perception (vision + audio)
- [ ] Whisper.cpp fallback for STT
- [ ] Output `Perception` objects matching `core/config.py` contract
- [ ] Handle webcam/mic failures gracefully

### Integration (Both Devs)
- [ ] Wire Agent 1 → Agent 2 via direct Python calls
- [ ] End-to-end test: camera+mic → perceive → scribe → analyst → architect → dashboard
- [ ] Handle edge cases: silence, empty board, rapid domain switches
- [ ] Performance tuning (target: <10s perception cycle)

### Demo Prep (Both Devs)
- [ ] 3 demo scenarios (engineering arch, sales pipeline, product planning)
- [ ] Printed props (diagrams, whiteboard sheets)
- [ ] Demo script (90 seconds)
- [ ] WiFi disconnect moment in demo
- [ ] Backup recorded video
- [ ] Pitch: problem → live demo → mic drop

---

## ⚠️ Blockers

| Blocker | Owner | Status |
|---|---|---|
| GPU VM has no internet access | Organizers | 🔄 Being fixed |
| GPU VM has no sudo access | Organizers | 🔄 Being fixed |

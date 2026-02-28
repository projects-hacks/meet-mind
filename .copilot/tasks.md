# MeetMind — Complete Task Plan

## Team Roles
- **You (Dev 1):** Agent 2 (Scribe + Analyst + Architect), Fine-tuning, Dashboard
- **Friend (Dev 2):** Agent 1 (Perceiver — camera + mic), Integration wiring

---

## 🟢 DONE

### Agent 2 Core (Dev 1)
- [x] Project structure (`backend/agents/` + `backend/core/` + `backend/main.py`)
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
- [x] Data generation script: 165 examples across 7 actions, 5 domains, 30+ edge cases
- [x] Contrastive pairs teaching insight vs gap vs suggestion distinctions
- [x] MLX LoRA fine-tuning: stratified splits, --mask-prompt, gradient accumulation
- [x] **v1:** 100 iters → 81.2% (provide_insight 0%)
- [x] **v2:** 300 iters → **85.7% (18/21)**, all 7 actions working
- [x] Fused model at `outputs/analyst-fused` (0.77 GB, 2.8 min training)
- [x] `provide_insight` 0% → 100%, `continue_observing` 50% → 100%
- [x] GPU fine-tuning script (fine_tune_gpu.py) for NVIDIA RTX PRO 6000

### Dashboard & Desktop (Dev 1) ✅
- [x] FastAPI server with SSE endpoint (localhost-first)
- [x] Dashboard HTML — dark premium theme with 12 live panels
- [x] Real-time panels: Timeline, Action Items, Decisions, Gaps, Suggestions, Insights
- [x] Artifact viewer panel + generated artifact content panel
- [x] Whiteboard state preview
- [x] Electron desktop shell (local backend + dashboard embedded)
- [x] Consolidated UI under `ui/` and backend runtime under `backend/`
- [x] Sample Perception button for testing without Agent 1
- [x] SSE heartbeat + auto-reconnect

### Agent 1: RoomScribe Perceiver (Dev 2 — Friend) ✅ Code exists
- [x] Camera capture module (OpenCV, configurable intervals)
- [x] Microphone STT module (faster-whisper, chunked real-time)
- [x] Gemma 3n E4B multimodal OCR (vision-language extraction)
- [x] STT transcript refinement via model
- [x] Standalone CLI (--stt-source=mic|camera|both|stdin)
- [x] OCR worker with threading (non-blocking frame processing)
- [x] Moved from `src/` into `backend/agents/roomscribe/`

---

## 🔴 NOT DONE — Critical Integration Gap

### Integration: RoomScribe → MeetMind Pipeline
- [ ] **Wire RoomScribe into dashboard_server** — RoomScribe outputs `Event` objects (kind=ocr/stt), but nothing converts them to `Perception` objects that MeetMind.process_perception() expects
- [ ] Add a bridge: RoomScribe Event → Perception converter
- [ ] Add `/api/start-capture` endpoint to start camera+mic in dashboard_server
- [ ] Add live audio/video status indicators to dashboard HTML
- [ ] End-to-end test: real camera + real mic → live dashboard updates
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
| RoomScribe not wired to MeetMind pipeline | Dev 1 | 🔴 Critical |
| GPU VM has no internet access | Organizers | 🔄 Being fixed |
| GPU VM has no sudo access | Organizers | 🔄 Being fixed |

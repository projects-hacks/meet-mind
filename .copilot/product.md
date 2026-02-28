# MeetMind — Product Definition

## One-Line Pitch
**The first on-device AI meeting advisor that doesn't just take notes — it actively thinks, advises, and catches what humans miss, in real-time.**

---

## ❌ What We Are NOT
We are NOT another AI note-taking app. Otter.ai, Fireflies, Fathom — they all:
- Passively record and transcribe
- Generate summaries **after** the meeting ends
- Require cloud uploads (privacy risk)
- Treat meetings as "audio to text" conversion

## ✅ What We ARE
MeetMind is an **autonomous meeting advisor** — a senior teammate sitting in every meeting who:

### 1. THINKS in Real-Time (Not Post-Meeting)
- **During** discussion: "You've discussed the API architecture but haven't assigned ownership of the database migration"
- **During** a sales call: "The client raised pricing concerns twice — consider addressing this before moving on"
- Other tools wait until the meeting ends. We act **while it matters**.

### 2. SEES + HEARS (Multimodal Intelligence)
- Camera watches the whiteboard/paper/screen
- Microphone captures the discussion
- Cross-references both: "The diagram on the board shows 3 services, but the team only discussed 2 — the third may need attention"
- No other on-device tool does vision + audio together.

### 3. ADVISES (Not Just Records)
| Passive Note-Taker (Otter.ai) | Active Advisor (MeetMind) |
|---|---|
| "Sarah said she'll do OAuth" | "Action: Sarah → OAuth by Friday (HIGH)" |
| "They discussed PostgreSQL vs MongoDB" | "Decision logged: PostgreSQL chosen. Rationale: relational data needs" |
| (misses entirely) | "⚠️ GAP: Database migration discussed but no owner assigned" |
| Summary generated after meeting | "📄 Architecture doc auto-generated from live discussion" |
| (nothing) | "💡 Suggestion: You've been on pricing for 12 min — consider timeboxing" |

### 4. RUNS ON-DEVICE (Privacy by Architecture)
- Everything runs locally — camera, mic, AI reasoning
- **Nothing leaves the room**
- Works in air-gapped environments (defense, healthcare, legal)
- No per-meeting API costs
- Demo: disconnect WiFi and it still works perfectly

---

## The 6 Agentic Actions (Beyond Note-Taking)

| Action | What It Does | Why It's Not Note-Taking |
|---|---|---|
| `extract_action_item` | Detects task assignments with owner + deadline | Auto-fills project tracking |
| `log_decision` | Records firm decisions with rejected alternatives | Prevents revisiting settled topics |
| `flag_gap` | Catches unresolved issues humans miss | **Proactive risk prevention** |
| `request_artifact` | Auto-generates docs from live discussion | Real-time deliverables |
| `suggest_next_step` | Advises what to discuss or do next | **Active meeting facilitation** |
| `provide_insight` | Surfaces relevant context or warnings | **Contextual intelligence** |

---

## Target Users & Pain Points

| User | Pain Point | MeetMind Solves |
|---|---|---|
| **Engineering teams** | Decisions get lost, nobody owns action items | Auto-tracks decisions + assigns items |
| **Sales teams** | Client objections missed, follow-ups forgotten | Real-time deal intelligence + briefs |
| **Product teams** | Scope creep, unresolved questions pile up | Gap detection prevents scope drift |
| **Exec teams** | Sensitive strategy discussions can't use cloud AI | 100% on-device, air-gap safe |
| **Compliance-heavy orgs** | HIPAA/SOX/legal prevents cloud recording | On-device = compliant by default |

---

## Why On-Device (Not Just "We Can")

The hackathon requires a *genuine reason* for on-device. Ours:

1. **Confidentiality** — M&A discussions, patent reviews, pricing strategy cannot touch a cloud server
2. **Economics** — Enterprise pays $15-30/user/month for cloud AI meeting tools. On-device = deploy once, zero marginal cost
3. **Real-time latency** — Our agent loop runs every 5s. Cloud round-trip would add 2-5s per cycle, breaking the real-time experience
4. **The "Chilling Effect"** — People self-censor when they know audio goes to the cloud. On-device removes that barrier
5. **Air-gapped environments** — Government, defense, healthcare facilities with no internet

---

## Technical Architecture Summary

```
Camera + Mic → PERCEIVER (Gemma 3n) → SCRIBE (Gemma 3 4B) → ANALYST (FunctionGemma) → ARCHITECT (Gemma 3 4B)
                                                                    ↓
                                                            DASHBOARD (FastAPI + SSE)
```

- **Perceiver (Agent 1)**: Gemma 3n E4B — vision + audio simultaneously (friend's responsibility)
- **Scribe (Agent 2)**: Gemma 3 4B via MLX — structures perceptions into meeting log
- **Analyst (Agent 3)**: FunctionGemma 270M (fine-tuned) — decides actions autonomously
- **Architect (Agent 4)**: Gemma 3 4B via MLX — generates structured business documents

All models run locally via MLX on Mac / Ollama on GPU VM.

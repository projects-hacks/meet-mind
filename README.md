# Meet Mind Agents (Prototype 1)

Private, on-device meeting agent prototype:
- real-time transcript ingestion (`stt` stream)
- camera capture every 3-5 seconds (`ocr` stream)
- one local Gemma model in MLX for text cleanup + image-to-text

First agent name: `RoomScribe`

## Why this version

This first milestone focuses on "model works end-to-end" with one Gemma model.
It runs fully on-device with no cloud API calls.

## Requirements

- Apple Silicon Mac
- Python 3.10+
- Webcam
- Microphone

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run

```bash
roomscribe \
  --camera-interval 4 \
  --stt-source both \
  --stt-model base \
  --stt-language en \
  --stt-chunk-seconds 3
```

The app:
- listens to your device microphone with local offline STT
- captures a camera frame every 3-5 seconds
- uses one Gemma MLX model for transcript cleanup and image-to-text

Optional strict model override:

```bash
roomscribe --model mlx-community/gemma-3n-e4b-it-4bit
```

Camera behavior option:

```bash
roomscribe --stt-source camera --ignore-people
```

Microphone-only option:

```bash
roomscribe --stt-source mic
```

## Notes on STT

- STT is local via `faster-whisper` (no cloud API).
- Fallback mode is available:

```bash
roomscribe --stt-source stdin
```

- First STT model load may download weights once, then runs locally.
- Model selection default behavior:
  - tries Gemma 3n E4B candidate IDs first
  - falls back to `mlx-community/gemma-3-4b-it-4bit` if none can load
  - first run downloads whichever selected model is not already cached

## Privacy

- no cloud endpoints in code
- all inference local via MLX
- suitable baseline for private company meetings and whiteboard discussions

## Structure

- `src/agents/roomscribe`: first agent implementation
- future agents should be added as `src/agents/<agent_name>`

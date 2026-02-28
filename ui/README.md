# MeetMind UI + Desktop

This directory contains **all UI code**:
- dashboard web UI: `ui/dashboard/index.html`
- Electron desktop shell: `ui/electron/`

Electron starts the local backend from `backend/main.py` and loads `http://127.0.0.1:8765`.

## Run

1. Install UI dependencies:

```bash
cd ui
npm install
```

2. Run desktop app:

```bash
npm run electron
```

3. Run strict air-gapped desktop app:

```bash
npm run electron:airgapped
```

## Optional env vars

- `MEETMIND_PYTHON` (default `python3`)
- `MEETMIND_PORT` (default `8765`)
- `MEETMIND_AIR_GAPPED=1`

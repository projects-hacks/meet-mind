# MeetMind (Local Desktop)

## Final structure

- **Backend (single place):** `backend/`
- **UI + Electron (single place):** `ui/`

## Run Electron app

### 1) Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2) Install UI/Electron dependencies

```bash
cd ../ui
npm install
```

### 3) Start desktop app (local mode)

```bash
npm run electron
```

### 4) Start desktop app (strict air-gapped mode)

```bash
npm run electron:airgapped
```

## Optional env vars

- `MEETMIND_PYTHON` (default: `python3`)
- `MEETMIND_PORT` (default: `8765`)
- `MEETMIND_AIR_GAPPED=1`

## Quick UI cross-check checklist

1. Desktop window opens and loads dashboard (no blank page).
2. Header shows local connection status turning green.
3. Click **Sample Perception** → Key Points/Timeline/Last Cycle update.
4. Click **Generate Summary** → Artifact Content panel updates.
5. Click **Reset Session** → state clears and health still updates.
6. In air-gapped mode, startup remains localhost-only.

const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const http = require('http');

const HOST = '127.0.0.1';
const PORT = Number(process.env.MEETMIND_PORT || 8765);
const ROOT_DIR = path.resolve(__dirname, '..', '..');
const VENV_PYTHON = path.join(ROOT_DIR, '.venv', 'bin', 'python');
const PYTHON_BIN = process.env.MEETMIND_PYTHON || (fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : 'python3');
const AIR_GAPPED = process.env.MEETMIND_AIR_GAPPED === '1';

let backendProcess = null;
let mainWindow = null;
let isQuitting = false;

function healthCheck() {
  return new Promise((resolve) => {
    const req = http.get(
      {
        host: HOST,
        port: PORT,
        path: '/api/health',
        timeout: 1000,
      },
      (res) => {
        resolve(res.statusCode === 200);
      }
    );

    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForBackendReady(maxAttempts = 60) {
  for (let i = 0; i < maxAttempts; i += 1) {
    // eslint-disable-next-line no-await-in-loop
    const ok = await healthCheck();
    if (ok) {
      return true;
    }
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

function startBackend() {
  const args = ['-m', 'backend.main', '--serve-dashboard', '--host', HOST, '--port', String(PORT)];
  if (AIR_GAPPED) {
    args.push('--air-gapped', '--no-remote-models');
  }

  backendProcess = spawn(PYTHON_BIN, args, {
    cwd: ROOT_DIR,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  backendProcess.stdout.on('data', (data) => {
    process.stdout.write(`[backend] ${data}`);
  });

  backendProcess.stderr.on('data', (data) => {
    process.stderr.write(`[backend] ${data}`);
  });

  backendProcess.on('exit', (code) => {
    if (!isQuitting) {
      dialog.showErrorBox(
        'MeetMind Backend Stopped',
        `The local backend exited unexpectedly (code: ${code ?? 'unknown'}).`
      );
      app.quit();
    }
  });
}

function stopBackend() {
  if (!backendProcess) {
    return;
  }
  try {
    backendProcess.kill('SIGTERM');
  } catch (_) {
    // ignore
  }
  backendProcess = null;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1420,
    height: 920,
    minWidth: 1120,
    minHeight: 720,
    show: false,
    autoHideMenuBar: true,
    title: 'MeetMind (Local)',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

async function bootstrap() {
  createWindow();
  startBackend();

  const ready = await waitForBackendReady();
  if (!ready) {
    dialog.showErrorBox(
      'MeetMind Startup Failed',
      `Could not start local backend at http://${HOST}:${PORT}.\n\n` +
        `Make sure Python dependencies are installed and '${PYTHON_BIN}' is available.`
    );
    app.quit();
    return;
  }

  await mainWindow.loadURL(`http://${HOST}:${PORT}`);
}

app.whenReady().then(bootstrap);

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    bootstrap();
  }
});

app.on('before-quit', () => {
  isQuitting = true;
  stopBackend();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    isQuitting = true;
    stopBackend();
    app.quit();
  }
});

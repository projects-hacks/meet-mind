const q = (id) => document.getElementById(id);

/* ── Inline SVG Icons (no CDN dependency — works air-gapped) ── */
const _s = (d, sz = 14) =>
  `<svg class="ico" width="${sz}" height="${sz}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;

const ICO = {
  mic: _s('<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>'),
  micOff: _s('<line x1="1" y1="1" x2="23" y2="23"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"/><path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2c0 .76-.13 1.48-.35 2.17"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/>'),
  camera: _s('<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>'),
  cameraOff: _s('<line x1="1" y1="1" x2="23" y2="23"/><path d="M21 21H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3m3-3h6l2 3h4a2 2 0 0 1 2 2v9.34m-7.72-2.06a4 4 0 1 1-5.56-5.56"/>'),
  play: _s('<polygon points="5,3 19,12 5,21"/>'),
  stop: _s('<rect x="6" y="6" width="12" height="12" rx="1"/>'),
  eye: _s('<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>'),
  fileText: _s('<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10,9 9,9 8,9"/>'),
  rotateCcw: _s('<polyline points="1,4 1,10 7,10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>'),
  shield: _s('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'),
  wifi: _s('<path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/>'),
  radio: _s('<circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14"/>'),
  brain: _s('<path d="M9.5 2A6.5 6.5 0 0 0 3 8.5c0 2.13 1 4.04 2.63 5.24C4.63 15.15 4 16.97 4 19h5.5"/><path d="M14.5 2A6.5 6.5 0 0 1 21 8.5c0 2.13-1 4.04-2.63 5.24C19.37 15.15 20 16.97 20 19h-5.5"/><line x1="12" y1="2" x2="12" y2="19"/>'),
  penTool: _s('<path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/>'),
  barChart: _s('<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>'),
  headphones: _s('<path d="M3 18v-6a9 9 0 0 1 18 0v6"/><path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zm-18 0a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z"/>'),
  clipboard: _s('<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>'),
  zap: _s('<polygon points="13,2 3,14 12,14 11,22 21,10 12,10"/>'),
  key: _s('<path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/>'),
  checkSq: _s('<polyline points="9,11 12,14 22,4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>'),
  scale: _s('<line x1="12" y1="3" x2="12" y2="21"/><polyline points="8,21 16,21"/><path d="M4.5 14.5L12 6l7.5 8.5"/><circle cx="4.5" cy="14.5" r="2.5"/><circle cx="19.5" cy="14.5" r="2.5"/>'),
  lightbulb: _s('<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"/>'),
  layout: _s('<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/>'),
  package: _s('<line x1="16.5" y1="9.4" x2="7.5" y2="4.21"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27,6.96 12,12.01 20.73,6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>'),
  activity: _s('<polyline points="22,12 18,12 15,21 9,3 6,12 2,12"/>'),
  cpu: _s('<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'),
};

const api = {
  summary: '/api/summary',
  reset: '/api/reset',
  sample: '/api/sample',
  health: '/api/health',
  captureStatus: '/api/capture/status',
  startCapture: '/api/start-capture',
  stopCapture: '/api/stop-capture',
  generateSummary: '/api/artifacts/meeting-summary',
  stream: '/api/stream',
  modelStatus: '/api/model-status',
};

let eventSource = null;

function renderList(el, items, fmt) {
  el.innerHTML = '';
  if (!items || !items.length) {
    el.innerHTML = '<li class="muted">No items yet</li>';
    return;
  }
  for (const item of items) {
    const li = document.createElement('li');
    li.textContent = fmt ? fmt(item) : String(item);
    el.appendChild(li);
  }
}

function render(summary) {
  const mode = summary.air_gapped ? 'AIR-GAPPED' : 'LOCAL';
  const lastAction = summary.last_action?.action || 'none';
  q('meta').textContent = `${summary.topic || 'Unknown'} · ${summary.domain || 'general'} · ${mode} · last action ${lastAction} · avg cycle ${summary.avg_cycle_time || 0}s`;

  // Overview metrics card (enterprise-style snapshot)
  const topicEl = q('overviewTopic');
  if (topicEl) {
    topicEl.textContent = summary.topic || 'Unknown meeting';
    const domainEl = q('overviewDomain');
    if (domainEl) domainEl.textContent = `${summary.domain || 'general'} · ${mode}`;
    const lastActionEl = q('overviewLastAction');
    if (lastActionEl) lastActionEl.textContent = lastAction;
    const perceptionsEl = q('overviewPerceptions');
    if (perceptionsEl) perceptionsEl.textContent = String(summary.perception_count ?? 0);
    const avgCycleEl = q('overviewAvgCycle');
    if (avgCycleEl) avgCycleEl.textContent = `${summary.avg_cycle_time || 0}s`;
  }

  renderList(q('keyPoints'), summary.key_points || []);
  renderList(q('actionItems'), summary.action_items || [], (a) => `${a.owner}: ${a.task} (${a.deadline})`);
  renderList(q('decisions'), summary.decisions || [], (d) => d.decision);
  const gapItems = summary.gaps || [];
  const suggestionItems = summary.suggestions || [];
  const insightItems = summary.insights || [];

  // Gaps (deprecated)
  const gapsHeader = q('hGaps');
  if (gapsHeader) gapsHeader.style.display = 'none';
  const gapsEl = q('gaps');
  if (gapsEl) gapsEl.style.display = 'none';


  // Suggestions
  const sugEl = q('suggestions');
  if (suggestionItems.length) {
    renderList(sugEl, suggestionItems, (s) => s.replace(/^💡\s*/, ''));
    sugEl.style.display = '';
  } else {
    sugEl.innerHTML = '';
    sugEl.style.display = 'none';
  }

  // Insights
  const insEl = q('insights');
  if (insightItems.length) {
    renderList(insEl, insightItems, (i) => i.replace(/^🔍\s*/, ''));
    insEl.style.display = '';
  } else {
    insEl.innerHTML = '';
    insEl.style.display = 'none';
  }

  // Combined empty state
  const emptyEl = q('insightsGapsEmpty');
  if (emptyEl) {
    emptyEl.style.display = (gapItems.length || suggestionItems.length || insightItems.length) ? 'none' : '';
  }
  renderList(q('timeline'), summary.timeline || [], (t) => `${t.time || '--'} · ${t.content || ''}`);
  q('whiteboard').textContent = summary.whiteboard || 'No whiteboard summary yet';
  renderList(
    q('artifacts'),
    summary.artifacts || [],
    (a) => `${a.type || 'artifact'} · ${a.status || 'pending'}${a.source ? ` · ${a.source}` : ''}${a.context ? ` · ${a.context}` : ''}`,
  );
}

function setConnection(connected) {
  q('connDot').className = `status-dot ${connected ? 'status-ok' : 'status-down'}`;
  q('connText').innerHTML = connected
    ? `${ICO.wifi} Local stream connected`
    : `${ICO.wifi} Local stream disconnected`;
}

function setCaptureStatus(capture) {
  const running = Boolean(capture?.running);
  const audio = Boolean(capture?.audio_active);
  const camera = Boolean(capture?.camera_active);
  const hasError = Boolean(capture?.last_error);
  const dotClass = hasError ? 'status-down' : running ? 'status-ok' : 'status-idle';

  q('captureDot').className = `status-dot ${dotClass}`;
  const micIco = audio ? ICO.mic : ICO.micOff;
  const camIco = camera ? ICO.camera : ICO.cameraOff;
  q('captureText').innerHTML = hasError
    ? `${ICO.radio} Capture error: ${capture.last_error}`
    : running
      ? `${ICO.radio} Capture running · ${micIco} ${audio ? 'on' : 'off'} · ${camIco} ${camera ? 'on' : 'off'}`
      : `${ICO.radio} Capture stopped`;

  const payload = {
    running,
    camera_active: camera,
    audio_active: audio,
    model_id: capture?.model_id || '',
    cycles: capture?.cycles || 0,
    audio_chunks: capture?.audio_chunks || 0,
    visual_events: capture?.visual_events || 0,
    last_audio_ts: capture?.last_audio_ts || null,
    last_visual_ts: capture?.last_visual_ts || null,
    last_cycle_ts: capture?.last_cycle_ts || null,
    last_error: capture?.last_error || '',
  };
  // User-facing capture metrics
  const summaryEl = q('captureSummary');
  if (summaryEl) {
    if (hasError) {
      summaryEl.innerHTML = `${ICO.radio} Capture error: ${payload.last_error}`;
    } else if (running) {
      const mI = audio ? ICO.mic : ICO.micOff;
      const cI = camera ? ICO.camera : ICO.cameraOff;
      summaryEl.innerHTML = `${ICO.radio} Capture running · ${mI} ${audio ? 'on' : 'off'} · ${cI} ${camera ? 'on' : 'off'}`;
    } else {
      summaryEl.innerHTML = `${ICO.radio} Capture stopped`;
    }
  }
}

async function fetchSummary() {
  const res = await fetch(api.summary);
  if (!res.ok) return;
  render(await res.json());
}

function updateModelStatus(statusObj) {
  let allReady = true;
  for (const [key, info] of Object.entries(statusObj)) {
    const el = q(`${key}Status`);
    if (el) {
      el.innerHTML = info.status === 'ready' ? `${ICO.activity} Ready` : info.status === 'error' ? `${ICO.radio} Error` : `${ICO.cpu} Loading...`;
      el.className = `status-${info.status}`;
    }
    if (info.status !== 'ready') allReady = false;
  }

  const overlay = q('loadingOverlay');
  if (overlay) {
    if (allReady) {
      overlay.classList.add('hidden');
    } else {
      overlay.classList.remove('hidden');
    }
  }
}

async function fetchModelStatus() {
  const res = await fetch(api.modelStatus);
  if (!res.ok) return;
  const data = await res.json();
  if (data.status) {
    updateModelStatus(data.status);
  }
}

async function fetchHealth() {
  const res = await fetch(api.health);
  if (!res.ok) return;
  const data = await res.json();
  const h = data.health || {};

  const pill = q('healthStatusPill');
  if (pill) {
    const healthy = Boolean(h.scribe_healthy);
    pill.innerHTML = healthy ? `${ICO.activity} Healthy` : `${ICO.activity} Degraded`;
    pill.className = `status-pill ${healthy ? 'ok' : 'warn'}`;
  }

  const cyclesEl = q('healthCycles');
  if (cyclesEl) cyclesEl.textContent = String(h.cycles ?? 0);
  const avgEl = q('healthAvgCycle');
  if (avgEl) avgEl.textContent = `${h.avg_cycle_time ?? 0}s`;
  const modeEl = q('healthMode');
  if (modeEl) modeEl.textContent = h.air_gapped ? 'AIR-GAPPED' : 'LOCAL';

  const debug = q('healthDebug');
  if (debug) debug.textContent = JSON.stringify(h, null, 2);
  if (data.capture) setCaptureStatus(data.capture);
}

async function fetchCaptureStatus() {
  const res = await fetch(api.captureStatus);
  if (!res.ok) return;
  const data = await res.json();
  if (data.capture) setCaptureStatus(data.capture);
}

function connectSSE() {
  if (eventSource) eventSource.close();

  eventSource = new EventSource(api.stream);
  eventSource.onopen = () => setConnection(true);
  eventSource.onerror = () => {
    setConnection(false);
    setTimeout(connectSSE, 1500);
  };

  eventSource.addEventListener('summary', (evt) => {
    try { render(JSON.parse(evt.data)); } catch (_) { }
  });

  eventSource.addEventListener('cycle_result', (evt) => {
    let parsed;
    try { parsed = JSON.parse(evt.data); } catch (_) { parsed = null; }
    if (parsed) {
      const action = parsed.action || {};
      const artifact = parsed.artifact || {};
      const actionName = action.action || 'none';
      const reasoning = action.reasoning || artifact.error || 'No reasoning available.';

      // Append to Architect Feed (Filter out noisy continue_observing)
      if (actionName !== 'continue_observing') {
        const feed = q('architectFeed');
        if (feed) {
          const item = document.createElement('div');
          item.className = 'feed-item feed-type-architect';
          const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
          item.innerHTML = `
            <div class="feed-meta"><span>${ts}</span><span>${ICO.zap} ${actionName.toUpperCase()}</span></div>
            <div>${reasoning}</div>
          `;
          feed.appendChild(item);
          feed.scrollTop = feed.scrollHeight;

          // Clear "Waiting for Architect actions..." if present
          const waiting = feed.querySelector('.muted.small-text');
          if (waiting) waiting.remove();
        }
      }
    }
    fetchSummary();
  });

  eventSource.addEventListener('raw_perception', (evt) => {
    try {
      const data = JSON.parse(evt.data);
      const feed = q('rawFeed');
      if (feed && data.text) {
        const item = document.createElement('div');
        item.className = `feed-item feed-type-${data.type}`;
        const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
        const sourceLabel = data.type === 'stt' ? `${ICO.mic} Microphone` : `${ICO.camera} Camera`;
        item.innerHTML = `
          <div class="feed-meta"><span>${ts}</span><span>${sourceLabel}</span></div>
          <div>${data.text}</div>
        `;
        feed.appendChild(item);
        feed.scrollTop = feed.scrollHeight;
      }
    } catch (_) { }
  });

  eventSource.addEventListener('artifact', (evt) => {
    try {
      const a = JSON.parse(evt.data);
      q('artifactContent').textContent = a.content || 'No artifact content';
    } catch (_) { }
    fetchSummary();
  });

  eventSource.addEventListener('capture_status', (evt) => {
    try {
      const status = JSON.parse(evt.data);
      setCaptureStatus(status);
    } catch (_) { }
  });

  eventSource.addEventListener('model_status', (evt) => {
    try {
      const status = JSON.parse(evt.data);
      updateModelStatus(status);
    } catch (_) { }
  });
}

q('resetBtn').addEventListener('click', async () => {
  await fetch(api.reset, { method: 'POST' });
  q('artifactContent').textContent = 'No generated artifact content yet';

  const rawFeed = q('rawFeed');
  if (rawFeed) rawFeed.innerHTML = '<div class="muted small-text">Listening for audio and camera text...</div>';

  const architectFeed = q('architectFeed');
  if (architectFeed) architectFeed.innerHTML = '<div class="muted small-text">Waiting for Architect actions...</div>';

  fetchSummary();
});

q('sampleBtn').addEventListener('click', async () => {
  await fetch(api.sample, { method: 'POST' });
});

// Pipeline auto-starts, so buttons are hidden in UI
// q('startCaptureBtn').addEventListener('click', async () => { ... });
// q('stopCaptureBtn').addEventListener('click', async () => { ... });

q('summaryBtn').addEventListener('click', async () => {
  const res = await fetch(api.generateSummary, { method: 'POST' });
  if (res.ok) {
    const payload = await res.json();
    q('artifactContent').textContent = payload.content || 'No artifact content';
    fetchSummary();
  }
});

/* ── Hydrate static icon labels ── */
function hydrateIcons() {
  const set = (id, html) => { const el = q(id); if (el) el.innerHTML = html; };

  // Loading overlay model labels
  set('vlmLabel', `${ICO.brain} RoomScribe VLM`);
  set('scribeLabel', `${ICO.penTool} Scribe Agent`);
  set('analystLabel', `${ICO.barChart} Analyst Agent`);

  // Header badge & buttons
  set('badgeLabel', `${ICO.shield} LOCAL · ON DEVICE`);
  const startBtn = q('startCaptureBtn');
  if (startBtn) startBtn.style.display = 'none';
  const stopBtn = q('stopCaptureBtn');
  if (stopBtn) stopBtn.style.display = 'none';
  set('sampleBtn', `${ICO.eye} Sample Perception`);
  set('summaryBtn', `${ICO.fileText} Generate Summary`);
  set('resetBtn', `${ICO.rotateCcw} Reset Session`);

  // Column headers
  set('col1Title', `${ICO.headphones} Transcribing (Agent 1)`);
  set('col1Sub', `${ICO.mic} STT  &  ${ICO.camera} OCR`);
  set('col2Title', `${ICO.clipboard} Meeting State (Agent 2)`);
  set('col2Sub', `${ICO.penTool} Scribe  &  ${ICO.barChart} Analyst`);
  set('col3Title', `${ICO.zap} Actions (Agent 3)`);
  set('col3Sub', `${ICO.cpu} Architect Workflows`);

  // Section headers
  set('hKeyPoints', `${ICO.key} Key Points`);
  set('hActionItems', `${ICO.checkSq} Action Items`);
  set('hDecisions', `${ICO.scale} Decisions`);
  set('hInsights', `${ICO.lightbulb} Insights & Gaps`);
  set('hGaps', `${ICO.activity} Gaps`);
  set('hWhiteboard', `${ICO.layout} Whiteboard / Visuals`);
  set('hArtifact', `${ICO.package} Generated Artifact`);
}
hydrateIcons();

fetchSummary();
fetchHealth();
fetchCaptureStatus();
fetchModelStatus();
connectSSE();
setInterval(fetchHealth, 5000);
setInterval(fetchCaptureStatus, 4000);

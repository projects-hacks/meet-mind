const q = (id) => document.getElementById(id);

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
  renderList(q('gaps'), summary.gaps || [], (g) => `${g.topic} [${g.gap_type}]`);
  renderList(q('suggestions'), summary.suggestions || [], (s) => s.replace(/^💡\s*/, ''));
  renderList(q('insights'), summary.insights || [], (i) => i.replace(/^🔍\s*/, ''));
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
  q('connText').textContent = connected ? 'Local stream connected' : 'Local stream disconnected';
}

function setCaptureStatus(capture) {
  const running = Boolean(capture?.running);
  const audio = Boolean(capture?.audio_active);
  const camera = Boolean(capture?.camera_active);
  const hasError = Boolean(capture?.last_error);
  const dotClass = hasError ? 'status-down' : running ? 'status-ok' : 'status-idle';

  q('captureDot').className = `status-dot ${dotClass}`;
  q('captureText').textContent = hasError
    ? `Capture error: ${capture.last_error}`
    : running
      ? `Capture running · audio ${audio ? 'on' : 'off'} · camera ${camera ? 'on' : 'off'}`
      : 'Capture stopped';

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
      summaryEl.textContent = `Capture error: ${payload.last_error}`;
    } else if (running) {
      summaryEl.textContent = `Capture running · audio ${audio ? 'on' : 'off'} · camera ${camera ? 'on' : 'off'}`;
    } else {
      summaryEl.textContent = 'Capture stopped';
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
      el.textContent = info.status === 'ready' ? 'Ready' : info.status === 'error' ? 'Error' : 'Loading...';
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
    pill.textContent = healthy ? 'Healthy' : 'Degraded';
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

      // Append to Architect Feed
      const feed = q('architectFeed');
      if (feed) {
        const item = document.createElement('div');
        item.className = 'feed-item feed-type-architect';
        const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
        item.innerHTML = `
          <div class="feed-meta"><span>${ts}</span><span>${actionName}</span></div>
          <div>${reasoning}</div>
        `;
        feed.appendChild(item);
        feed.scrollTop = feed.scrollHeight;
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
        const sourceLabel = data.type === 'stt' ? 'Microphone' : 'Camera';
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

q('startCaptureBtn').addEventListener('click', async () => {
  const res = await fetch(api.startCapture, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      camera_interval: 4.0,
      stt_model: 'base',
      stt_language: 'en',
      stt_chunk_seconds: 3.0,
      stt_sample_rate: 16000,
      ignore_people: false,
    }),
  });
  if (res.ok) {
    const payload = await res.json();
    if (payload.capture) setCaptureStatus(payload.capture);
  }
});

q('stopCaptureBtn').addEventListener('click', async () => {
  const res = await fetch(api.stopCapture, { method: 'POST' });
  if (res.ok) {
    const payload = await res.json();
    if (payload.capture) setCaptureStatus(payload.capture);
  }
});

q('summaryBtn').addEventListener('click', async () => {
  const res = await fetch(api.generateSummary, { method: 'POST' });
  if (res.ok) {
    const payload = await res.json();
    q('artifactContent').textContent = payload.content || 'No artifact content';
    fetchSummary();
  }
});

fetchSummary();
fetchHealth();
fetchCaptureStatus();
fetchModelStatus();
connectSSE();
setInterval(fetchHealth, 5000);
setInterval(fetchCaptureStatus, 4000);

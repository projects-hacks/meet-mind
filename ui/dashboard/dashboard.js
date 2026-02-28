const q = (id) => document.getElementById(id);

const api = {
  summary: '/api/summary',
  reset: '/api/reset',
  sample: '/api/perceptions/sample',
  health: '/api/health',
  captureStatus: '/api/capture/status',
  startCapture: '/api/start-capture',
  stopCapture: '/api/stop-capture',
  generateSummary: '/api/artifacts/meeting-summary',
  stream: '/api/stream',
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
  q('captureStatus').textContent = JSON.stringify(payload, null, 2);
}

async function fetchSummary() {
  const res = await fetch(api.summary);
  if (!res.ok) return;
  render(await res.json());
}

async function fetchHealth() {
  const res = await fetch(api.health);
  if (!res.ok) return;
  const data = await res.json();
  q('health').textContent = JSON.stringify(data.health || {}, null, 2);
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
    try { render(JSON.parse(evt.data)); } catch (_) {}
  });

  eventSource.addEventListener('cycle_result', (evt) => {
    try { q('cycleResult').textContent = JSON.stringify(JSON.parse(evt.data), null, 2); } catch (_) {}
    fetchSummary();
    fetchHealth();
  });

  eventSource.addEventListener('artifact', (evt) => {
    try {
      const a = JSON.parse(evt.data);
      q('artifactContent').textContent = a.content || 'No artifact content';
    } catch (_) {}
    fetchSummary();
  });

  eventSource.addEventListener('capture_status', (evt) => {
    try {
      const status = JSON.parse(evt.data);
      setCaptureStatus(status);
    } catch (_) {}
  });
}

q('resetBtn').addEventListener('click', async () => {
  await fetch(api.reset, { method: 'POST' });
  q('artifactContent').textContent = 'No generated artifact content yet';
  q('cycleResult').textContent = '{}';
  fetchHealth();
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
connectSSE();
setInterval(fetchHealth, 5000);
setInterval(fetchCaptureStatus, 4000);

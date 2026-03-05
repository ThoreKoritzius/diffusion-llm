let runId = null;
let runSessionId = 0;
let terminalStateReached = false;
let terminalSqlText = "";

let es = null;
let pollTimer = null;
let playbackTimer = null;
let rateLimitTimer = null;
let queueCountdownTimer = null;
let fallbackPolling = false;
let pollDelayMs = 250;

let historySnapshots = [];
let historyByStep = new Map();
let pendingSnapshots = [];
let donePayload = null;
let lastSnapshotCount = 0;
let lastStatusMsg = "";

let queueEtaBaseline = null;
let queueEtaSeconds = null;
let queueEtaSyncedAtMs = 0;
let queuePositionCurrent = null;

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

const stepBox = document.getElementById('stepBox');
const statusBox = document.getElementById('status');
const runBtn = document.getElementById('runBtn');
const runBtnLabel = document.getElementById('runBtnLabel');
const runBtnProgress = document.getElementById('runBtnProgress');
const runForm = document.getElementById('runForm');
const sliderRow = document.getElementById('sliderRow');
const snapSlider = document.getElementById('snapSlider');
const snapSliderLabel = document.getElementById('snapSliderLabel');
const queueChip = document.getElementById('queueChip');
const liveFront = document.getElementById('liveTextFront');
const liveBack = document.getElementById('liveTextBack');

const viewTabs = Array.from(document.querySelectorAll('.view-tab'));
const inferencePanel = document.getElementById('view-inference');
const infoPanel = document.getElementById('view-info');
const introModal = document.getElementById('introModal');
const introDismiss = document.getElementById('introDismiss');
const introStart = document.getElementById('introStart');

const UI_MODES = {
  IDLE: 'idle',
  QUEUED: 'queued',
  RUNNING: 'running',
  FINALIZING: 'finalizing',
  RATE_LIMITED: 'rate_limited',
  ERROR: 'error',
};

function setStatus(msg) {
  statusBox.textContent = msg || '';
}

function setQueueChip(text, state = '') {
  queueChip.textContent = text || '';
  queueChip.dataset.state = state;
}

function formatCountdown(seconds) {
  const safe = Math.max(0, Math.ceil(seconds));
  const mins = Math.floor(safe / 60);
  const secs = safe % 60;
  if (mins <= 0) return `${secs}s`;
  if (mins < 60) return `${mins}m ${String(secs).padStart(2, '0')}s`;
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  return `${hours}h ${String(remMins).padStart(2, '0')}m`;
}

function getQueueRemainingSeconds() {
  if (!Number.isFinite(queueEtaSeconds) || queueEtaSeconds <= 0 || !queueEtaSyncedAtMs) {
    return null;
  }
  const elapsed = (Date.now() - queueEtaSyncedAtMs) / 1000;
  return Math.max(0, Math.ceil(queueEtaSeconds - elapsed));
}

function stopQueueCountdown() {
  if (queueCountdownTimer) {
    clearInterval(queueCountdownTimer);
    queueCountdownTimer = null;
  }
}

function renderQueuedCountdown() {
  const posText = Number.isFinite(queuePositionCurrent) && queuePositionCurrent > 0 ? `#${queuePositionCurrent}` : '#?';
  const remaining = getQueueRemainingSeconds();

  if (!Number.isFinite(remaining)) {
    runBtnLabel.textContent = `Queued ${posText} • ~estimating`;
    setQueueChip(`Queued ${posText}`, 'queued');
    setStatus(`Queued ${posText} • ~estimating`);
    setProgressIndeterminate();
    return;
  }

  const countdownText = formatCountdown(remaining);
  runBtnLabel.textContent = `Queued ${posText} • ${countdownText}`;
  setQueueChip(`Queued ${posText} • ${countdownText}`, 'queued');
  setStatus(`Queued ${posText} • ${countdownText}`);

  if (!Number.isFinite(queueEtaBaseline) || queueEtaBaseline <= 0 || remaining > queueEtaBaseline) {
    queueEtaBaseline = remaining;
  }
  const pct = (remaining / Math.max(queueEtaBaseline, 1)) * 100;
  setProgressValue(Math.max(0, Math.min(100, pct)));
}

function syncQueuedEstimate(queuePosition, etaSeconds) {
  queuePositionCurrent = Number.isFinite(queuePosition) && queuePosition > 0 ? queuePosition : null;

  if (Number.isFinite(etaSeconds) && etaSeconds > 0) {
    queueEtaSeconds = etaSeconds;
    queueEtaSyncedAtMs = Date.now();
    if (!Number.isFinite(queueEtaBaseline) || queueEtaBaseline <= 0 || etaSeconds > queueEtaBaseline) {
      queueEtaBaseline = etaSeconds;
    }
  } else {
    queueEtaSeconds = null;
    queueEtaSyncedAtMs = 0;
  }

  renderQueuedCountdown();

  if (!queueCountdownTimer) {
    queueCountdownTimer = setInterval(() => {
      renderQueuedCountdown();
    }, 250);
  }
}

function setProgressHidden() {
  runBtn.dataset.progress = 'hidden';
  runBtnProgress.style.width = '0%';
}

function setProgressIndeterminate() {
  runBtn.dataset.progress = 'indeterminate';
  runBtnProgress.style.width = '36%';
}

function setProgressValue(pct) {
  runBtn.dataset.progress = 'visible';
  const clamped = Math.max(0, Math.min(100, Math.round(pct || 0)));
  runBtnProgress.style.width = `${clamped}%`;
}

function setUiState(mode, data = {}) {
  runBtn.dataset.mode = mode;

  if (mode === UI_MODES.IDLE) {
    stopQueueCountdown();
    runBtn.disabled = false;
    runBtnLabel.textContent = 'Run Generation';
    setProgressHidden();
    setQueueChip('', '');
    return;
  }

  if (mode === UI_MODES.QUEUED) {
    runBtn.disabled = true;
    syncQueuedEstimate(data.queuePosition, data.etaSeconds);
    return;
  }

  if (mode === UI_MODES.RUNNING) {
    stopQueueCountdown();
    runBtn.disabled = true;
    const stepLabel = data.stepLabel || 'Running';
    runBtnLabel.textContent = `Running • ${stepLabel}`;
    setQueueChip('Running', 'running');

    if (Number.isFinite(data.progressPct)) {
      setProgressValue(Math.max(5, 100 - data.progressPct));
    } else {
      setProgressIndeterminate();
    }
    return;
  }

  if (mode === UI_MODES.FINALIZING) {
    stopQueueCountdown();
    runBtn.disabled = true;
    runBtnLabel.textContent = 'Finalizing...';
    setQueueChip('Finalizing', 'running');
    setProgressIndeterminate();
    return;
  }

  if (mode === UI_MODES.RATE_LIMITED) {
    stopQueueCountdown();
    runBtn.disabled = true;
    runBtnLabel.textContent = `Rate limited • ${data.remaining || 0}s`;
    setQueueChip('Rate limited', 'rate_limited');
    setProgressValue(data.progressPct || 100);
    return;
  }

  if (mode === UI_MODES.ERROR) {
    stopQueueCountdown();
    runBtn.disabled = true;
    runBtnLabel.textContent = 'Run Failed';
    setQueueChip('Error', 'rate_limited');
    setProgressHidden();
  }
}

function stopTransports() {
  fallbackPolling = false;
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  if (es) {
    es.close();
    es = null;
  }
}

function resetRunState() {
  stopTransports();
  if (playbackTimer) {
    clearInterval(playbackTimer);
    playbackTimer = null;
  }
  if (rateLimitTimer) {
    clearInterval(rateLimitTimer);
    rateLimitTimer = null;
  }
  stopQueueCountdown();

  runId = null;
  runSessionId += 1;
  terminalStateReached = false;
  terminalSqlText = '';

  pollDelayMs = 250;
  lastSnapshotCount = 0;
  lastStatusMsg = '';
  donePayload = null;

  historySnapshots = [];
  historyByStep = new Map();
  pendingSnapshots = [];

  queueEtaBaseline = null;
  queueEtaSeconds = null;
  queueEtaSyncedAtMs = 0;
  queuePositionCurrent = null;

  sliderRow.classList.remove('visible');
  setUiState(UI_MODES.IDLE);
}

function fitLiveFontForElem(elem) {
  const wrapper = elem.parentElement;
  const width = Math.max(220, wrapper.clientWidth - 28);
  const text = elem.textContent || '';
  const lines = text.split('\n');
  const longest = lines.reduce((m, line) => Math.max(m, line.length), 0) || 20;
  const minFont = window.innerWidth <= 980 ? 14 : 22;
  const maxFont = window.innerWidth <= 980 ? 28 : 72;
  const calc = Math.floor(width / Math.max(8, longest) * 1.9);
  elem.style.fontSize = `${Math.max(minFont, Math.min(maxFont, calc))}px`;
}

function fitLiveFont() {
  fitLiveFontForElem(liveFront);
  fitLiveFontForElem(liveBack);
}

window.addEventListener('resize', fitLiveFont);

let animating = false;
function animateBlurToText(newText) {
  if (prefersReducedMotion.matches) {
    renderImmediate(newText);
    return;
  }

  if (animating) {
    liveFront.textContent = newText;
    fitLiveFontForElem(liveFront);
    return;
  }

  liveBack.textContent = newText || '';
  fitLiveFontForElem(liveBack);
  liveBack.style.filter = 'blur(8px)';
  liveBack.style.opacity = '0';
  liveBack.style.zIndex = '2';
  liveFront.style.zIndex = '1';

  requestAnimationFrame(() => {
    animating = true;
    liveBack.style.filter = 'blur(0px)';
    liveBack.style.opacity = '1';
    liveFront.style.opacity = '0';

    setTimeout(() => {
      liveFront.textContent = newText || '';
      fitLiveFontForElem(liveFront);
      liveFront.style.opacity = '1';
      liveFront.style.zIndex = '2';
      liveBack.style.opacity = '0';
      liveBack.style.filter = 'blur(8px)';
      liveBack.style.zIndex = '1';
      animating = false;
    }, 220);
  });
}

function renderImmediate(text) {
  liveFront.textContent = text || '(empty)';
  fitLiveFontForElem(liveFront);
}

function extractSQL(s) {
  const start = s.indexOf('<SQL>');
  const end = s.indexOf('</SQL>');
  if (start !== -1 && end !== -1) {
    return s.substring(start + 5, end).trim();
  }
  return s;
}

function normalizeSQLDisplay(sql) {
  if (!sql) return '(empty)';
  const raw = String(sql).replace(/\r/g, '');
  return raw.trim() || '(empty)';
}

function sanitizeFinalSql(sql) {
  if (!sql) return '';
  let out = String(sql);
  // Normalize zero-width / NBSP style spacing artifacts first.
  out = out.replace(/[\u200B-\u200D\uFEFF]/g, '').replace(/\u00A0/g, ' ');
  // Collapse letter-spaced special tags like "< p a d >", "< / s >", "< m a s k >", "[ p a d ]".
  out = out
    .replace(/<\s*\/?\s*p\s*a\s*d\s*>/gi, ' ')
    .replace(/\[\s*p\s*a\s*d\s*\]/gi, ' ')
    .replace(/<\s*\/?\s*s\s*>/gi, ' ')
    .replace(/<\s*m\s*a\s*s\s*k\s*>/gi, ' ');
  return out
    // Remove common special/pad tokens that must never appear in terminal output.
    .replace(/<\s*\/?\s*pad\s*>/gi, ' ')
    .replace(/\[\s*pad\s*\]/gi, ' ')
    .replace(/<\s*\/?\s*s\s*>/gi, ' ')
    .replace(/<\s*mask\s*>/gi, ' ')
    // Remove contiguous underscore masks like "____"
    .replace(/\s*_{2,}\s*/g, ' ')
    // Remove tokenized/space-separated underscore runs like "_ _ _ _"
    .replace(/(?:^|\s)(?:_+\s+){1,}_+(?=\s|$)/g, ' ')
    // Remove any remaining standalone underscore-only tokens
    .replace(/(?:^|\s)_+(?=\s|$)/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function snapshotText(snap) {
  return normalizeSQLDisplay(snap.sql_only || extractSQL(snap.text || ''));
}

function isTerminalSnapshot(snap) {
  if (!snap) return false;
  const step = Number(snap.step);
  const total = Number(snap.total_steps);
  return Number.isFinite(step) && Number.isFinite(total) && total > 0 && step === total;
}

function upsertSnapshot(snap) {
  if (!snap || terminalStateReached) return;
  const step = Number.isFinite(Number(snap.step)) ? Number(snap.step) : historySnapshots.length;
  if (historyByStep.has(step)) return;
  historyByStep.set(step, snap);
  historySnapshots = Array.from(historyByStep.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([, v]) => v);
}

function updateSliderLabel() {
  if (!historySnapshots.length) {
    snapSliderLabel.textContent = '';
    return;
  }
  const idx = parseInt(snapSlider.value, 10);
  const snap = historySnapshots[idx] || {};
  snapSliderLabel.textContent = `Step ${snap.step || idx} / ${snap.total_steps || historySnapshots.length - 1}`;
}

function applySnapshotAnimated(snap) {
  if (!snap) return;
  const text = isTerminalSnapshot(snap)
    ? sanitizeFinalSql(snapshotText(snap))
    : snapshotText(snap);
  animateBlurToText(text);
  stepBox.textContent = `Step ${snap.step} / ${snap.total_steps}`;
}

function applySnapshotImmediate(snap) {
  if (!snap) return;
  const text = isTerminalSnapshot(snap)
    ? sanitizeFinalSql(snapshotText(snap))
    : snapshotText(snap);
  renderImmediate(text);
  stepBox.textContent = `Step ${snap.step} / ${snap.total_steps}`;
}

function ensurePlayback() {
  if (playbackTimer) return;
  playbackTimer = setInterval(() => {
    if (!pendingSnapshots.length) {
      clearInterval(playbackTimer);
      playbackTimer = null;
      maybeFinalize();
      return;
    }
    applySnapshotAnimated(pendingSnapshots.shift());
    maybeFinalize();
  }, 85);
}

function enqueueAnimated(snap) {
  if (!snap) return;
  pendingSnapshots.push(snap);
  ensurePlayback();
}

function replayAllOnceAndFinalize() {
  setUiState(UI_MODES.FINALIZING);
  // Replay all non-terminal history steps, then append one canonical terminal frame.
  pendingSnapshots = historySnapshots.filter((snap) => !isTerminalSnapshot(snap));
  const terminalFromHistory = historySnapshots.find((snap) => isTerminalSnapshot(snap));
  const terminalStep = terminalFromHistory
    ? Number(terminalFromHistory.step)
    : (historySnapshots.length ? Number(historySnapshots[historySnapshots.length - 1].step) : 0);
  const terminalTotal = terminalFromHistory
    ? Number(terminalFromHistory.total_steps)
    : (historySnapshots.length ? Number(historySnapshots[historySnapshots.length - 1].total_steps) : terminalStep);
  if (terminalSqlText) {
    pendingSnapshots.push({
      step: Number.isFinite(terminalStep) ? terminalStep : 0,
      total_steps: Number.isFinite(terminalTotal) ? terminalTotal : (Number.isFinite(terminalStep) ? terminalStep : 0),
      sql_only: sanitizeFinalSql(terminalSqlText),
      text: `<SQL>${sanitizeFinalSql(terminalSqlText)}</SQL>`,
    });
  }
  if (!pendingSnapshots.length) {
    maybeFinalize();
    return;
  }
  ensurePlayback();
}

function maybeFinalize() {
  if (!donePayload || pendingSnapshots.length) return;
  finishRun(donePayload);
  donePayload = null;
}

function finishRun(payload) {
  stopTransports();
  terminalStateReached = true;

  if (payload && payload.sql_only) {
    terminalSqlText = sanitizeFinalSql(payload.sql_only);
  }
  if (!terminalSqlText && historySnapshots.length > 0) {
    terminalSqlText = sanitizeFinalSql(snapshotText(historySnapshots[historySnapshots.length - 1]));
  }

  if (payload && payload.state === 'error') {
    setStatus(payload.status || 'Run failed.');
    renderImmediate('Run failed. Check status for details.');
    setUiState(UI_MODES.ERROR);
    setTimeout(() => setUiState(UI_MODES.IDLE), 600);
    return;
  }

  if (payload && payload.state === 'timed_out') {
    setStatus(payload.status || 'Run timed out.');
    renderImmediate('Run timed out.');
    setUiState(UI_MODES.IDLE);
    return;
  }

  if (payload && payload.state === 'stopped') {
    setStatus(payload.status || 'Run stopped.');
    renderImmediate('Run stopped.');
    setUiState(UI_MODES.IDLE);
    return;
  }

  setStatus('Completed.');

  if (historySnapshots.length > 0) {
    snapSlider.max = String(historySnapshots.length - 1);
    snapSlider.value = snapSlider.max;
    sliderRow.classList.add('visible');
    updateSliderLabel();
    if (terminalSqlText) {
      renderImmediate(terminalSqlText);
    } else {
      applySnapshotImmediate(historySnapshots[historySnapshots.length - 1]);
    }
  } else if (terminalSqlText) {
    renderImmediate(terminalSqlText);
  }

  setUiState(UI_MODES.IDLE);
}

function parseStepProgress(statusMsg) {
  if (!statusMsg) return null;
  const m = statusMsg.match(/step\s+(\d+)\/(\d+)/i);
  if (!m) return null;
  const step = parseInt(m[1], 10);
  const total = parseInt(m[2], 10);
  if (!Number.isFinite(step) || !Number.isFinite(total) || total <= 0) return null;
  return {
    label: `Step ${step}/${total}`,
    progressPct: Math.round((step / total) * 100),
  };
}

function setQueuedUi(queuePosition, etaSeconds, _etaConfidence) {
  setUiState(UI_MODES.QUEUED, {
    queuePosition,
    etaSeconds,
  });
}

function updateRunningUi(statusMsg) {
  const step = parseStepProgress(statusMsg);
  if (step) {
    setUiState(UI_MODES.RUNNING, { stepLabel: step.label, progressPct: step.progressPct });
    setStatus(`Running • ${step.label}`);
  } else {
    setUiState(UI_MODES.RUNNING, { stepLabel: 'processing' });
    setStatus('Running...');
  }
}

async function startRateLimitCooldown(retryAfterSeconds, message) {
  if (rateLimitTimer) {
    clearInterval(rateLimitTimer);
    rateLimitTimer = null;
  }
  const total = Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0 ? Math.ceil(retryAfterSeconds) : 10;
  let remaining = total;
  setStatus(message || 'Rate limited. Please retry shortly.');
  setUiState(UI_MODES.RATE_LIMITED, { remaining, progressPct: 100 });

  rateLimitTimer = setInterval(() => {
    remaining -= 1;
    if (remaining <= 0) {
      clearInterval(rateLimitTimer);
      rateLimitTimer = null;
      setStatus('Ready.');
      setUiState(UI_MODES.IDLE);
      return;
    }
    const pct = Math.max(0, Math.round((remaining / total) * 100));
    setUiState(UI_MODES.RATE_LIMITED, { remaining, progressPct: pct });
  }, 1000);
}

async function pollRunState(id) {
  if (terminalStateReached) return;

  try {
    const resp = await fetch(`/run/${id}?after=${encodeURIComponent(lastSnapshotCount)}`, { cache: 'no-store' });
    if (!resp.ok) {
      if (resp.status === 429) {
        const retryAfter = parseInt(resp.headers.get('Retry-After') || '', 10);
        await startRateLimitCooldown(retryAfter, 'Rate limited while polling.');
      }
      return;
    }

    const data = await resp.json();
    const list = data.snapshots || [];

    for (const snap of list) {
      upsertSnapshot(snap);
    }

    if (list.length === 1) {
      enqueueAnimated(list[0]);
    } else if (list.length > 1) {
      enqueueAnimated(list[0]);
      enqueueAnimated(list[list.length - 1]);
    }

    if (Number.isFinite(data.snapshot_count)) {
      lastSnapshotCount = data.snapshot_count;
    } else {
      lastSnapshotCount += list.length;
    }

    if (data.state === 'queued') {
      setQueuedUi(data.queue_position, data.eta_seconds, data.eta_confidence);
      pollDelayMs = 800;
    } else if (data.state === 'running') {
      updateRunningUi(data.status || '');
      pollDelayMs = 250;
    } else if (data.state === 'stopping') {
      setUiState(UI_MODES.RUNNING, { stepLabel: 'stopping' });
      setStatus('Stopping...');
      pollDelayMs = 300;
    }

    if (data.status && data.status !== lastStatusMsg) {
      lastStatusMsg = data.status;
    }

    if (data.done) {
      stopTransports();
      terminalStateReached = true;
      donePayload = data;
      if (data.sql_only) {
        terminalSqlText = sanitizeFinalSql(data.sql_only);
      } else if (historySnapshots.length > 0) {
        terminalSqlText = sanitizeFinalSql(snapshotText(historySnapshots[historySnapshots.length - 1]));
      }
      replayAllOnceAndFinalize();
    }
  } catch (err) {
    console.debug('Polling exception', err);
  }
}

function startPollingFallback(id) {
  if (fallbackPolling || terminalStateReached) return;
  fallbackPolling = true;
  if (es) {
    es.close();
    es = null;
  }

  const loop = async () => {
    if (!fallbackPolling || terminalStateReached) return;
    await pollRunState(id);
    if (!fallbackPolling || terminalStateReached) return;
    pollTimer = setTimeout(loop, pollDelayMs);
  };

  loop();
}

function openStream(id, sessionId) {
  if (es) {
    es.close();
    es = null;
  }

  es = new EventSource(`/stream/${id}`);

  es.onerror = (err) => {
    if (terminalStateReached || sessionId !== runSessionId) return;
    console.debug('SSE error; switching to polling fallback', err);
    startPollingFallback(id);
  };

  es.addEventListener('snapshot', (ev) => {
    if (terminalStateReached || sessionId !== runSessionId) return;
    const snap = JSON.parse(ev.data);
    upsertSnapshot(snap);
    enqueueAnimated(snap);
    lastSnapshotCount = historySnapshots.length;
  });

  es.addEventListener('status', (ev) => {
    if (terminalStateReached || sessionId !== runSessionId) return;
    const info = JSON.parse(ev.data);
    if (!info || !info.msg) return;

    lastStatusMsg = info.msg;
    if (info.msg.toLowerCase().startsWith('step ')) {
      updateRunningUi(info.msg);
    }
  });

  es.addEventListener('done', (ev) => {
    if (terminalStateReached || sessionId !== runSessionId) return;
    stopTransports();
    terminalStateReached = true;

    const payload = JSON.parse(ev.data) || {};
    donePayload = payload;
    if (payload.sql_only) {
      terminalSqlText = sanitizeFinalSql(payload.sql_only);
    } else if (historySnapshots.length > 0) {
      terminalSqlText = sanitizeFinalSql(snapshotText(historySnapshots[historySnapshots.length - 1]));
    }
    replayAllOnceAndFinalize();
  });
}

runForm.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  resetRunState();

  renderImmediate('Queueing request...');
  setUiState(UI_MODES.QUEUED, { queuePosition: null, etaSeconds: null });
  setStatus('Submitting run...');

  const form = new FormData(runForm);

  try {
    const resp = await fetch('/start', { method: 'POST', body: form });
    if (!resp.ok) {
      const bodyText = await resp.text();
      let bodyJson = null;
      try {
        bodyJson = JSON.parse(bodyText);
      } catch (_ignored) {}

      if (resp.status === 429) {
        const retryAfter = parseInt(resp.headers.get('Retry-After') || '', 10);
        await startRateLimitCooldown(retryAfter, bodyJson && bodyJson.message ? bodyJson.message : 'Rate limited.');
        return;
      }

      const msg = bodyJson && bodyJson.message ? bodyJson.message : bodyText;
      setStatus(`Error: ${msg}`);
      setUiState(UI_MODES.ERROR);
      setTimeout(() => setUiState(UI_MODES.IDLE), 800);
      return;
    }

    const payload = await resp.json();
    runId = payload.run_id;
    setQueuedUi(payload.queue_position, payload.eta_seconds, payload.eta_confidence);
    openStream(runId, runSessionId);
  } catch (err) {
    setStatus(`Error: ${err}`);
    setUiState(UI_MODES.ERROR);
    setTimeout(() => setUiState(UI_MODES.IDLE), 800);
  }
});

snapSlider.addEventListener('input', () => {
  if (!historySnapshots.length) return;
  const idx = parseInt(snapSlider.value, 10);
  const snap = historySnapshots[idx];

  if (terminalSqlText && idx === historySnapshots.length - 1) {
    renderImmediate(terminalSqlText);
    if (snap) {
      stepBox.textContent = `Step ${snap.step} / ${snap.total_steps}`;
    }
  } else {
    applySnapshotImmediate(snap);
  }

  updateSliderLabel();
});

function guessType(value) {
  if (/^[\d]+$/.test(value)) return 'INT';
  if (/^[\d\.]+$/.test(value) && value.includes('.')) return 'FLOAT';
  return 'TEXT';
}

function processCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) return;

  const headers = lines[0].split(',');
  let dataRow = null;
  for (let i = 1; i < lines.length; i += 1) {
    if (lines[i].trim()) {
      dataRow = lines[i].split(',');
      break;
    }
  }
  if (!dataRow) dataRow = headers.map(() => '');

  const types = dataRow.map(guessType);
  const dataRows = lines.slice(1).map((row) => row.split(',')).slice(0, 30);

  let html = `<table><thead><tr>${headers.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>`;
  if (!dataRows.length) {
    html += `<tr><td colspan="${headers.length}"><em>No data rows</em></td></tr>`;
  }
  dataRows.forEach((row) => {
    html += `<tr>${headers.map((_, i) => `<td>${row[i] !== undefined ? row[i] : ''}</td>`).join('')}</tr>`;
  });
  html += '</tbody></table>';

  document.getElementById('csvTableWrapper').innerHTML = html;

  const colDefs = headers.map((h, i) => `  ${h} ${types[i] || 'TEXT'}`).join(',\n');
  const createStmt = `CREATE TABLE table_name (\n${colDefs}\n);`;
  const context = document.getElementById('context');
  if (context) context.value = createStmt;
}

document.getElementById('csvInput').addEventListener('change', (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => processCSV(e.target.result);
  reader.readAsText(file);
});

function activateView(viewName) {
  viewTabs.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === viewName);
  });
  inferencePanel.classList.toggle('active', viewName === 'inference');
  infoPanel.classList.toggle('active', viewName === 'info');
}

viewTabs.forEach((btn) => {
  btn.addEventListener('click', () => {
    activateView(btn.dataset.view);
  });
});

function closeIntroModal() {
  introModal.classList.remove('visible');
  sessionStorage.setItem('diffusion_intro_seen', '1');
}

if (introDismiss) introDismiss.addEventListener('click', closeIntroModal);
if (introStart) {
  introStart.addEventListener('click', () => {
    activateView('inference');
    closeIntroModal();
  });
}

if (introModal && !sessionStorage.getItem('diffusion_intro_seen')) {
  introModal.classList.add('visible');
}

fitLiveFont();
setUiState(UI_MODES.IDLE);
setStatus('Ready.');
renderImmediate('Ready for generation. Submit a prompt to start.');

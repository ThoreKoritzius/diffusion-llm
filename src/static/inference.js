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
let exportableRunId = null;
let lastSnapshotCount = 0;
let lastStatusMsg = "";
let lastCompletedSnapshots = [];
let lastCompletedTerminalSql = "";
let lastCompletedSignature = "";
let pendingRunSignature = "";

let queueEtaBaseline = null;
let queueEtaSeconds = null;
let queueEtaSyncedAtMs = 0;
let queuePositionCurrent = null;
let queueDemandCurrent = '';
let queueZeroSinceMs = 0;

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

const stepBox = document.getElementById('stepBox');
const effPill = document.getElementById('effPill');
const statusBox = document.getElementById('status');
const runBtn = document.getElementById('runBtn');
const runBtnLabel = document.getElementById('runBtnLabel');
const runBtnSub = document.getElementById('runBtnSub');
const runBtnProgress = document.getElementById('runBtnProgress');
const runForm = document.getElementById('runForm');
const sliderRow = document.getElementById('sliderRow');
const snapSlider = document.getElementById('snapSlider');
const snapSliderLabel = document.getElementById('snapSliderLabel');
const queueChip = document.getElementById('queueChip');
const exportGifBtn = document.getElementById('exportGifBtn');
const liveTokenPre = document.getElementById('liveTokenPre');
const promptInput = document.getElementById('prompt');
const contextInput = document.getElementById('context');
const sqlLenInput = document.getElementById('sql_len');
const maxLenInput = document.getElementById('max_len');

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

const CHARS_PER_TOKEN_EST = 3.0;
const STRUCTURE_TOKEN_RESERVE = 32;
const PROMPT_BUDGET_RATIO = 0.35;

function setStatus(msg) {
  statusBox.textContent = msg || '';
}

function setQueueChip(text, state = '') {
  if (!queueChip) return;
  queueChip.textContent = text || '';
  queueChip.dataset.state = state;
}

function setRunButtonText(primary, secondary = '') {
  runBtnLabel.textContent = primary || '';
  if (runBtnSub) runBtnSub.textContent = secondary || '';
}

function cloneSnapshots(snaps) {
  return (snaps || []).map((snap) => ({ ...snap }));
}

function buildRunSignature(form) {
  const fields = [
    'prompt',
    'context',
    'steps',
    'max_len',
    'sql_len',
    'top_k',
    'top_p',
    'model_dir',
  ];
  const data = {};
  fields.forEach((k) => {
    data[k] = String(form.get(k) || '');
  });
  return JSON.stringify(data);
}

function replayCachedRunIfAvailable(signature) {
  if (!signature || !lastCompletedSignature || signature !== lastCompletedSignature) return false;
  if (!lastCompletedSnapshots.length) return false;
  resetRunState();
  historySnapshots = cloneSnapshots(lastCompletedSnapshots);
  historyByStep = new Map(
    historySnapshots.map((snap, idx) => {
      const step = Number(snap.step);
      return [Number.isFinite(step) ? step : idx, snap];
    }),
  );
  terminalSqlText = lastCompletedTerminalSql || '';
  donePayload = { state: 'done', status: 'replayed previous run' };
  setStatus('Replaying previous generation...');
  replayAllOnceAndFinalize();
  return true;
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

function formatElapsed(ms) {
  return formatCountdown(Math.max(0, Math.floor(ms / 1000)));
}

function parseRetryAfter(resp, bodyJson = null, fallbackSeconds = 5) {
  const bodyRetry = bodyJson && Number(bodyJson.retry_after);
  if (Number.isFinite(bodyRetry) && bodyRetry > 0) return Math.ceil(bodyRetry);
  const headerRetry = parseInt(resp.headers.get('Retry-After') || '', 10);
  if (Number.isFinite(headerRetry) && headerRetry > 0) return headerRetry;
  return fallbackSeconds;
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
    queueZeroSinceMs = 0;
    setRunButtonText(`Queued ${posText}`, 'Estimating wait');
    setQueueChip(`Queued ${posText}`, 'queued');
    setStatus(`Queued ${posText} • ~estimating`);
    setProgressIndeterminate();
    return;
  }

  if (remaining <= 0) {
    if (!queueZeroSinceMs) queueZeroSinceMs = Date.now();
    const elapsed = formatElapsed(Date.now() - queueZeroSinceMs);
    setRunButtonText('Next up', `Queued ${posText} • handoff ${elapsed}`);
    setQueueChip(`Next up ${posText}`, 'queued');
    setStatus(`Queued ${posText} • handing off to worker • ${elapsed}`);
    setProgressIndeterminate();
    return;
  }

  queueZeroSinceMs = 0;
  const countdownText = formatCountdown(remaining);
  const demand = queueDemandCurrent && queueDemandCurrent !== 'low' ? ` • ${queueDemandCurrent}` : '';
  setRunButtonText(`Queued ${posText}`, `First frame in ~${countdownText}${demand}`);
  setQueueChip(`Queued ${posText} • ${countdownText}${demand}`, 'queued');
  setStatus(`Queued ${posText} • ${countdownText}${demand}`);

  if (!Number.isFinite(queueEtaBaseline) || queueEtaBaseline <= 0 || remaining > queueEtaBaseline) {
    queueEtaBaseline = remaining;
  }
  const pct = (remaining / Math.max(queueEtaBaseline, 1)) * 100;
  setProgressValue(Math.max(0, Math.min(100, pct)));
}

function syncQueuedEstimate(queuePosition, etaSeconds, demand = '') {
  queuePositionCurrent = Number.isFinite(queuePosition) && queuePosition > 0 ? queuePosition : null;
  queueDemandCurrent = demand || '';

  if (Number.isFinite(etaSeconds) && etaSeconds > 0) {
    if (!Number.isFinite(queueEtaSeconds) || Math.abs(queueEtaSeconds - etaSeconds) > 2) {
      queueEtaBaseline = etaSeconds;
    }
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
  runBtn.style.setProperty('--pct', '0%');
  runBtnProgress.style.width = '0%';
}

function setProgressIndeterminate() {
  runBtn.dataset.progress = 'indeterminate';
  runBtnProgress.style.width = '36%';
}

function setProgressValue(pct) {
  runBtn.dataset.progress = 'visible';
  const clamped = Math.max(0, Math.min(100, pct || 0));
  runBtn.style.setProperty('--pct', `${clamped}%`);
  runBtnProgress.style.width = `${Math.round(clamped)}%`;
}

function setUiState(mode, data = {}) {
  runBtn.dataset.mode = mode;

  if (mode === UI_MODES.IDLE) {
    stopQueueCountdown();
    runBtn.disabled = false;
    queueZeroSinceMs = 0;
    setRunButtonText('Run Generation', '');
    setProgressHidden();
    setQueueChip('', '');
    return;
  }

  if (mode === UI_MODES.QUEUED) {
    runBtn.disabled = true;
    syncQueuedEstimate(data.queuePosition, data.etaSeconds, data.demand || '');
    return;
  }

  if (mode === UI_MODES.RUNNING) {
    stopQueueCountdown();
    runBtn.disabled = true;
    const stepLabel = data.stepLabel || 'Running';
    const isStep = Number.isFinite(data.progressPct);
    const primary = data.primaryLabel || (isStep ? stepLabel : stepLabel);
    const secondary = data.secondaryLabel || (isStep ? 'Rendering diffusion steps' : 'Preparing first preview');
    setRunButtonText(primary, secondary);
    setQueueChip('Running', 'running');

    if (isStep) {
      setProgressValue(Math.max(5, data.progressPct));
    } else {
      setProgressIndeterminate();
    }
    return;
  }

  if (mode === UI_MODES.FINALIZING) {
    stopQueueCountdown();
    runBtn.disabled = true;
    setRunButtonText('Finalizing', 'Preparing final output');
    setQueueChip('Finalizing', 'running');
    setProgressIndeterminate();
    return;
  }

  if (mode === UI_MODES.RATE_LIMITED) {
    stopQueueCountdown();
    runBtn.disabled = true;
    setRunButtonText('Rate limited', `Retry in ${data.remaining || 0}s`);
    setQueueChip('Rate limited', 'rate_limited');
    setProgressValue(data.progressPct || 100);
    return;
  }

  if (mode === UI_MODES.ERROR) {
    stopQueueCountdown();
    runBtn.disabled = true;
    setRunButtonText('Run Failed', 'Check status below');
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
  exportableRunId = null;
  if (exportGifBtn) exportGifBtn.disabled = true;
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
  tokenSlots = [];
  while (liveTokenPre.firstChild) liveTokenPre.removeChild(liveTokenPre.firstChild);

  queueEtaBaseline = null;
  queueEtaSeconds = null;
  queueEtaSyncedAtMs = 0;
  queuePositionCurrent = null;
  queueDemandCurrent = '';
  queueZeroSinceMs = 0;

  // Keep the timeline + GIF controls in the layout at all times; just disable
  // them until a run completes, so nothing appears/disappears mid-generation.
  sliderRow.classList.remove('visible');
  if (snapSlider) snapSlider.disabled = true;
  if (snapSliderLabel) snapSliderLabel.textContent = 'Timeline';
  setEfficiencyIdle();
  setUiState(UI_MODES.IDLE);
}

// Token slot state — each entry is {element, text, isMask}
let tokenSlots = [];

function tokenizeSql(sql) {
  const parts = (sql || '').split(/(\s+|_{2,}|[(),*=<>!;])/);
  return parts.filter((t) => t.length > 0);
}

function isMaskToken(tok) {
  return /^_{2,}$/.test(tok);
}

const SQL_KEYWORDS = new Set(['SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'HAVING',
  'LIMIT', 'OFFSET', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'FULL', 'ON', 'AND', 'OR',
  'NOT', 'IN', 'AS', 'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'CREATE', 'TABLE',
  'DISTINCT', 'UNION', 'ALL', 'ASC', 'DESC', 'BETWEEN', 'LIKE', 'IS', 'NULL', 'EXISTS',
  'CASE', 'WHEN', 'THEN', 'ELSE', 'END']);
const SQL_FUNCS = new Set(['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'ROUND', 'ABS', 'COALESCE',
  'UPPER', 'LOWER', 'LENGTH', 'NOW', 'DATE', 'CAST']);

// Pretty-print the query across lines so it reads like an editor instead of one
// long shrinking line. Only rewrites existing whitespace (keeps token count
// stable so the per-token reveal animation isn't disturbed).
function formatSqlForDisplay(sql) {
  if (!sql) return sql;
  const base = String(sql).replace(/\s+/g, ' ').trim();
  // Split out single-quoted string literals (incl. '' escapes and an unterminated
  // quote mid-animation) so we never insert line breaks inside a value.
  const parts = base.split(/('(?:[^']|'')*'?)/);
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) continue; // odd indices are quoted literals — leave intact
    parts[i] = parts[i]
      .replace(/\s+\b(FROM|WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|UNION(?: ALL)?|INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL JOIN|OUTER JOIN|JOIN|VALUES|SET)\b/gi, '\n$1')
      .replace(/\s+\b(AND|OR)\b/gi, '\n  $1');
  }
  return parts.join('');
}

function classifyToken(tok) {
  const t = (tok || '').trim();
  if (!t) return '';
  const up = t.toUpperCase();
  if (SQL_KEYWORDS.has(up)) return ' keyword';
  if (SQL_FUNCS.has(up)) return ' fn';
  if (/^'.*'$/.test(t) || /^".*"$/.test(t)) return ' str';
  if (/^[(),*=<>!;.+/-]+$/.test(t)) return ' punct';
  return '';
}

function getOrCreateSlot(index) {
  if (tokenSlots[index]) return tokenSlots[index];
  const span = document.createElement('span');
  span.className = 'tok';
  liveTokenPre.appendChild(span);
  tokenSlots[index] = { element: span, text: null, isMask: null };
  return tokenSlots[index];
}

function updateTokenDisplay(sql) {
  const tokens = tokenizeSql(formatSqlForDisplay(sql));

  // If the pre has stale non-span content (e.g. text node from renderImmediate), nuke it
  if (tokenSlots.length === 0 && liveTokenPre.firstChild) {
    liveTokenPre.textContent = '';
  }

  // Remove excess slots from the end
  while (tokenSlots.length > tokens.length) {
    const removed = tokenSlots.pop();
    liveTokenPre.removeChild(removed.element);
  }

  tokens.forEach((tok, i) => {
    const slot = getOrCreateSlot(i);
    const mask = isMaskToken(tok);

    if (slot.text === tok) return;
    slot.text = tok;

    const typeClass = mask ? '' : classifyToken(tok);
    if (mask) {
      if (!slot.isMask) {
        slot.element.style.minWidth = `max(${tok.length}ch, 2ch)`;
      }
      slot.element.className = 'tok masked';
      slot.element.textContent = tok;
    } else if (slot.isMask) {
      // Mask → resolved: fade in
      slot.element.className = 'tok resolving' + typeClass;
      slot.element.textContent = tok;
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          slot.element.className = 'tok resolved' + typeClass;
          setTimeout(() => { slot.element.style.minWidth = ''; }, 200);
        });
      });
    } else {
      slot.element.className = 'tok resolved' + typeClass;
      slot.element.textContent = tok;
    }

    slot.isMask = mask;
  });

  fitLiveFont();
}

function fitLiveFont() {
  const box = liveTokenPre.closest('.live-box') || liveTokenPre.parentElement;
  const cs = getComputedStyle(box);
  const padX = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
  const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
  const availW = Math.max(200, box.clientWidth - padX);
  const availH = Math.max(160, box.clientHeight - padY);
  const text = liveTokenPre.textContent || '';
  const lines = text.split('\n');
  const longest = lines.reduce((m, line) => Math.max(m, line.length), 0) || 12;
  const nLines = Math.max(lines.length, 1);
  const isMobile = window.innerWidth <= 980;
  const minFont = isMobile ? 15 : 18;
  const maxFont = isMobile ? 26 : 34;
  // Fit to both width (~0.62em per mono glyph) and height (line-height 1.7),
  // then clamp to a comfortable, stable range so the font doesn't jump per frame.
  const byWidth = availW / (longest * 0.62);
  const byHeight = availH / (nLines * 1.7);
  const size = Math.floor(Math.min(byWidth, byHeight));
  liveTokenPre.style.fontSize = `${Math.max(minFont, Math.min(maxFont, size))}px`;
}

window.addEventListener('resize', fitLiveFont);

function renderImmediate(text) {
  // Clear all token slots and reset to plain text (used for status messages)
  while (tokenSlots.length) {
    const removed = tokenSlots.pop();
    liveTokenPre.removeChild(removed.element);
  }
  liveTokenPre.textContent = text || '(empty)';
  fitLiveFont();
}

function extractSQL(s) {
  const safe = String(s || '');
  const start = safe.lastIndexOf('<SQL>');
  if (start === -1) {
    return '';
  }
  const bodyStart = start + 5;
  const end = safe.indexOf('</SQL>', bodyStart);
  if (end === -1) {
    return safe.substring(bodyStart).trim();
  }
  return safe.substring(bodyStart, end).trim();
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

const GIF_SIZE = 480;
const GIF_PALETTE = [
  [255, 255, 255], [246, 247, 251], [239, 243, 250], [215, 222, 234],
  [15, 23, 42], [85, 96, 113], [31, 111, 235], [26, 127, 75],
  [124, 58, 237], [180, 35, 63], [167, 111, 29], [196, 208, 226],
  [224, 233, 246], [207, 224, 255], [233, 240, 255], [232, 246, 237],
];

while (GIF_PALETTE.length < 256) GIF_PALETTE.push([0, 0, 0]);

function gifNearestPaletteIndex(r, g, b, a) {
  if (a < 128) return 0;
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < 16; i += 1) {
    const p = GIF_PALETTE[i];
    const dr = r - p[0];
    const dg = g - p[1];
    const db = b - p[2];
    const dist = dr * dr + dg * dg + db * db;
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

function gifBytesFromCanvas(ctx, width, height) {
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const indexed = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 1) {
    indexed[j] = gifNearestPaletteIndex(rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]);
  }
  return indexed;
}

function pushWordWrapped(ctx, text, x, y, maxWidth, lineHeight, maxLines = 3) {
  const words = String(text || '').split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';
  for (const word of words) {
    const next = line ? `${line} ${word}` : word;
    if (ctx.measureText(next).width <= maxWidth || !line) {
      line = next;
    } else {
      lines.push(line);
      line = word;
      if (lines.length >= maxLines) break;
    }
  }
  if (line && lines.length < maxLines) lines.push(line);
  lines.forEach((row, idx) => ctx.fillText(row, x, y + idx * lineHeight));
  return y + lines.length * lineHeight;
}

function gifFormatSqlLines(sql, ctx, maxWidth, maxLines) {
  const formatted = formatSqlForDisplay(String(sql || '').trim() || '____');
  const rawLines = formatted.split('\n');
  const lines = [];
  for (const raw of rawLines) {
    let line = '';
    for (const part of raw.split(/(\s+)/)) {
      const next = line + part;
      if (ctx.measureText(next).width <= maxWidth || !line) {
        line = next;
      } else {
        lines.push(line.trimEnd());
        line = part.trimStart();
        if (lines.length >= maxLines) return lines;
      }
    }
    lines.push(line.trimEnd());
    if (lines.length >= maxLines) return lines;
  }
  return lines;
}

function renderGifFrame(ctx, snap, prompt, size = GIF_SIZE) {
  const pad = 44;
  const innerW = size - pad * 2;
  const sql = isTerminalSnapshot(snap) ? sanitizeFinalSql(snapshotText(snap)) : snapshotText(snap);
  const step = Number(snap.step) || 0;
  const total = Math.max(1, Number(snap.total_steps) || step || 1);
  const frac = Math.max(0, Math.min(1, total > 0 ? step / total : 0));

  ctx.fillStyle = '#f6f7fb';
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(pad - 16, pad - 16, innerW + 32, size - pad * 2 + 32);
  ctx.strokeStyle = '#d7deea';
  ctx.lineWidth = 2;
  ctx.strokeRect(pad - 16, pad - 16, innerW + 32, size - pad * 2 + 32);

  ctx.fillStyle = '#1f6feb';
  ctx.font = '700 28px system-ui, -apple-system, Segoe UI, sans-serif';
  ctx.fillText('Text to SQL Diffusion', pad, pad + 10);

  ctx.fillStyle = '#556071';
  ctx.font = '500 19px system-ui, -apple-system, Segoe UI, sans-serif';
  pushWordWrapped(ctx, prompt || 'SQL generation', pad, pad + 54, innerW, 26, 3);

  const barY = 180;
  ctx.fillStyle = '#eff3fa';
  ctx.fillRect(pad, barY, innerW, 14);
  ctx.fillStyle = '#1f6feb';
  ctx.fillRect(pad, barY, Math.max(14, innerW * frac), 14);
  ctx.fillStyle = '#556071';
  ctx.font = '700 18px system-ui, -apple-system, Segoe UI, sans-serif';
  ctx.fillText(frac >= 1 ? 'done' : `denoising - step ${step}/${total}`, pad, barY + 42);

  ctx.font = '700 26px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
  const lines = gifFormatSqlLines(sql, ctx, innerW, 13);
  let y = 280;
  const lineH = 38;
  for (const line of lines) {
    let x = pad;
    for (const tok of tokenizeSql(line)) {
      const txt = tok || '';
      const w = ctx.measureText(txt).width;
      if (isMaskToken(txt)) {
        ctx.fillStyle = '#c4d0e2';
        ctx.fillRect(x, y - 24, Math.max(16, w), 28);
      } else {
        const cls = classifyToken(txt);
        if (cls.includes('keyword')) ctx.fillStyle = '#1f6feb';
        else if (cls.includes('fn')) ctx.fillStyle = '#7c3aed';
        else if (cls.includes('str')) ctx.fillStyle = '#1a7f4b';
        else if (cls.includes('punct')) ctx.fillStyle = '#556071';
        else ctx.fillStyle = '#0f172a';
        ctx.fillText(txt, x, y);
      }
      x += w;
    }
    y += lineH;
  }
}

function gifWriteShort(out, value) {
  out.push(value & 255, (value >> 8) & 255);
}

function gifWriteSubBlocks(out, bytes) {
  for (let i = 0; i < bytes.length; i += 255) {
    const chunk = bytes.slice(i, i + 255);
    out.push(chunk.length, ...chunk);
  }
  out.push(0);
}

function gifLzwEncode(indices, minCodeSize = 8) {
  const clear = 1 << minCodeSize;
  const end = clear + 1;
  const codeSize = minCodeSize + 1;
  let bitBuf = 0;
  let bitLen = 0;
  const bytes = [];

  function writeCode(code) {
    bitBuf |= code << bitLen;
    bitLen += codeSize;
    while (bitLen >= 8) {
      bytes.push(bitBuf & 255);
      bitBuf >>= 8;
      bitLen -= 8;
    }
  }

  // Literal-code GIF LZW. This is intentionally low-compression but robust:
  // emit clear codes frequently enough that code size never grows past 9 bits.
  for (let i = 0; i < indices.length;) {
    writeCode(clear);
    const chunkEnd = Math.min(indices.length, i + 240);
    while (i < chunkEnd) {
      writeCode(indices[i]);
      i += 1;
    }
  }
  writeCode(end);
  if (bitLen > 0) bytes.push(bitBuf & 255);
  return bytes;
}

function buildGifBlob(frames, prompt) {
  const canvas = document.createElement('canvas');
  canvas.width = GIF_SIZE;
  canvas.height = GIF_SIZE;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const out = [];
  'GIF89a'.split('').forEach((ch) => out.push(ch.charCodeAt(0)));
  gifWriteShort(out, GIF_SIZE);
  gifWriteShort(out, GIF_SIZE);
  out.push(0xf7, 0, 0);
  GIF_PALETTE.forEach((rgb) => out.push(rgb[0], rgb[1], rgb[2]));
  out.push(0x21, 0xff, 0x0b);
  'NETSCAPE2.0'.split('').forEach((ch) => out.push(ch.charCodeAt(0)));
  out.push(0x03, 0x01, 0x00, 0x00, 0x00);

  frames.forEach((snap, idx) => {
    renderGifFrame(ctx, snap, prompt, GIF_SIZE);
    const delay = idx === 0 ? 70 : (idx === frames.length - 1 ? 220 : 26);
    out.push(0x21, 0xf9, 0x04, 0x00);
    gifWriteShort(out, delay);
    out.push(0x00, 0x00);
    out.push(0x2c);
    gifWriteShort(out, 0);
    gifWriteShort(out, 0);
    gifWriteShort(out, GIF_SIZE);
    gifWriteShort(out, GIF_SIZE);
    out.push(0x00, 0x08);
    gifWriteSubBlocks(out, gifLzwEncode(gifBytesFromCanvas(ctx, GIF_SIZE, GIF_SIZE), 8));
  });
  out.push(0x3b);
  return new Blob([new Uint8Array(out)], { type: 'image/gif' });
}

function selectGifFrames(frames, maxFrames = 18) {
  if (frames.length <= maxFrames) return frames;
  const out = [];
  const lastIdx = frames.length - 1;
  for (let i = 0; i < maxFrames; i += 1) {
    const idx = Math.round((i / (maxFrames - 1)) * lastIdx);
    if (!out.length || out[out.length - 1] !== frames[idx]) out.push(frames[idx]);
  }
  return out;
}

function exportGifInBrowser() {
  const frames = cloneSnapshots(historySnapshots);
  if (terminalSqlText) {
    const last = frames[frames.length - 1] || {};
    if (!isTerminalSnapshot(last) || sanitizeFinalSql(snapshotText(last)) !== terminalSqlText) {
      frames.push({
        step: Number(last.step) || Number(last.total_steps) || frames.length,
        total_steps: Number(last.total_steps) || Number(last.step) || frames.length,
        sql_only: terminalSqlText,
        text: `<SQL>${terminalSqlText}</SQL>`,
      });
    }
  }
  if (!frames.length) throw new Error('No animation frames available.');
  const blob = buildGifBlob(selectGifFrames(frames), promptInput ? promptInput.value : '');
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'text2sql-diffusion.gif';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return blob.size;
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

// --- Efficiency vs autoregressive decoding (in forward passes / "steps") ---
// Masked diffusion fills the whole SQL window in a fixed number of steps,
// independent of output length. An autoregressive decoder emits ~1 token per
// forward pass, so it needs roughly one step per generated token. We estimate
// the output token count from the final SQL and compare against the steps used.
function computeEfficiency() {
  const sql = (terminalSqlText || '').trim();
  if (!sql) return null;

  let steps = 0;
  if (historySnapshots.length) {
    const last = historySnapshots[historySnapshots.length - 1] || {};
    steps = Number(last.total_steps) || (historySnapshots.length - 1) || historySnapshots.length;
  }
  if (!steps) {
    const stepsEl = document.getElementById('steps');
    steps = stepsEl ? parseInt(stepsEl.value, 10) : 0;
  }
  if (!steps || steps < 1) return null;

  // Autoregressive steps ≈ number of generated tokens (subword estimate).
  const arSteps = Math.max(1, Math.round(sql.length / CHARS_PER_TOKEN_EST));
  return { steps, arSteps, ratio: arSteps / steps };
}

function setEfficiencyIdle() {
  if (!effPill) return;
  effPill.textContent = '⚡ vs AR: —';
  effPill.dataset.state = 'idle';
  effPill.title = 'After a run, this shows how many fewer forward passes (steps) '
    + 'masked-diffusion decoding used versus autoregressive decoding.';
}

function updateEfficiencyReadout() {
  if (!effPill) return;
  const eff = computeEfficiency();
  if (!eff) { setEfficiencyIdle(); return; }
  const r = eff.ratio;
  if (r >= 1.05) {
    effPill.textContent = `⚡ ${r.toFixed(1)}× fewer steps`;
    effPill.dataset.state = 'good';
  } else {
    effPill.textContent = `⚡ ${eff.steps} steps`;
    effPill.dataset.state = 'neutral';
  }
  effPill.title = `This run: ${eff.steps} diffusion steps filled the SQL window in parallel. `
    + `Autoregressive decoding emits ~1 token per step (~${eff.arSteps} steps for this output) → `
    + `${r.toFixed(2)}× ${r >= 1 ? 'fewer' : 'more'} forward passes.`;
}

function applySnapshotAnimated(snap) {
  if (!snap) return;
  const isTerm = isTerminalSnapshot(snap);
  const text = isTerm
    ? sanitizeFinalSql(snapshotText(snap))
    : snapshotText(snap);
  updateTokenDisplay(text);
  stepBox.textContent = `Step ${snap.step} / ${snap.total_steps}`;
}

function applySnapshotImmediate(snap) {
  if (!snap) return;
  const text = isTerminalSnapshot(snap)
    ? sanitizeFinalSql(snapshotText(snap))
    : snapshotText(snap);
  updateTokenDisplay(text);
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
  }, 60);
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
  const resolvedTerminalText = terminalSqlText || (terminalFromHistory ? sanitizeFinalSql(snapshotText(terminalFromHistory)) : '');
  if (resolvedTerminalText) {
    if (!terminalSqlText) terminalSqlText = resolvedTerminalText;
    pendingSnapshots.push({
      step: Number.isFinite(terminalStep) ? terminalStep : 0,
      total_steps: Number.isFinite(terminalTotal) ? terminalTotal : (Number.isFinite(terminalStep) ? terminalStep : 0),
      sql_only: resolvedTerminalText,
      text: `<SQL>${resolvedTerminalText}</SQL>`,
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
    const lastTerminal = historySnapshots.find((s) => isTerminalSnapshot(s));
    const lastSnap = lastTerminal || historySnapshots[historySnapshots.length - 1];
    terminalSqlText = sanitizeFinalSql(snapshotText(lastSnap));
  }
  if (payload && payload.state === 'error') {
    pendingRunSignature = '';
    setStatus(payload.status || 'Run failed.');
    renderImmediate('Run failed. Check status for details.');
    setUiState(UI_MODES.ERROR);
    setTimeout(() => setUiState(UI_MODES.IDLE), 600);
    return;
  }

  if (payload && payload.state === 'timed_out') {
    pendingRunSignature = '';
    setStatus(payload.status || 'Run timed out.');
    renderImmediate('Run timed out.');
    setUiState(UI_MODES.IDLE);
    return;
  }

  if (payload && payload.state === 'stopped') {
    pendingRunSignature = '';
    setStatus(payload.status || 'Run stopped.');
    renderImmediate('Run stopped.');
    setUiState(UI_MODES.IDLE);
    return;
  }

  if (historySnapshots.length > 0) {
    lastCompletedSnapshots = cloneSnapshots(historySnapshots);
    lastCompletedTerminalSql = terminalSqlText || '';
    if (pendingRunSignature) lastCompletedSignature = pendingRunSignature;
  }
  pendingRunSignature = '';

  setStatus('Completed.');

  if (historySnapshots.length > 0) {
    snapSlider.max = String(historySnapshots.length - 1);
    snapSlider.value = snapSlider.max;
    snapSlider.disabled = false;
    sliderRow.classList.add('visible');
    updateSliderLabel();
    if (terminalSqlText) {
      updateTokenDisplay(terminalSqlText);
    } else {
      applySnapshotImmediate(historySnapshots[historySnapshots.length - 1]);
    }
  } else if (terminalSqlText) {
    updateTokenDisplay(terminalSqlText);
  }

  if (exportGifBtn && historySnapshots.length > 0 && runId) {
    exportableRunId = runId;
    exportGifBtn.disabled = false;
  }

  updateEfficiencyReadout();
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

function runningStatusLabel(statusMsg) {
  const msg = String(statusMsg || '').toLowerCase();
  if (msg.includes('worker claimed')) return 'Worker claimed';
  if (msg.includes('loading model') || msg.includes('loading model/tokenizer')) return 'Loading model and tokenizer';
  if (msg.includes('effective sequence length')) return 'Preparing input';
  if (msg.includes('sql window')) return 'Preparing SQL window';
  if (msg.includes('starting denoising')) return 'Starting denoising';
  if (msg.includes('building gif')) return 'Building GIF';
  return 'Preparing first preview';
}

function firstPreviewSecondary(data = {}) {
  const snapCount = Number(data.snapshot_count);
  if (Number.isFinite(snapCount) && snapCount > 0) return '';

  const remaining = Number(data.first_frame_remaining_seconds);
  if (Number.isFinite(remaining) && remaining > 0) {
    return `First preview in ~${formatCountdown(remaining)}`;
  }

  const elapsed = Number(data.running_elapsed_seconds);
  if (Number.isFinite(elapsed) && elapsed > 0) {
    return `First preview pending • ${formatElapsed(elapsed * 1000)}`;
  }

  const avg = Number(data.avg_first_frame_seconds);
  if (Number.isFinite(avg) && avg > 0) {
    return `First preview usually ~${formatCountdown(avg)}`;
  }

  return 'Preparing first preview';
}

function setQueuedUi(queuePosition, etaSeconds, _etaConfidence, demand = '') {
  setUiState(UI_MODES.QUEUED, {
    queuePosition,
    etaSeconds,
    demand,
  });
}

function updateRunningUi(statusMsg, data = {}) {
  const step = parseStepProgress(statusMsg);
  if (step) {
    setUiState(UI_MODES.RUNNING, { stepLabel: step.label, progressPct: step.progressPct });
    setStatus(`Running • ${step.label}`);
  } else {
    const label = runningStatusLabel(statusMsg);
    const secondary = firstPreviewSecondary(data);
    setUiState(UI_MODES.RUNNING, { stepLabel: label, secondaryLabel: secondary });
    setStatus(`Running • ${label}${secondary ? ` • ${secondary}` : ''}`);
  }
}

async function startRateLimitCooldown(retryAfterSeconds, message) {
  if (rateLimitTimer) {
    clearInterval(rateLimitTimer);
    rateLimitTimer = null;
  }
  const totalMs = (Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0 ? Math.ceil(retryAfterSeconds) : 5) * 1000;
  const startMs = Date.now();
  setStatus(message || 'Rate limited. Please retry shortly.');
  setUiState(UI_MODES.RATE_LIMITED, { remaining: Math.ceil(totalMs / 1000), progressPct: 100 });

  rateLimitTimer = setInterval(() => {
    const elapsed = Date.now() - startMs;
    const remainingMs = totalMs - elapsed;
    if (remainingMs <= 0) {
      clearInterval(rateLimitTimer);
      rateLimitTimer = null;
      setStatus('Ready.');
      setUiState(UI_MODES.IDLE);
      return;
    }
    const remaining = Math.ceil(remainingMs / 1000);
    const pct = Math.max(0, (remainingMs / totalMs) * 100);
    setUiState(UI_MODES.RATE_LIMITED, { remaining, progressPct: pct });
  }, 250);
}

async function pollRunState(id) {
  if (terminalStateReached) return;

  try {
    const resp = await fetch(`/run/${id}?after=${encodeURIComponent(lastSnapshotCount)}`, { cache: 'no-store' });
    if (!resp.ok) {
      if (resp.status === 429) {
        let bodyJson = null;
        try {
          bodyJson = await resp.json();
        } catch (_ignored) {}
        const retryAfter = parseRetryAfter(resp, bodyJson, 3);
        await startRateLimitCooldown(retryAfter, bodyJson && bodyJson.message ? bodyJson.message : 'Rate limited while polling.');
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
      setQueuedUi(data.queue_position, data.eta_seconds, data.eta_confidence, data.demand);
      pollDelayMs = 350;
    } else if (data.state === 'running') {
      updateRunningUi(data.status || '', data);
      pollDelayMs = data.snapshot_count > 0 ? 250 : 350;
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
  } catch (_err) {
    // polling exception — will retry on next interval
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

  es.onerror = () => {
    if (terminalStateReached || sessionId !== runSessionId) return;
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
    if (info.state === 'queued') {
      setQueuedUi(info.queue_position, info.eta_seconds, info.eta_confidence, info.demand);
      return;
    }
    updateRunningUi(info.msg, info);
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
  const active = document.activeElement === promptInput ? promptInput : (document.activeElement === contextInput ? contextInput : null);
  applyCharLimits(active);
  const form = new FormData(runForm);
  const runSignature = buildRunSignature(form);
  if (replayCachedRunIfAvailable(runSignature)) {
    return;
  }
  pendingRunSignature = runSignature;
  resetRunState();

  renderImmediate('Queueing request...');
  setUiState(UI_MODES.QUEUED, { queuePosition: null, etaSeconds: null });
  setStatus('Submitting run...');

  try {
    const resp = await fetch('/start', { method: 'POST', body: form });
    if (!resp.ok) {
      const bodyText = await resp.text();
      let bodyJson = null;
      try {
        bodyJson = JSON.parse(bodyText);
      } catch (_ignored) {}

      if (resp.status === 429 || (resp.status === 503 && bodyJson && bodyJson.error === 'queue_full')) {
        const retryAfter = parseRetryAfter(resp, bodyJson, resp.status === 503 ? 8 : 5);
        const msg = bodyJson && bodyJson.message ? bodyJson.message : 'Please retry shortly.';
        pendingRunSignature = '';
        await startRateLimitCooldown(retryAfter, msg);
        return;
      }

      const msg = bodyJson && bodyJson.message ? bodyJson.message : bodyText;
      pendingRunSignature = '';
      setStatus(`Error: ${msg}`);
      setUiState(UI_MODES.ERROR);
      setTimeout(() => setUiState(UI_MODES.IDLE), 800);
      return;
    }

    const payload = await resp.json();
    runId = payload.run_id;
    if (payload.cache_hit) {
      setStatus('Replaying cached result...');
      setUiState(UI_MODES.RUNNING, { stepLabel: 'cached replay' });
      await pollRunState(runId);
      return;
    }
    setQueuedUi(payload.queue_position, payload.eta_seconds, payload.eta_confidence, payload.demand);
    openStream(runId, runSessionId);
  } catch (err) {
    pendingRunSignature = '';
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
    updateTokenDisplay(terminalSqlText);
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
  if (contextInput) {
    contextInput.value = createStmt;
    applyCharLimits(contextInput);
  }
}

// Slider live-value display
function bindSlider(inputId, displayId, fmt) {
  const input = document.getElementById(inputId);
  const display = document.getElementById(displayId);
  if (!input || !display) return;
  const update = () => { display.textContent = fmt ? fmt(input.value) : input.value; };
  update();
  input.addEventListener('input', update);
}

bindSlider('steps',   'stepsVal',  null);
bindSlider('sql_len', 'sqlLenVal', null);
bindSlider('top_k',   'topKVal',   null);
bindSlider('top_p',   'topPVal',   (v) => parseFloat(v).toFixed(2));

const promptCount = document.getElementById('promptCount');
const contextCount = document.getElementById('contextCount');

function computeCharHeuristicLimits() {
  const maxLen = Math.max(8, parseInt(maxLenInput && maxLenInput.value ? maxLenInput.value : '512', 10) || 512);
  const sqlLen = Math.max(1, parseInt(sqlLenInput && sqlLenInput.value ? sqlLenInput.value : '64', 10) || 64);
  const effectiveSqlReserve = Math.max(sqlLen, 16);
  const textTokenBudget = Math.max(32, maxLen - effectiveSqlReserve - STRUCTURE_TOKEN_RESERVE);
  const combinedCharBudget = Math.floor(textTokenBudget * CHARS_PER_TOKEN_EST);
  const promptCharCap = Math.max(120, Math.floor(combinedCharBudget * PROMPT_BUDGET_RATIO));
  const contextCharCap = Math.max(240, combinedCharBudget - promptCharCap);
  return { promptCharCap, contextCharCap, combinedCharBudget };
}

function truncateFieldTail(field, targetLen) {
  if (!field) return;
  const safeTarget = Math.max(0, targetLen);
  const oldVal = field.value || '';
  if (oldVal.length <= safeTarget) return;
  const selectionStart = typeof field.selectionStart === 'number' ? field.selectionStart : oldVal.length;
  const selectionEnd = typeof field.selectionEnd === 'number' ? field.selectionEnd : oldVal.length;
  const nextVal = oldVal.slice(0, safeTarget);
  field.value = nextVal;
  const nextStart = Math.min(selectionStart, nextVal.length);
  const nextEnd = Math.min(selectionEnd, nextVal.length);
  try {
    field.setSelectionRange(nextStart, nextEnd);
  } catch (_ignored) {}
}

function enforceCombinedBudget(activeField, limits) {
  if (!promptInput || !contextInput) return;
  const overflow = (promptInput.value.length + contextInput.value.length) - limits.combinedCharBudget;
  if (overflow <= 0) return;
  if (activeField === promptInput) {
    truncateFieldTail(promptInput, promptInput.value.length - overflow);
    return;
  }
  if (activeField === contextInput) {
    truncateFieldTail(contextInput, contextInput.value.length - overflow);
    return;
  }
  // Non-input caller (e.g. slider change): trim context first, then prompt.
  const cutContext = Math.min(overflow, contextInput.value.length);
  truncateFieldTail(contextInput, contextInput.value.length - cutContext);
  const remaining = (promptInput.value.length + contextInput.value.length) - limits.combinedCharBudget;
  if (remaining > 0) {
    truncateFieldTail(promptInput, promptInput.value.length - remaining);
  }
}

function updateCharCounters(limits) {
  if (promptInput && promptCount) {
    const len = promptInput.value.length;
    promptCount.textContent = `${len} / ${limits.promptCharCap}`;
    promptCount.className = 'char-count' +
      (len >= limits.promptCharCap ? ' at-limit' : len >= limits.promptCharCap * 0.85 ? ' near-limit' : '');
  }
  if (contextInput && contextCount) {
    const len = contextInput.value.length;
    contextCount.textContent = `${len} / ${limits.contextCharCap}`;
    contextCount.className = 'char-count' +
      (len >= limits.contextCharCap ? ' at-limit' : len >= limits.contextCharCap * 0.85 ? ' near-limit' : '');
  }
}

function applyCharLimits(activeField = null) {
  if (!promptInput || !contextInput) return;
  const limits = computeCharHeuristicLimits();
  if (activeField === promptInput) {
    truncateFieldTail(promptInput, limits.promptCharCap);
  } else if (activeField === contextInput) {
    truncateFieldTail(contextInput, limits.contextCharCap);
  } else {
    truncateFieldTail(promptInput, limits.promptCharCap);
    truncateFieldTail(contextInput, limits.contextCharCap);
  }
  enforceCombinedBudget(activeField, limits);
  updateCharCounters(limits);
}

if (promptInput) {
  promptInput.addEventListener('input', () => {
    applyCharLimits(promptInput);
  });
}
if (contextInput) {
  contextInput.addEventListener('input', () => {
    applyCharLimits(contextInput);
  });
}
if (sqlLenInput) {
  sqlLenInput.addEventListener('input', () => {
    const active = document.activeElement === promptInput ? promptInput : (document.activeElement === contextInput ? contextInput : null);
    applyCharLimits(active);
  });
}

applyCharLimits();

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

if (exportGifBtn) {
  exportGifBtn.addEventListener('click', async () => {
    if (exportGifBtn.disabled) return;
    const label = exportGifBtn.innerHTML;
    exportGifBtn.disabled = true;
    exportGifBtn.textContent = 'Building GIF...';
    const resetLabel = () => {
      exportGifBtn.innerHTML = label;
      exportGifBtn.disabled = false;
    };
    try {
      const size = exportGifInBrowser();
      setStatus(`GIF ready (${Math.max(1, Math.round(size / 1024))} KB).`);
      setTimeout(resetLabel, 800);
    } catch (err) {
      exportGifBtn.textContent = 'Export failed';
      setStatus(err && err.message ? err.message : 'GIF export failed.');
      setTimeout(() => { exportGifBtn.innerHTML = label; }, 1600);
      exportGifBtn.disabled = false;
    }
  });
}

fitLiveFont();
setUiState(UI_MODES.IDLE);
setStatus('Ready.');
renderImmediate('Ready for generation. Submit a prompt to start.');

let runId = null;
let es = null;
let historySnapshots = [];
let historyByStep = new Map();
let pollTimer = null;
let pollDelayMs = 250;
let fallbackPolling = false;
let lastSnapshotCount = 0;
let lastStatusMsg = "";
let pendingSnapshots = [];
let playbackTimer = null;
let donePayload = null;
let runInProgress = false;
let finalReplayStarted = false;
let queueEtaBaseline = null;
let rateLimitTimer = null;
let terminalStateReached = false;
let runSessionId = 0;
let terminalSqlText = '';

const stepBox = document.getElementById('stepBox');
const statusBox = document.getElementById('status');
const runBtn = document.getElementById('runBtn');
const runForm = document.getElementById('runForm');
const sliderRow = document.getElementById('sliderRow');
const snapSlider = document.getElementById('snapSlider');
const snapSliderLabel = document.getElementById('snapSliderLabel');
const queueChip = document.getElementById('queueChip');
const liveFront = document.getElementById('liveTextFront');
const liveBack = document.getElementById('liveTextBack');

function setStatus(s){ statusBox.textContent = s; }

function setQueueChip(text) {
  if (!text) {
    queueChip.style.display = "none";
    queueChip.textContent = "";
    return;
  }
  queueChip.style.display = "block";
  queueChip.textContent = text;
}

function ensureBtnProgressNode() {
  let n = runBtn.querySelector('.btn-progress-bar');
  if (!n) {
    n = document.createElement('span');
    n.className = 'btn-progress-bar';
    runBtn.appendChild(n);
  }
  return n;
}

function setButtonVisual(mode, label, remainingPct) {
  runBtn.classList.remove('btn-queued', 'btn-running', 'btn-stopping', 'btn-rate-limited');
  if (mode) runBtn.classList.add(mode);
  runBtn.textContent = label;
  const bar = ensureBtnProgressNode();
  bar.style.width = `${Math.max(0, Math.min(100, remainingPct || 0))}%`;
  bar.style.opacity = mode ? '1' : '0';
}

function setBtnStateIdle() {
  runInProgress = false;
  runBtn.disabled = false;
  setButtonVisual('', 'Run Generation', 0);
  setQueueChip('');
  if (rateLimitTimer) {
    clearInterval(rateLimitTimer);
    rateLimitTimer = null;
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

function updateRunUx(state, queuePos, etaSeconds, statusMsg){
  if (state === 'queued') {
    const posText = Number.isFinite(queuePos) && queuePos > 0 ? `#${queuePos}` : '#1';
    const etaText = Number.isFinite(etaSeconds) && etaSeconds > 0 ? `~${etaSeconds}s` : '~10s';
    let remaining = 100;
    if (Number.isFinite(etaSeconds) && etaSeconds > 0) {
      if (!Number.isFinite(queueEtaBaseline) || queueEtaBaseline <= 0 || etaSeconds > queueEtaBaseline) {
        queueEtaBaseline = etaSeconds;
      }
      remaining = Math.round((etaSeconds / Math.max(queueEtaBaseline, 1)) * 100);
      remaining = Math.max(8, Math.min(100, remaining));
    } else if (Number.isFinite(queuePos) && queuePos > 0) {
      remaining = Math.max(12, 100 - ((queuePos - 1) * 18));
    }
    runInProgress = true;
    runBtn.disabled = true;
    setButtonVisual('btn-queued', `Queued ${posText} ${etaText}`, remaining);
    setQueueChip(`Queued ${posText} ${etaText}`);
    renderImmediate('____ ____ ____ ____');
    return;
  }
  if (state === 'running') {
    runInProgress = true;
    runBtn.disabled = true;
    let pct = 45;
    let label = 'Running…';
    if (statusMsg && statusMsg.startsWith('step ')) {
      const m = statusMsg.match(/step\s+(\d+)\/(\d+)/i);
      if (m) {
        const i = parseInt(m[1], 10);
        const j = parseInt(m[2], 10);
        if (j > 0) pct = Math.round((i / j) * 100);
      }
      label = `${statusMsg}`;
      setQueueChip(`Running (${statusMsg})`);
    } else {
      setQueueChip('Running');
    }
    setButtonVisual('btn-running', label, Math.max(2, 100 - pct));
    return;
  }
  if (state === 'stopping') {
    runInProgress = true;
    runBtn.disabled = true;
    setButtonVisual('btn-stopping', 'Stopping…', 5);
    setQueueChip('Stopping');
    return;
  }
  setBtnStateIdle();
}

function startRateLimitCooldown(retryAfterSeconds, message){
  const total = Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0 ? Math.ceil(retryAfterSeconds) : 10;
  let remaining = total;
  runInProgress = true;
  runBtn.disabled = true;
  setQueueChip('Rate limited');
  setStatus(message || 'Rate limited. Please retry shortly.');
  setButtonVisual('btn-rate-limited', `Rate limited • ${remaining}s`, 100);
  if (rateLimitTimer) clearInterval(rateLimitTimer);
  rateLimitTimer = setInterval(()=>{
    remaining -= 1;
    if (remaining <= 0) {
      clearInterval(rateLimitTimer);
      rateLimitTimer = null;
      setBtnStateIdle();
      setStatus('Ready.');
      return;
    }
    const pct = Math.max(0, Math.round((remaining / total) * 100));
    setButtonVisual('btn-rate-limited', `Rate limited • ${remaining}s`, pct);
  }, 1000);
}

function fitLiveFontForElem(elem) {
  const container = elem.parentElement;
  let width = container.clientWidth - 40;
  if (window.innerWidth <= 600) {
    width = Math.max(180, width * 0.9);
  }
  const lines = (elem.textContent || '').split('\n');
  const longest = lines.reduce((a,b)=> Math.max(a, b.length), 0) || 40;
  const minFont = window.innerWidth < 600 ? 13 : 22;
  const maxFont = window.innerWidth < 600 ? 26 : 88;
  const candidate = Math.floor(Math.max(minFont, Math.min(maxFont, width / Math.max(8, longest) * 2.0)));
  elem.style.fontSize = candidate + 'px';
}

function fitLiveFont(){
  fitLiveFontForElem(liveFront);
  fitLiveFontForElem(liveBack);
}
window.addEventListener('resize', fitLiveFont);

let animating = false;
function animateBlurToText(newText){
  if (animating){
    liveFront.style.transition = '';
    liveBack.style.transition = '';
    liveFront.textContent = newText;
    fitLiveFontForElem(liveFront);
    animating = false;
    return;
  }

  liveBack.textContent = newText || '';
  fitLiveFontForElem(liveBack);
  const blurMax = 4;
  liveBack.style.filter = `blur(${blurMax}px)`;
  liveBack.style.opacity = '0';
  liveBack.style.zIndex = 2;
  liveFront.style.zIndex = 1;

  requestAnimationFrame(()=> {
    liveBack.style.transition = 'filter 240ms cubic-bezier(.2,.9,.2,1), opacity 240ms cubic-bezier(.2,.9,.2,1)';
    liveFront.style.transition = 'opacity 240ms cubic-bezier(.2,.9,.2,1)';
    liveBack.style.filter = 'blur(0px)';
    liveBack.style.opacity = '1';
    liveFront.style.opacity = '0';
    animating = true;

    setTimeout(()=> {
      liveFront.textContent = newText || '';
      fitLiveFontForElem(liveFront);
      liveFront.style.opacity = '1';
      liveFront.style.transition = '';
      liveFront.style.filter = 'none';
      liveFront.style.zIndex = 2;
      liveBack.style.opacity = '0';
      liveBack.style.filter = `blur(${blurMax}px)`;
      liveBack.style.transition = '';
      liveBack.style.zIndex = 1;
      animating = false;
    }, 260);
  });
}

function renderImmediate(text){
  liveFront.textContent = text || '(empty)';
  fitLiveFontForElem(liveFront);
}

function extractSQL(s){
  const start = s.indexOf('<SQL>');
  const end = s.indexOf('</SQL>');
  if(start !== -1 && end !== -1){
    return s.substring(start + 5, end).trim();
  }
  return s;
}

function normalizeSQLDisplay(sql){
  if (!sql) return '(empty)';
  const raw = String(sql).replace(/\r/g, '');
  return raw.trim() ? raw : '(empty)';
}

function snapshotText(snap){
  return normalizeSQLDisplay(snap.sql_only || extractSQL((snap.text || '')));
}

function upsertSnapshot(snap){
  if (!snap) return;
  const step = Number.isFinite(Number(snap.step)) ? Number(snap.step) : historySnapshots.length;
  const existing = historyByStep.get(step);
  if (!existing || JSON.stringify(existing) !== JSON.stringify(snap)) {
    historyByStep.set(step, snap);
    historySnapshots = Array.from(historyByStep.entries()).sort((a,b)=>a[0]-b[0]).map((x)=>x[1]);
  }
}

async function stopRunIfRunning() {
  return;
}

runBtn.onclick = async function(ev) {
  if (runInProgress) ev.preventDefault();
};

function resetRunState(){
  stopTransports();
  if (playbackTimer){ clearInterval(playbackTimer); playbackTimer = null; }
  pollDelayMs = 250;
  lastSnapshotCount = 0;
  lastStatusMsg = '';
  pendingSnapshots = [];
  donePayload = null;
  finalReplayStarted = false;
  terminalStateReached = false;
  runSessionId += 1;
  terminalSqlText = '';
  queueEtaBaseline = null;
  if (rateLimitTimer){ clearInterval(rateLimitTimer); rateLimitTimer = null; }
  historySnapshots = [];
  historyByStep = new Map();
  sliderRow.style.display = 'none';
  setBtnStateIdle();
}

runForm.addEventListener('submit', async (ev)=>{
  ev.preventDefault();
  resetRunState();
  setStatus('Submitting run...');
  renderImmediate('Queueing request...');
  runInProgress = true;
  setButtonVisual('btn-queued', 'Queued #1 ~10s', 100);

  const form = new FormData(ev.target);
  try {
    const resp = await fetch('/start', { method:'POST', body: form });
    if(!resp.ok){
      let bodyText = await resp.text();
      let bodyJson = null;
      try { bodyJson = JSON.parse(bodyText); } catch (_) {}
      if (resp.status === 429) {
        const retryAfterHeader = resp.headers.get('Retry-After');
        const retryAfter = retryAfterHeader ? parseInt(retryAfterHeader, 10) : NaN;
        const msg = (bodyJson && bodyJson.message) ? bodyJson.message : ('HTTP 429: ' + bodyText);
        startRateLimitCooldown(retryAfter, msg);
        return;
      }
      const pretty = bodyJson && bodyJson.message ? bodyJson.message : bodyText;
      setStatus('ERROR: ' + resp.status + ' - ' + pretty);
      setBtnStateIdle();
      return;
    }
    const j = await resp.json();
    runId = j.run_id;
    updateRunUx(j.state, j.queue_position, j.eta_seconds, '');
    setStatus('Run accepted. Streaming updates...');
    openStream(runId, runSessionId);
  } catch(e){
    setStatus('Exception starting run: ' + e.toString());
    setBtnStateIdle();
  }
});

function updateSliderLabel() {
  if (!historySnapshots.length) {
    snapSliderLabel.innerText = '';
    return;
  }
  const idx = parseInt(snapSlider.value, 10);
  const snap = historySnapshots[idx] || {};
  snapSliderLabel.innerText = `Step ${snap.step || (idx+1)} / ${snap.total_steps || historySnapshots.length}`;
}

function applySnapshotAnimated(obj){
  if (!obj) return;
  animateBlurToText(snapshotText(obj));
  stepBox.textContent = `Step ${obj.step} / ${obj.total_steps}`;
}

function applySnapshotImmediate(obj){
  if (!obj) return;
  renderImmediate(snapshotText(obj));
  stepBox.textContent = `Step ${obj.step} / ${obj.total_steps}`;
}

function maybeFinalize(){
  if (donePayload && pendingSnapshots.length === 0){
    finishRun(donePayload);
    donePayload = null;
  }
}

function ensurePlayback(){
  if (playbackTimer) return;
  playbackTimer = setInterval(()=>{
    if (!pendingSnapshots.length){
      clearInterval(playbackTimer);
      playbackTimer = null;
      maybeFinalize();
      return;
    }
    const next = pendingSnapshots.shift();
    applySnapshotAnimated(next);
    maybeFinalize();
  }, 120);
}

function enqueueAnimated(snap){
  if (!snap) return;
  pendingSnapshots.push(snap);
  ensurePlayback();
}

function replayAllOnceAndFinalize(){
  if (finalReplayStarted) return;
  finalReplayStarted = true;
  pendingSnapshots = [];
  renderImmediate('Finalizing replay...');
  for (const snap of historySnapshots) pendingSnapshots.push(snap);
  if (terminalSqlText) {
    const last = historySnapshots.length ? historySnapshots[historySnapshots.length - 1] : { step: 0, total_steps: 0 };
    pendingSnapshots.push({
      ...last,
      sql_only: terminalSqlText,
      text: `<SQL>${terminalSqlText}</SQL>`,
    });
  }
  ensurePlayback();
}

function finishRun(payload){
  stopTransports();
  terminalStateReached = true;
  if (payload && payload.sql_only) {
    terminalSqlText = normalizeSQLDisplay(payload.sql_only);
  }
  if (payload && payload.state === 'error') {
    setStatus(payload.status || 'Run failed.');
    renderImmediate('Run failed. Check status details.');
  } else if (payload && payload.state === 'timed_out') {
    setStatus(payload.status || 'Run timed out.');
    renderImmediate('Run timed out.');
  } else if (payload && payload.state === 'stopped') {
    setStatus(payload.status || 'Run stopped.');
    renderImmediate('Run stopped.');
  } else {
    setStatus(payload && payload.status ? payload.status : 'Run finished.');
  }
  setBtnStateIdle();
  runId = null;
  if (historySnapshots.length > 1) {
    snapSlider.max = String(historySnapshots.length - 1);
    snapSlider.value = snapSlider.max;
    sliderRow.style.display = 'flex';
    updateSliderLabel();
    if (terminalSqlText) {
      renderImmediate(terminalSqlText);
    } else {
      applySnapshotImmediate(historySnapshots[historySnapshots.length - 1]);
    }
  } else if (terminalSqlText) {
    renderImmediate(terminalSqlText);
  }
}

async function pollRunState(id){
  if (terminalStateReached) return;
  try {
    const resp = await fetch('/run/' + id + '?after=' + encodeURIComponent(lastSnapshotCount), { cache: 'no-store' });
    if(!resp.ok){
      if (resp.status === 429) {
        const retryAfterHeader = resp.headers.get('Retry-After');
        const retryAfter = retryAfterHeader ? parseInt(retryAfterHeader, 10) : NaN;
        startRateLimitCooldown(retryAfter, 'Rate limited while polling. Retrying automatically.');
        return;
      }
      setStatus('Polling failed: HTTP ' + resp.status);
      return;
    }
    const data = await resp.json();
    const list = data.snapshots || [];
    for (const s of list) upsertSnapshot(s);

    if (list.length === 1) {
      enqueueAnimated(list[0]);
    } else if (list.length > 1) {
      enqueueAnimated(list[0]);
      enqueueAnimated(list[list.length - 1]);
    }

    if (typeof data.snapshot_count === 'number') {
      lastSnapshotCount = data.snapshot_count;
    } else {
      lastSnapshotCount += list.length;
    }

    updateRunUx(data.state, data.queue_position, data.eta_seconds, data.status || '');
    pollDelayMs = (data.state === 'queued') ? 800 : 250;

    if (data.status && data.status !== lastStatusMsg) {
      lastStatusMsg = data.status;
      setStatus(data.status);
    }

    if (data.done) {
      stopTransports();
      terminalStateReached = true;
      donePayload = data;
      if (data.sql_only) {
        terminalSqlText = normalizeSQLDisplay(data.sql_only);
      }
      replayAllOnceAndFinalize();
      maybeFinalize();
    }
  } catch (e) {
    console.warn('Polling exception', e);
  }
}

function startPollingFallback(id){
  if (fallbackPolling || terminalStateReached) return;
  fallbackPolling = true;
  if (es){ es.close(); es = null; }
  setStatus('SSE unavailable via proxy. Switched to polling fallback.');

  const loop = async ()=>{
    if (!fallbackPolling || terminalStateReached) return;
    await pollRunState(id);
    if (!fallbackPolling || terminalStateReached) return;
    pollTimer = setTimeout(loop, pollDelayMs);
  };
  loop();
}

function openStream(id, sessionId){
  if (es){ es.close(); es = null; }
  es = new EventSource('/stream/' + id);
  es.onopen = ()=> console.log('SSE opened');
  es.onerror = (e)=>{
    if (terminalStateReached || sessionId !== runSessionId) return;
    console.warn('SSE error', e);
    const state = es ? es.readyState : -1;
    setStatus('SSE connection issue (state ' + state + '). Falling back to polling.');
    startPollingFallback(id);
  };
  es.addEventListener('snapshot', (ev)=>{
    if (terminalStateReached || sessionId !== runSessionId) return;
    const obj = JSON.parse(ev.data);
    upsertSnapshot(obj);
    enqueueAnimated(obj);
    lastSnapshotCount = historySnapshots.length;
  });
  es.addEventListener('status', (ev)=>{
    if (terminalStateReached || sessionId !== runSessionId) return;
    const info = JSON.parse(ev.data);
    if (info && info.msg) {
      lastStatusMsg = info.msg;
      setStatus(info.msg);
      updateRunUx('running', null, null, info.msg);
    }
  });
  es.addEventListener('done', (ev)=>{
    if (terminalStateReached || sessionId !== runSessionId) return;
    stopTransports();
    terminalStateReached = true;
    const payload = JSON.parse(ev.data);
    donePayload = payload || {};
    if (payload && payload.sql_only) {
      terminalSqlText = normalizeSQLDisplay(payload.sql_only);
    }
    replayAllOnceAndFinalize();
    maybeFinalize();
  });
}

snapSlider.addEventListener('input', ()=>{
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

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.onclick = function() {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('tab-btn-active'));
    btn.classList.add('tab-btn-active');
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('tab-pane-active'));
    document.getElementById('tab-' + btn.dataset.tab).classList.add('tab-pane-active');
  }
});

document.getElementById('csvInput').addEventListener('change', function(ev){
  const file = ev.target.files[0];
  if(!file) return;
  const reader = new FileReader();
  reader.onload = function(e){
    processCSV(e.target.result);
  };
  reader.readAsText(file);
});

function guessType(val){
  if(/^[\d]+$/.test(val)) return 'INT';
  if(/^[\d\.]+$/.test(val) && val.includes('.')) return 'FLOAT';
  return 'TEXT';
}

function processCSV(text){
  const lines = text.trim().split(/\r?\n/);
  if(!lines.length) return;
  const headers = lines[0].split(',');
  let dataRow = null;
  for(let i=1;i<lines.length;i++){
    if(lines[i].trim()) {
      dataRow = lines[i].split(',');
      break;
    }
  }
  if(!dataRow) dataRow = headers.map(()=> '');
  const types = dataRow.map(guessType);
  const dataRows = lines.slice(1).map(r=>r.split(',')).slice(0,30);
  let html = '<table><thead><tr>' + headers.map(h=>`<th>${h}</th>`).join('') + '</tr></thead><tbody>';
  if(!dataRows.length){html += '<tr><td colspan="'+headers.length+'"><em>No data rows</em></td></tr>';}
  dataRows.forEach(row=>{
    html += '<tr>' + headers.map((_,i)=>`<td>${row[i]!==undefined?row[i]:''}</td>`).join('') + '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('csvTableWrapper').innerHTML = html;

  const colDefs = headers.map((h,i)=>`  ${h} ${types[i]||'TEXT'}`).join(',\n');
  const create = `CREATE TABLE table_name (\n${colDefs}\n);`;
  const context = document.getElementById('context');
  if(context) context.value = create;
}

fitLiveFont();
setBtnStateIdle();
renderImmediate('Ready for generation. Submit a prompt to start.');

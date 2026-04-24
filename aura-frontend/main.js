'use strict';
/* ════════════════════════════════════════════════════════
   Aura · main.js  — Supabase Realtime + LiveKit edition
   ════════════════════════════════════════════════════════
   Communication model:
     • Backend → Supabase DB INSERT → Supabase Realtime → here (final_verdict)
     • Browser Mic → LiveKit room → Bot → Signal Processor → Backend
   ════════════════════════════════════════════════════════ */

// ── Configuration ─────────────────────────────────────────────────────────────
let BACKEND_URL = (window.ENV_BACKEND_URL || '').replace(/\/$/, '');
const SUPABASE_URL = (window.ENV_SUPABASE_URL || '');
const SUPABASE_ANON_KEY = (window.ENV_SUPABASE_ANON_KEY || '');

window._configValid = true;

function showPersistentBanner(msg) {
  const banner = document.createElement('div');
  banner.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#EF476F;color:#fff;text-align:center;padding:8px;font-size:12px;z-index:9999;font-weight:bold;font-family:"Space Mono",monospace;';
  banner.textContent = msg;
  document.body.prepend(banner);
}

function validateConfig() {
  const status = { BACKEND_URL: 'OK', SUPABASE_URL: 'OK', SUPABASE_ANON_KEY: 'OK' };
  
  if (!SUPABASE_URL || SUPABASE_URL.includes('__SUPABASE_URL__')) {
    status.SUPABASE_URL = 'MISSING';
    window._configValid = false;
  }
  if (!SUPABASE_ANON_KEY || SUPABASE_ANON_KEY.includes('__SUPABASE_ANON_KEY__')) {
    status.SUPABASE_ANON_KEY = 'MISSING';
    window._configValid = false;
  }
  if (!BACKEND_URL || BACKEND_URL.includes('__BACKEND_URL__')) {
    status.BACKEND_URL = 'MISSING/PLACEHOLDER';
    BACKEND_URL = 'http://localhost:8000';
  }

  console.groupCollapsed('[Aura:Config] Environment Validation');
  console.table(status);
  console.groupEnd();

  if (!window._configValid) {
    showPersistentBanner('⚠️ Configuration missing – Supabase credentials not set. Demo mode only.');
  } else if (status.BACKEND_URL !== 'OK') {
    showPersistentBanner('⚠️ Backend URL placeholder detected. Run inject-env.js. Defaulting to localhost:8000.');
  } else if (BACKEND_URL === 'http://localhost:8000' && window.location.hostname !== 'localhost') {
    console.warn('[Aura:Config] Backend URL is localhost but served from non-localhost origin.');
  }
  return status;
}
validateConfig();

// ── Emotion helpers (delegate to unified palette) ─────────────────────────────
/**
 * Get the full emotion definition. Falls back to Neutral.
 * @param {string} name
 */
function em(name) {
  return window.EMOTION_PALETTE ? window.EMOTION_PALETTE.getEmotion(name)
    : { r:160, g:170, b:181, freq:.50, amp:.040, base:.27, glow:.40 };
}
/**
 * Returns true for sarcasm-family emotions.
 * @param {string} name
 * @returns {boolean}
 */
function isSarcastic(name) {
  return window.EMOTION_PALETTE ? window.EMOTION_PALETTE.isSarcastic(name)
    : name === 'Sarcastic' || name === 'Passive-Aggressive';
}

// ── Mutable State ─────────────────────────────────────────────────────────────
const S = {
  emotion: 'Neutral',
  lastEmotion: null, lastEmotionTime: 0,
  cur: [160,170,181], tgt: [160,170,181],
  curR: 0, tgtR: 0,
  curGlow: 0.4, tgtGlow: 0.4,
  phase: 0,
  jx: 0, jy: 0, jTimer: 0,
  slowSession: null, awaitVerdict: false, interrupted: false,
};

// ── Session Control State ────────────────────────────────────────────────────
/** Whether the session is currently paused by the user. */
let _isPaused = false;
/** Timer for the double-confirm cancel pattern. */
let _cancelConfirmTimer = null;
/** False = backend reachable; true = offline; null = untested yet. */
window._isBackendOffline = null;

// ── Supabase state ────────────────────────────────────────────────────────────
let supabaseClient = null;
let supabaseChannel = null;
let currentSessionId = null;

if (!window._verdictHistory) window._verdictHistory = [];

// ── Canvas (legacy 2D aura — kept alive for compatibility) ────────────────────
const canvas = document.getElementById('aura-canvas');
const ctx    = canvas ? canvas.getContext('2d') : null;
let W = 0, H = 0, prevT = 0;

function resize() {
  if (!canvas) return;
  W = canvas.width  = window.innerWidth;
  H = canvas.height = window.innerHeight;
}

function lerp(a, b, t) { return a + (b - a) * t; }

const LERP = 0.065;

/**
 * Set the visual target for the 2D aura canvas and the particle orb.
 * @param {string} name  Emotion label.
 */
function setEmotionTarget(name) {
  const e = em(name);
  S.tgt     = [e.r, e.g, e.b];
  S.tgtGlow = e.glow;
  // Also drive the particle orb
  if (window.ParticleAura) window.ParticleAura.setEmotion(name);
}

/**
 * Main 2D canvas draw loop — LERP-based colour and radius animation.
 * Pulse frequency and amplitude are emotion-specific.
 * @param {number} ts  Timestamp from requestAnimationFrame.
 */
function draw(ts) {
  if (!ctx) { requestAnimationFrame(draw); return; }
  const dt = Math.min((ts - prevT) / 1000, 0.1);
  prevT = ts;
  const e = em(S.emotion);

  S.cur[0] = lerp(S.cur[0], S.tgt[0], LERP);
  S.cur[1] = lerp(S.cur[1], S.tgt[1], LERP);
  S.cur[2] = lerp(S.cur[2], S.tgt[2], LERP);
  S.curGlow = lerp(S.curGlow, S.tgtGlow, LERP);

  S.phase += e.freq * dt * Math.PI * 2;
  let mod = 1 + e.amp * Math.sin(S.phase);
  if (S.emotion === 'Surprised') mod = 1 + e.amp * Math.abs(Math.sin(S.phase));

  if (['Frustrated','Agitated','Angry'].includes(S.emotion)) {
    S.jTimer -= dt;
    if (S.jTimer <= 0) {
      const minD = Math.min(W, H);
      S.jx = (Math.random() - .5) * minD * .014;
      S.jy = (Math.random() - .5) * minD * .014;
      S.jTimer = .04 + Math.random() * .06;
    }
  } else {
    S.jx = lerp(S.jx, 0, .18);
    S.jy = lerp(S.jy, 0, .18);
  }

  const minD = Math.min(W, H);
  S.tgtR  = e.base * minD * mod;
  S.curR  = lerp(S.curR, S.tgtR, LERP * 2);

  const [r, g, b] = S.cur;
  const gl = S.curGlow;
  const cx = W / 2 + S.jx, cy = H / 2 + S.jy;

  ctx.clearRect(0, 0, W, H);

  const layers = [
    { r: S.curR * 2.8, a0: gl * .16, a1: gl * .06 },
    { r: S.curR * 1.7, a0: gl * .44, a1: gl * .16 },
    { r: S.curR,       a0: gl * .92, a1: 0         },
  ];
  for (const l of layers) {
    const g2 = ctx.createRadialGradient(cx, cy, 0, cx, cy, l.r);
    if (l === layers[2]) {
      g2.addColorStop(0,   `rgba(255,255,255,${(gl * .55).toFixed(3)})`);
      g2.addColorStop(.18, `rgba(${r|0},${g|0},${b|0},${l.a0.toFixed(3)})`);
      g2.addColorStop(1,   `rgba(${r|0},${g|0},${b|0},0)`);
    } else {
      g2.addColorStop(0, `rgba(${r|0},${g|0},${b|0},${l.a0.toFixed(3)})`);
      g2.addColorStop(1, `rgba(${r|0},${g|0},${b|0},0)`);
    }
    ctx.fillStyle = g2;
    ctx.fillRect(0, 0, W, H);
  }
  requestAnimationFrame(draw);
}

// ── Momentum Smoother ─────────────────────────────────────────────────────────
const MOMENTUM_MS = 1000;

/**
 * Require the same emotion twice within MOMENTUM_MS to commit it visually.
 * This prevents single-frame false positives from the fast-path classifier.
 * @param {string} emotion
 */
function feedMomentum(emotion) {
  const now = Date.now();
  if (emotion === S.lastEmotion && (now - S.lastEmotionTime) <= MOMENTUM_MS) {
    _commitEmotion(emotion);
    S.lastEmotion = null;
    S.lastEmotionTime = 0;
  } else {
    S.lastEmotion = emotion;
    S.lastEmotionTime = now;
  }
}

/**
 * Commit an emotion to the visual state immediately.
 * @param {string} name
 */
function _commitEmotion(name) {
  if (name === S.emotion) return;
  S.emotion = name;
  setEmotionTarget(name);
}

// ── Progress Ring ─────────────────────────────────────────────────────────────
const ringWrap    = document.getElementById('ring-wrap');
const ringArc     = document.getElementById('ring-arc');
const ringCountEl = document.getElementById('ring-countdown');
const CIRC        = 263.89;
const RING_DUR_MS  = 1500;
const RING_FADE_MS = 200;
let ringRafId = null, ringFadeTimer = null;

/**
 * Show the analysis progress ring with a countdown (#7).
 */
function showRing() {
  if (ringFadeTimer) { clearTimeout(ringFadeTimer); ringFadeTimer = null; }
  ringWrap.classList.add('visible');
  ringArc.style.strokeDashoffset = CIRC;

  const t0 = performance.now();
  function step(now) {
    const p = Math.min((now - t0) / RING_DUR_MS, 1);
    ringArc.style.strokeDashoffset = CIRC * (1 - p);
    // Countdown label (#7)
    if (ringCountEl) {
      ringCountEl.textContent = p < 1 ? ((1 - p) * RING_DUR_MS / 1000).toFixed(1) + 's' : '';
    }
    if (p < 1) ringRafId = requestAnimationFrame(step);
  }
  if (ringRafId) cancelAnimationFrame(ringRafId);
  ringRafId = requestAnimationFrame(step);
}

/**
 * Hide the analysis progress ring.
 * @param {boolean} [immediate]  If true, skip fade transition.
 */
function hideRing(immediate) {
  if (ringRafId) { cancelAnimationFrame(ringRafId); ringRafId = null; }
  if (ringCountEl) ringCountEl.textContent = '';
  if (immediate) { ringWrap.classList.remove('visible'); return; }
  ringWrap.style.transition = `opacity ${RING_FADE_MS}ms ease, visibility ${RING_FADE_MS}ms ease`;
  ringWrap.classList.remove('visible');
  ringFadeTimer = setTimeout(() => {
    ringWrap.style.transition = '';
    ringFadeTimer = null;
  }, RING_FADE_MS);
}

// ── Card Manager ──────────────────────────────────────────────────────────────
const cardsEl = document.getElementById('cards');

/**
 * Render a final verdict as a glassmorphic emotion card.
 * Stores the verdict in sessionStorage for the result page.
 * @param {{ transcript:string, final_emotion:string, confidence:number, reasoning:string }} data
 */
function showCard(data) {
  const { transcript, final_emotion, confidence, reasoning } = data;
  const e   = em(final_emotion);
  const pct = Math.round((confidence || 0) * 100);
  const col = `rgb(${e.r},${e.g},${e.b})`;

  window._verdictHistory.push({ ...data, ts: Date.now() });
  try {
    sessionStorage.setItem('aura_verdicts', JSON.stringify(window._verdictHistory));
    if (currentSessionId) sessionStorage.setItem('aura_session_id', currentSessionId);
  } catch (_) {}

  cardsEl.querySelectorAll('.card').forEach(c => {
    c.classList.add('card--exit');
    setTimeout(() => c.remove(), 210);
  });

  const card = document.createElement('article');
  card.className = 'card' + (isSarcastic(final_emotion) ? ' card--sarcastic' : '');
  card.setAttribute('aria-label', `Emotion: ${final_emotion}`);

  const reportUrl = currentSessionId
    ? `result.html?session=${currentSessionId}`
    : 'result.html?session=demo';

  card.innerHTML = `
    <div class="card-head">
      <span class="pill" style="background:${col};box-shadow:0 0 12px rgba(${e.r},${e.g},${e.b},.42)">
        <span class="pill-dot"></span>${final_emotion}
      </span>
      <div class="conf-track" title="${pct}% confidence">
        <div class="conf-fill" style="width:${pct}%;background:linear-gradient(90deg,rgba(${e.r},${e.g},${e.b},.55),rgba(${e.r},${e.g},${e.b},1))"></div>
      </div>
    </div>
    ${transcript ? `<p class="transcript">"${transcript}"</p>` : ''}
    ${reasoning  ? `<p class="reasoning">${reasoning}</p>`    : ''}
    <div class="card-foot">
      <span>Confidence</span>
      <span>${pct}%</span>
    </div>
    <a href="${reportUrl}" class="report-btn" aria-label="View full analysis report">VIEW REPORT →</a>`;

  setTimeout(() => cardsEl.appendChild(card), 220);
}

function clearCards() {
  cardsEl.querySelectorAll('.card').forEach(c => {
    c.classList.add('card--exit');
    setTimeout(() => c.remove(), 210);
  });
}

// ── Session Controls (Pause / Resume / Cancel) ────────────────────────────────

/**
 * Pause the session: mute mic, silence orb, halt event processing.
 */
function pauseSession() {
  if (_isPaused) return;
  _isPaused = true;
  if (typeof window.muteMicrophone === 'function') window.muteMicrophone(true);
  if (window.ParticleAura) window.ParticleAura.setVolume(0);
  const icon  = document.getElementById('pause-icon');
  const label = document.getElementById('pause-label');
  const btn   = document.getElementById('btn-pause');
  if (icon)  icon.textContent  = '▶';
  if (label) label.textContent = 'Resume';
  if (btn)   btn.setAttribute('aria-pressed', 'true');
  showToast('⏸ Session paused — mic muted', 2200);
}

/**
 * Resume a paused session: unmute mic and re-enable event processing.
 */
function resumeSession() {
  if (!_isPaused) return;
  _isPaused = false;
  if (typeof window.muteMicrophone === 'function') window.muteMicrophone(false);
  const icon  = document.getElementById('pause-icon');
  const label = document.getElementById('pause-label');
  const btn   = document.getElementById('btn-pause');
  if (icon)  icon.textContent  = '⏸';
  if (label) label.textContent = 'Pause';
  if (btn)   btn.setAttribute('aria-pressed', 'false');
  showToast('▶ Session resumed', 2200);
}

/** Toggle between paused and resumed states. */
function togglePause() {
  _isPaused ? resumeSession() : pauseSession();
}

/**
 * Full session teardown — disconnect LiveKit, reset state, return to hero.
 * Called on second cancel click.
 */
async function cancelSession() {
  _isPaused = false;
  hideRing(true);
  showListening(false);
  clearCards();
  setConn('off');
  if (window.ParticleAura) { window.ParticleAura.setEmotion('Neutral'); window.ParticleAura.setVolume(0); }
  if (typeof window.disconnectFromLiveKit === 'function') {
    try { await window.disconnectFromLiveKit(); } catch (_) {}
  }
  showToast('Session cancelled', 1800);
  setTimeout(() => {
    // Return to hero — reload clears all timers and state cleanly
    location.href = location.pathname;
  }, 1900);
}

/**
 * Handle cancel button click using a double-confirm pattern.
 * First click: button turns red and shows "Confirm?". Second click within 3s: cancel.
 */
function handleCancelClick() {
  const btn = document.getElementById('btn-cancel');
  if (!btn) { cancelSession(); return; }

  if (btn.dataset.confirming === 'true') {
    // Second click — execute
    clearTimeout(_cancelConfirmTimer);
    btn.dataset.confirming = 'false';
    btn.classList.remove('confirming');
    cancelSession();
  } else {
    // First click — enter confirming state
    btn.dataset.confirming = 'true';
    btn.classList.add('confirming');
    document.getElementById('cancel-label').textContent = 'Confirm?';
    _cancelConfirmTimer = setTimeout(() => {
      btn.dataset.confirming = 'false';
      btn.classList.remove('confirming');
      document.getElementById('cancel-label').textContent = 'Cancel';
    }, 3000);
  }
}

// ── Utterance Finalization & Silence Detection ────────────────────────────────

let _lastActiveTime = Date.now();
let _hasSpokenThisUtterance = false;
const SILENCE_THRESHOLD = 0.05;
const SILENCE_DUR_MS = 600;
const TIMEOUT_DUR_MS = 10000;
let _ringTimeoutTimer = null;

let _volLogTimer = 0;
function feedVolume(avg) {
  if (!S.slowSession || _isPaused) return;

  const now = Date.now();
  if (now - _volLogTimer > 2000) {
    // console.log(`[Aura:Silence] Current avg volume: ${avg.toFixed(3)}`);
    _volLogTimer = now;
  }

  const silenceBar = document.getElementById('silence-bar');
  if (S.awaitVerdict) {
    if (silenceBar) silenceBar.style.width = '0%';
    return;
  }

  if (avg > SILENCE_THRESHOLD) {
    _lastActiveTime = Date.now();
    _hasSpokenThisUtterance = true;
    if (silenceBar) silenceBar.style.width = '0%';
  } else if (_hasSpokenThisUtterance) {
    const silentFor = Date.now() - _lastActiveTime;
    const progress = Math.min((silentFor / SILENCE_DUR_MS) * 100, 100);
    if (silenceBar) silenceBar.style.width = progress + '%';

    if (silentFor > SILENCE_DUR_MS) {
      console.log('[Aura:Silence] Triggering progress ring.');
      S.awaitVerdict = true;
      _hasSpokenThisUtterance = false;
      if (silenceBar) silenceBar.style.width = '0%';
      showRing();
      
      clearTimeout(_ringTimeoutTimer);
      _ringTimeoutTimer = setTimeout(() => {
        if (S.awaitVerdict) {
          S.awaitVerdict = false;
          hideRing();
          showToast('Analysis timeout', 3000);
        }
      }, TIMEOUT_DUR_MS);
    }
  }
}

function handleDoneClick() {
  if (S.awaitVerdict) {
    showToast('Already processing', 2000);
    return;
  }
  const btn = document.getElementById('btn-done');
  if (btn) btn.disabled = true;

  console.log('[main.js] Manual done clicked.');
  S.awaitVerdict = true;
  _hasSpokenThisUtterance = false;
  showRing();
  
  if (currentSessionId) {
    fetch(`${BACKEND_URL}/session/end`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: currentSessionId })
    })
      .then(res => {
        if (!res.ok) console.warn('[Aura:Backend] session/end returned', res.status);
      })
      .catch(e => console.warn('Finalize endpoint failed (fallback active):', e));
  }
  
  setTimeout(() => { if (btn) btn.disabled = false; }, 3000);
}

// ── Status & Toast ────────────────────────────────────────────────────────────
const connDot    = document.getElementById('conn-dot');
const connTxt    = document.getElementById('conn-text');
const connToggle = document.getElementById('conn-toggle');
const listenEl   = document.getElementById('listening');
const toastEl    = document.getElementById('toast');
let toastTimer = null, listenTimer = null;

// Click the status bar to manually reconnect when disconnected
document.getElementById('status-bar')?.addEventListener('click', () => {
  if (_isPaused) { resumeSession(); return; }
  if (window._isBackendOffline !== true && typeof startAudioCapture === 'function') {
    if (connTxt && connTxt.textContent === 'Disconnected') {
      showToast('↺ Reconnecting…', 2200);
      startAudioCapture();
    }
  }
});

let _connState = 'DISCONNECTED';
function setConn(mode) {
  // Legacy mode mapping to new states
  if (mode === 'ok') _connState = 'CONNECTED';
  else if (mode === 'warn') _connState = 'CONNECTING';
  else if (mode === 'off') _connState = 'DISCONNECTED';
  else _connState = mode;

  const map = {
    DISCONNECTED: ['dot--off',  'Disconnected', ''],
    CONNECTING:   ['dot--warn', 'Connecting…', 'warn'],
    CONNECTED:    ['dot--ok',   'Live', 'on'],
    ERROR:        ['dot--off',  'Backend Error', ''],
    MOCK:         ['dot--warn', 'Demo Mode', 'warn']
  };
  const [cls, txt, tgl] = map[_connState] || map.DISCONNECTED;
  
  if (connDot) connDot.className = 'dot ' + cls;
  if (connTxt) connTxt.textContent = txt;
  if (connToggle) {
    connToggle.classList.remove('on', 'warn');
    if (tgl) connToggle.classList.add(tgl);
  }
}

async function checkBackendHealth() {
  if (window.location.search.includes('mock=true') || !window._configValid) {
    console.log('[Aura:Backend] Mock parameter or invalid config detected.');
    startMockSequence();
    return;
  }
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { method: 'GET' });
    if (res.ok) {
      console.log('[Aura:Backend] Health check OK');
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (e) {
    console.warn('[Aura:Backend] Health check failed:', e.message);
    setConn('ERROR');
    window._isBackendOffline = true;
    startMockSequence();
  }
}

function startMockSequence() {
  console.log('[Aura:Mock] Starting mock sequence');
  setConn('MOCK');
  showListening(true);
  S.awaitVerdict = false;
  S.slowSession = 'demo';
  currentSessionId = 'demo';
  
  if (!window._mockTimer) {
    window._mockTimer = setInterval(() => {
      if (_isPaused) return;
      const emotions = ['Happy', 'Sad', 'Neutral', 'Surprised', 'Frustrated'];
      const em = emotions[Math.floor(Math.random() * emotions.length)];
      showCard({
        transcript: 'This is a simulated utterance since the backend or Supabase is unavailable.',
        final_emotion: em,
        confidence: 0.85 + Math.random() * 0.1,
        reasoning: 'Demo mode activated. Backend not reached.'
      });
      _commitEmotion(em);
    }, 12000);
  }
}

function showListening(on) {
  if (!listenEl) return;
  if (on) { listenEl.hidden = false; listenEl.style.opacity = '1'; }
  else {
    listenEl.style.opacity = '0';
    clearTimeout(listenTimer);
    listenTimer = setTimeout(() => { listenEl.hidden = true; listenEl.style.opacity = ''; }, 220);
  }
}

/**
 * Show a non-blocking toast notification.
 * @param {string} msg   Message text.
 * @param {number} [ms]  Auto-dismiss duration (default 4500ms).
 */
function showToast(msg, ms = 4500) {
  if (!toastEl) return;
  toastEl.hidden = false;
  toastEl.textContent = msg;
  requestAnimationFrame(() => toastEl.classList.add('show'));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.classList.remove('show');
    setTimeout(() => { toastEl.hidden = true; }, 220);
  }, ms);
}

// ── Supabase Integration ──────────────────────────────────────────────────────

/**
 * Initialise the Supabase client from config constants.
 * @returns {boolean} true if credentials are available.
 */
function initSupabase() {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY || SUPABASE_URL.includes('__') || SUPABASE_ANON_KEY.includes('__')) {
    console.error('[Aura:Supabase] Credentials missing or invalid. Realtime disabled.');
    supabaseClient = null;
    return false;
  }
  if (!/^https:\/\/[a-z0-9-]+\.supabase\.co$/.test(SUPABASE_URL)) {
    console.error('[Aura:Supabase] Invalid URL format.');
    supabaseClient = null;
    return false;
  }
  try {
    supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
    window._supabaseReady = true;
    console.log('[Aura:Supabase] Initialized successfully.');
    return true;
  } catch (err) {
    console.error('[Aura:Supabase] init failed:', err);
    supabaseClient = null;
    return false;
  }
}

/**
 * Subscribe to Supabase Postgres Realtime for final verdicts on a session.
 * Inserts on emotion_sessions trigger showCard() and emotion transitions.
 * @param {string} sessionId
 */
let _supabaseRetryCount = 0;
let _verdictTimeoutTimer = null;
let _verdictPollingTimer = null;

function subscribeToVerdicts(sessionId) {
  if (!supabaseClient) {
    console.warn('[Aura:Supabase] Not initialized, falling back to mock.');
    setTimeout(startMockSequence, 1000);
    return;
  }
  
  if (supabaseChannel) { 
    console.log('[Aura:Supabase] Cleaning up previous channel.');
    supabaseClient.removeChannel(supabaseChannel); 
    supabaseChannel = null; 
  }
  clearTimeout(_verdictTimeoutTimer);
  clearInterval(_verdictPollingTimer);

  console.log(`[Aura:Supabase] Starting subscription for session: ${sessionId}`);
  
  // FALLBACK: Start polling after 5 seconds if Realtime is quiet
  _verdictPollingTimer = setInterval(async () => {
    if (!S.awaitVerdict) {
      clearInterval(_verdictPollingTimer);
      return;
    }
    console.log('[Aura:Supabase] Polling for verdict fallback...');
    try {
      const { data, error } = await supabaseClient
        .from('emotion_sessions')
        .select('*')
        .eq('session_id', sessionId)
        .maybeSingle();

      if (data && data.final_emotion) {
        console.log('[Aura:Supabase] Polling found verdict!');
        clearInterval(_verdictPollingTimer);
        _verdictPollingTimer = null; // Prevent double trigger
        clearTimeout(_verdictTimeoutTimer);
        processVerdict(data, sessionId);
      }
    } catch (e) {
      console.warn('[Aura:Supabase] Polling error:', e);
    }
  }, 3000);

  _verdictTimeoutTimer = setTimeout(() => {
    if (S.awaitVerdict) {
      console.warn(`[Aura:Supabase] Timeout reached. No matching row found for: ${sessionId}`);
      S.awaitVerdict = false;
      clearInterval(_verdictPollingTimer);
      _verdictPollingTimer = null;
      hideRing();
      showToast('⚠️ Verdict took too long. Check reports page.', 6000);
      setConn('ERROR');
    }
  }, 45000);

  try {
    supabaseChannel = supabaseClient
      .channel('db-changes')
      .on('postgres_changes', 
        { event: '*', schema: 'public', table: 'emotion_sessions' }, 
        (payload) => {
          console.log('[Aura:Supabase] REALTIME RECEIVED:', payload);
          const row = payload.new;
          if (row && row.session_id === sessionId && S.awaitVerdict) {
            console.log('[Aura:Supabase] MATCH FOUND VIA REALTIME!');
            clearInterval(_verdictPollingTimer);
            _verdictPollingTimer = null;
            clearTimeout(_verdictTimeoutTimer);
            processVerdict(row, sessionId);
          }
        }
      )
      .subscribe((status, err) => {
        console.log('[Aura:Supabase] Subscription Status:', status);
        if (status === 'SUBSCRIBED') {
          setConn('CONNECTED');
        }
      });
  } catch (err) {
    console.error('[Aura:Supabase] Subscription crash:', err);
  }
}

function processVerdict(row, sessionId) {
  // Clear any persistent processing toasts
  const toast = document.getElementById('toast');
  if (toast && toast.textContent.includes('Processing')) {
    toast.classList.remove('show');
    setTimeout(() => { toast.hidden = true; }, 220);
  }

  const verdict = {
    session_id:    row.session_id,
    transcript:    row.utterance_text || '',
    final_emotion: row.final_emotion  || 'Neutral',
    confidence:    row.confidence     || 0,
    reasoning:     row.judge_reasoning || '',
  };
  
  if (!isSarcastic(verdict.final_emotion)) _commitEmotion(verdict.final_emotion);
  S.awaitVerdict = false;
  hideRing(false);
  showListening(false);
  showCard(verdict);
}

// ── Audio Capture via LiveKit ─────────────────────────────────────────────────

/**
 * Start a new emotion detection session:
 *  1. POST /session/start → get session_id + LiveKit token
 *  2. Subscribe to Supabase Realtime for that session
 *  3. Connect mic to LiveKit room
 */
async function startAudioCapture() {
  if (_connState === 'MOCK' || window.location.search.includes('mock=true')) {
    startMockSequence();
    return;
  }

  console.log(`[Aura:Backend] startAudioCapture() called. Fetching: ${BACKEND_URL}/session/start`);
  try {
    setConn('CONNECTING');
    const response = await fetch(`${BACKEND_URL}/session/start`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({}),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    console.log('[Aura:Backend] Session started:', data.session_id, '| LiveKit URL:', data.livekit_url);
    if (!data.livekit_url) {
      throw new Error('livekit_url missing in response');
    }

    currentSessionId = data.session_id;
    window._isBackendOffline = false;

    // Subscribe to Supabase Realtime BEFORE connecting mic so we don't miss verdicts
    subscribeToVerdicts(currentSessionId);

    if (typeof window.connectToLiveKit === 'function') {
      // Reuse stream obtained during the mic-test step
      const ok = await window.connectToLiveKit(data.livekit_token, data.livekit_url, window._auraMicStream || null);
      if (!ok) {
        showToast('⚠️ Microphone connection failed — check permissions and refresh', 7000);
        setConn('ERROR');
      } else {
        setConn('CONNECTED');
      }
    } else {
      console.error('[Aura:LiveKit] window.connectToLiveKit is not defined.');
    }

    showListening(true);
    S.awaitVerdict = false;
    S.slowSession  = currentSessionId;

  } catch (err) {
    console.error('[Aura:Backend] Backend unavailable:', err.message);
    window._isBackendOffline = true;
    setConn('ERROR');
    
    let reason = 'Backend offline';
    if (err.message.includes('HTTP')) reason = 'Backend endpoint error';
    if (err.message.includes('livekit_url')) reason = 'Backend misconfigured (No LiveKit URL)';
    
    showToast(`⚠️ ${reason} - Entering Demo Mode`, 5000);
    setTimeout(startMockSequence, 2000);
  }
}

// Expose for bootstrap module trigger and livekit.js retry
window._auraRetryAudioCapture = startAudioCapture;
window._auraStartCapture      = startAudioCapture;

// ── Replay Session ────────────────────────────────────────────────────────────

/**
 * Replay a session's emotion timeline on the particle orb.
 * Fetches real timeline data from the backend.
 * Mic input is disabled during replay.
 * @param {string} sessionId  The session ID to replay.
 */
async function startReplaySession(sessionId) {
  showToast('▶ Replaying session…', 60000);
  showListening(true);
  if (listenEl) {
    const span = listenEl.querySelector('span:last-child');
    if (span) span.textContent = 'Replaying session…';
  }

  let timeline = null;
  try {
    const resp = await fetch(`${BACKEND_URL}/session/${sessionId}/report`);
    if (resp.ok) timeline = (await resp.json()).timeline;
  } catch (_) {}

  if (!timeline || !timeline.length) {
    showToast('⚠️ Could not load replay data.', 4000);
    showListening(false);
    return;
  }

  const totalTime = (timeline[timeline.length - 1]?.time || 60) * 1000;
  timeline.forEach(point => {
    const emotions = Object.entries(point).filter(([k]) => k !== 'time');
    const dominant = emotions.sort((a, b) => b[1] - a[1])[0];
    if (!dominant) return;
    const delay = ((point.time || 0) / (timeline[timeline.length - 1]?.time || 60)) * totalTime;
    setTimeout(() => setEmotionTarget(dominant[0]), delay);
  });

  setTimeout(() => {
    showListening(false);
    showToast('Replay complete.', 2500);
  }, totalTime + 500);
}

// ── File Upload ───────────────────────────────────────────────────────────────

const MAX_FILE_SIZE_MB = 10;
const ALLOWED_EXTENSIONS = ['wav', 'mp3', 'm4a', 'ogg', 'flac'];

async function handleFileUpload(file) {
  // Validate file size
  if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
    showToast(`⚠️ File too large. Max size is ${MAX_FILE_SIZE_MB}MB.`, 5000);
    return;
  }

  // Validate extension
  const ext = file.name.split('.').pop().toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    showToast(`⚠️ Invalid file format. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`, 5000);
    return;
  }

  const uploadBtn = document.getElementById('ob-upload-btn');
  if (uploadBtn) {
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<span class="btn-icon">⏳</span> Uploading...';
  }

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${BACKEND_URL}/session/analyze-file`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const sessionId = data.session_id;

    if (!sessionId) {
      throw new Error('No session ID returned from backend.');
    }

    // Success! Enter listening state
    if (typeof window._auraEnterListeningState === 'function') {
      window._auraEnterListeningState();
    }
    
    currentSessionId = sessionId;
    S.awaitVerdict = true;
    S.slowSession = sessionId;
    
    // Subscribe to realtime verdict
    subscribeToVerdicts(sessionId);
    
    // Show UI progress ring
    showRing();
    showToast('Processing audio file...', 30000); // cleared when card shows
    
    // Set a 30s timeout for processing
    clearTimeout(_ringTimeoutTimer);
    _ringTimeoutTimer = setTimeout(() => {
      if (S.awaitVerdict) {
        S.awaitVerdict = false;
        hideRing();
        showToast('⚠️ Processing took too long.', 5000);
        setConn('ERROR');
      }
    }, 30000);

  } catch (err) {
    console.error('[Aura:Upload] File upload failed:', err);
    showToast(`⚠️ Upload failed: ${err.message}`, 5000);
  } finally {
    if (uploadBtn) {
      uploadBtn.disabled = false;
      uploadBtn.innerHTML = '<span class="btn-icon">📁</span> Upload Audio File';
    }
  }
}

window._auraHandleFileUpload = handleFileUpload;

// ── Onboarding ────────────────────────────────────────────────────────────────
// The #ob-btn click handler is registered in the bootstrap module in index.html.
function initOnboarding() {}

// ── Boot ──────────────────────────────────────────────────────────────────────

function boot() {
  try {
    resize();
    window.addEventListener('resize', resize);

    // Expose cross-module helpers
    window._aura = {
      setEmotionTarget, showCard, showRing,
      hideRing, setConn, showToast, supabase: () => supabaseClient,
      togglePause, cancelSession, handleCancelClick, feedVolume,
    };

    // Wire session control buttons
    const btnPause  = document.getElementById('btn-pause');
    const btnDone   = document.getElementById('btn-done');
    const btnCancel = document.getElementById('btn-cancel');
    if (btnPause)  btnPause.addEventListener('click',  togglePause);
    if (btnDone)   btnDone.addEventListener('click',   handleDoneClick);
    if (btnCancel) btnCancel.addEventListener('click', handleCancelClick);

    setEmotionTarget('Neutral');
    if (ctx) requestAnimationFrame(t => { prevT = t; draw(t); });

    // Detect replay mode
    const replayId = new URLSearchParams(location.search).get('replay');
    if (replayId) {
      startReplaySession(replayId);
      return;
    }

    // NOTE: startAudioCapture() is triggered by window._auraStartCapture()
    // from the bootstrap module AFTER the user clicks "Start Listening"
    // and completes the mic test — keeps getUserMedia inside a user gesture.
    // Check health and initialize Supabase before capturing
    initSupabase();
    setTimeout(checkBackendHealth, 500);
    
    // Debug panel hook
    window.addEventListener('keydown', (e) => {
      if (e.key === 'D' && e.ctrlKey && e.shiftKey) toggleDebugPanel();
    });
    if (window.location.search.includes('debug=true')) toggleDebugPanel();
  } catch (err) {
    console.error('[Aura:Boot] Error during initialization:', err);
    window._auraStartCapture = function() {
      if (window._aura && window._aura.showToast) {
        window._aura.showToast('⚠️ Critical Application Error. Please refresh.', 10000);
      } else {
        const toast = document.getElementById('toast');
        if (toast) {
          toast.textContent = '⚠️ Critical Application Error. Please refresh.';
          toast.hidden = false;
          requestAnimationFrame(() => toast.classList.add('show'));
        }
      }
    };
  }
}

function toggleDebugPanel() {
  let panel = document.getElementById('debug-panel');
  if (panel) { panel.remove(); return; }
  panel = document.createElement('div');
  panel.id = 'debug-panel';
  panel.style.cssText = 'position:fixed;bottom:10px;right:10px;background:rgba(0,0,0,0.8);color:#0f0;padding:10px;font-family:monospace;font-size:11px;z-index:9999;border:1px solid #0f0;';
  document.body.appendChild(panel);
  setInterval(() => {
    if(!document.getElementById('debug-panel')) return;
    panel.innerHTML = `
      <b>AURA DEBUG</b><br>
      ConnState: ${_connState}<br>
      ConfigValid: ${window._configValid}<br>
      BackendOffline: ${window._isBackendOffline}<br>
      SupabaseReady: ${window._supabaseReady}<br>
      SessionID: ${currentSessionId}<br>
      <button onclick="startMockSequence()" style="margin-top:5px;background:#333;color:#fff;border:1px solid #555;padding:2px 4px;cursor:pointer">Force Mock</button>
      <button onclick="localStorage.clear();sessionStorage.clear();location.reload()" style="margin-top:5px;background:#333;color:#fff;border:1px solid #555;padding:2px 4px;cursor:pointer">Clear Cache</button>
    `;
  }, 1000);
}

document.readyState === 'loading'
  ? document.addEventListener('DOMContentLoaded', boot)
  : boot();

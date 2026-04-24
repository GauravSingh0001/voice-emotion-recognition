/**
 * livekit.js
 * ─────────────────────────────────────────────────────────────────────────────
 * LiveKit client module for microphone capture and room connection.
 *
 * Exposes:
 *   window.connectToLiveKit(token, serverUrl)  → Promise<boolean>
 *   window.disconnectFromLiveKit()             → void
 *   window.retryMicrophone()                   → void  (#3)
 *
 * Requires LiveKit Client SDK loaded before this script.
 */
'use strict';

(function () {
  /** @type {import('livekit-client').Room | null} */
  let _room = null;
  /** @type {import('livekit-client').LocalAudioTrack | null} */
  let _audioTrack = null;

  /**
   * Whether the last mic request was denied by the user or system.
   * Prevents automatic retry loops.
   * @type {boolean}
   */
  let _micPermissionDenied = false;

  // ── Public: Connect ────────────────────────────────────────────────────────

  /**
   * Request microphone access, connect to a LiveKit room, and publish the
   * local audio track.
   *
   * @param {string}      token          LiveKit JWT from /session/start.
   * @param {string}      serverUrl      LiveKit WebSocket URL (wss://...).
   * @param {MediaStream} [existingStream]  Pre-existing mic stream from the
   *                                        mic-test step — skips getUserMedia.
   * @returns {Promise<boolean>} true on success.
   */
  async function connectToLiveKit(token, serverUrl, existingStream = null) {
    if (typeof window.LivekitClient === 'undefined') {
      console.error('[Aura:LiveKit] SDK not loaded.');
      return false;
    }

    if (!serverUrl) {
      console.error('[Aura:LiveKit] Missing serverUrl.');
      return false;
    }

    _lkServerUrl = serverUrl;
    _lkToken = token;

    const { Room, RoomEvent, Track, createLocalAudioTrack } = window.LivekitClient;

    if (_room) await disconnectFromLiveKit();

    // ── Acquire mic stream ─────────────────────────────────────────────────────
    let stream = existingStream;
    if (stream) {
      console.log('[Aura:LiveKit] Reusing mic stream from mic-test step.');
      _micPermissionDenied = false;
    } else {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        _micPermissionDenied = false;
      } catch (err) {
        _handleMicError(err);
        return false;
      }
    }

    // Feed stream to ParticleAura for real-time volume visualisation
    if (typeof window._auraSetMicStream === 'function') {
      window._auraSetMicStream(stream);
    }

    // ── Create Room ────────────────────────────────────────────────────────
    _room = new Room({
      adaptiveStream: true,
      dynacast: true,
      audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true },
    });

    _room.on(RoomEvent.Connected,    () => { console.log('[Aura:LiveKit] Connected:', _room.name); _setConnStatus('ok'); });
    _room.on(RoomEvent.Disconnected, () => { 
      console.log('[Aura:LiveKit] Disconnected.');
      _setConnStatus('off');
      _attemptLivekitReconnect();
    });
    _room.on(RoomEvent.Reconnecting, () => { console.warn('[Aura:LiveKit] Reconnecting…');          _setConnStatus('warn'); });
    _room.on(RoomEvent.Reconnected,  () => { console.log('[Aura:LiveKit] Reconnected.');             _setConnStatus('ok'); });

    // ── Connect ────────────────────────────────────────────────────────────
    try {
      _setConnStatus('warn');
      await _room.connect(serverUrl, token, { autoSubscribe: false });
    } catch (err) {
      console.error('[Aura:LiveKit] Room connect failed:', err);
      _setConnStatus('off');
      _room = null;
      return false;
    }

    // ── Publish microphone track ───────────────────────────────────────────
    try {
      _audioTrack = await createLocalAudioTrack({
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 48000,
      });
      await _room.localParticipant.publishTrack(_audioTrack, {
        source: Track.Source.Microphone,
      });
      console.log('[Aura:LiveKit] Microphone track published.');
    } catch (err) {
      console.error('[Aura:LiveKit] Failed to publish audio track:', err);
      // Non-fatal — room is connected, bot may still receive silence
    }

    return true;
  }

  /**
   * Stop the local audio track and disconnect from the LiveKit room.
   */
  async function disconnectFromLiveKit() {
    if (_audioTrack) {
      try {
        _audioTrack.stop();
        if (_room?.localParticipant) await _room.localParticipant.unpublishTrack(_audioTrack);
      } catch (err) {
        console.warn('[Aura:LiveKit] Error stopping audio track:', err);
      }
      _audioTrack = null;
    }
    if (_room) {
      try { await _room.disconnect(); } catch (err) { console.warn('[Aura:LiveKit] Disconnect error:', err); }
      _room = null;
    }
    _setConnStatus('off');
  }

  let _lkRetryCount = 0;
  let _lkServerUrl = null;
  let _lkToken = null;

  async function _attemptLivekitReconnect() {
    if (!_lkServerUrl || !_lkToken || _lkRetryCount >= 3) {
      console.log('[Aura:LiveKit] Max retries reached or missing credentials.');
      return;
    }
    _lkRetryCount++;
    const delay = _lkRetryCount * 2000;
    console.log(`[Aura:LiveKit] Attempting reconnect ${_lkRetryCount}/3 in ${delay}ms...`);
    setTimeout(async () => {
      const ok = await connectToLiveKit(_lkToken, _lkServerUrl, window._auraMicStream || null);
      if (ok) {
        console.log('[Aura:LiveKit] Reconnect successful.');
        _lkRetryCount = 0;
      }
    }, delay);
  }

  /**
   * Mute or unmute the active local microphone track.
   * Used by the Pause/Resume session controls in main.js.
   * @param {boolean} muted  true to mute, false to unmute.
   */
  async function muteMicrophone(muted) {
    if (!_audioTrack) {
      console.warn('[Aura:LiveKit] muteMicrophone: no active audio track.');
      return;
    }
    try {
      await _audioTrack.setMuted(muted);
      console.log('[Aura:LiveKit] Microphone', muted ? 'muted ⏸' : 'unmuted ▶');
    } catch (err) {
      console.warn('[Aura:LiveKit] Could not mute microphone:', err);
    }
  }

  // ── Mic error handling (#3) ────────────────────────────────────────────────

  /**
   * Classify a getUserMedia error and show an appropriate toast to the user.
   * @param {DOMException} err  The caught error from getUserMedia.
   */
  function _handleMicError(err) {
    _micPermissionDenied = true;
    _setConnStatus('off');
    console.error('[Aura:LiveKit] Microphone error:', err.name, err.message);

    let msg;
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      msg = '🎤 Microphone access denied. Allow mic permissions in your browser settings and refresh the page.';
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      msg = '🎤 No microphone found. Please connect a microphone and refresh the page.';
    } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError' || err.name === 'OverconstrainedError') {
      msg = '🎤 Could not access microphone. Ensure no other app is currently using it.';
    } else {
      msg = `🎤 Microphone error: ${err.message || err.name}`;
    }

    _showToast(msg, 9000);
  }

  /**
   * Show a toast via the Aura UI system, or fall back to direct DOM update.
   * @param {string} msg   Message text.
   * @param {number} [ms]  Duration in milliseconds.
   */
  function _showToast(msg, ms = 6000) {
    if (window._aura?.showToast) {
      window._aura.showToast(msg, ms);
      return;
    }
    const toast = document.getElementById('toast');
    if (!toast) return;
    toast.textContent = msg;
    toast.hidden = false;
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
      toast.classList.remove('show');
      setTimeout(() => { toast.hidden = true; }, 220);
    }, ms);
  }

  /**
   * Update the connection status indicator in the Aura UI.
   * @param {'ok'|'warn'|'off'} mode
   */
  function _setConnStatus(mode) {
    if (window._aura?.setConn) { window._aura.setConn(mode); return; }
    const map = { ok:['dot dot--ok','Live'], warn:['dot dot--warn','Connecting…'], off:['dot dot--off','Disconnected'] };
    const [cls, label] = map[mode] || map.off;
    const dot = document.getElementById('conn-dot');
    const txt = document.getElementById('conn-text');
    if (dot) dot.className = cls;
    if (txt) txt.textContent = label;
  }

  // ── Public: Retry (#3) ────────────────────────────────────────────────────

  /**
   * Re-request microphone permissions after a previous denial.
   * Can be called from UI (e.g. a "Try Again" button).
   */
  async function retryMicrophone() {
    if (!_micPermissionDenied) return;
    _micPermissionDenied = false;
    _showToast('Requesting microphone access…', 3000);
    // Delegate back to main.js's audio capture flow
    if (typeof window._auraRetryAudioCapture === 'function') {
      window._auraRetryAudioCapture();
    }
  }

  // ── Expose ────────────────────────────────────────────────────────────────
  window.connectToLiveKit      = connectToLiveKit;
  window.disconnectFromLiveKit = disconnectFromLiveKit;
  window.muteMicrophone        = muteMicrophone;
  window.retryMicrophone       = retryMicrophone;

  console.log('[livekit.js] LiveKit client module loaded.');
})();

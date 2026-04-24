/**
 * particle-aura.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Three.js fluid-particle orb driven by Aura's emotion state and mic volume.
 * Depends on emotion-config.js being loaded first (window.EMOTION_PALETTE).
 *
 * Public API via window.ParticleAura:
 *   _init(THREE, EffectComposer, RenderPass, UnrealBloomPass) — boot scene
 *   setEmotion(name)   — transition orb palette to emotion
 *   setVolume(v)       — drive explosion from audio volume (0–1)
 *   destroy()          — tear down WebGL context
 *
 * Future work: OffscreenCanvas — defer rendering to a Web Worker via
 *   OffscreenCanvas.transferControlToOffscreen() to fully remove the
 *   rendering cost from the main thread. Blocked on Three.js OffscreenCanvas
 *   support maturity. Track: https://github.com/mrdoob/three.js/issues/13072
 */
'use strict';

(function (global) {

  // ── Helpers ────────────────────────────────────────────────────────────────

  /**
   * Get particle colors for an emotion, delegating to the global palette.
   * Falls back to Neutral if palette is not yet loaded.
   * @param {string} name
   * @returns {{ c1: string, c2: string }}
   */
  function getColors(name) {
    if (window.EMOTION_PALETTE) return window.EMOTION_PALETTE.getEmotion(name);
    // Minimal inline fallback — should never be reached if load order is correct
    const fb = { Neutral:'#A0AAB5', Happy:'#FFD166', Sad:'#4A6FA5', Frustrated:'#EF476F', Surprised:'#06D6A0', Sarcastic:'#9D4EDD' };
    return { c1: fb[name] || fb.Neutral, c2: '#6b7787' };
  }

  /**
   * Returns true on mobile viewports for performance tuning.
   * @returns {boolean}
   */
  function isMobile() {
    const isLowConcurrency = navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4;
    const isLowMemory = navigator.deviceMemory && navigator.deviceMemory <= 4;
    return window.innerWidth < 768 ||
      /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
      isLowConcurrency || isLowMemory;
  }

  // ── Shaders ────────────────────────────────────────────────────────────────

  const vertexShader = `
    uniform float uTime;
    uniform float uVolume;
    attribute float aRandom;
    varying float vDistance;

    void main() {
      vec3 pos = position;
      float t = uTime * 0.45;
      pos.x += sin(t + aRandom * 10.0) * 0.08;
      pos.y += cos(t + aRandom * 10.0) * 0.08;
      pos.z += sin(t * 0.6 + aRandom * 10.0) * 0.08;

      // Volume-driven explosion
      float explosion = 1.0 + uVolume * 2.2 * aRandom;
      pos *= explosion;

      vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
      gl_PointSize = (14.0 / -mvPosition.z) * (1.0 + uVolume * 0.7);
      gl_Position  = projectionMatrix * mvPosition;
      vDistance = length(pos);
    }
  `;

  const fragmentShader = `
    uniform vec3 uColor1;
    uniform vec3 uColor2;
    varying float vDistance;

    void main() {
      float d = length(gl_PointCoord - vec2(0.5));
      if (d > 0.5) discard;
      vec3 color = mix(uColor1, uColor2, clamp(vDistance / 3.0, 0.0, 1.0));
      gl_FragColor = vec4(color, 1.0 - d * 1.85);
    }
  `;

  // ── Module state ───────────────────────────────────────────────────────────
  let scene, camera, renderer, composer, material;
  let rafId = null;
  let THREE = null;
  let targetColor1 = null;
  let targetColor2 = null;
  let currentEmotion = 'Neutral';

  // Volume throttle state (#11)
  let _volumeTarget   = 0;
  let _lerpedVolume   = 0;
  let _lastVolumeSent = -1;
  const VOLUME_THRESHOLD = 0.02; // min change to update uniform

  // ── Scene construction ─────────────────────────────────────────────────────

  /**
   * Build the Three.js scene, post-processing stack, and particle geometry.
   * @param {typeof import('three')} T
   * @param {Function} EffectComposer
   * @param {Function} RenderPass
   * @param {Function} UnrealBloomPass
   */
  function _buildScene(T, EffectComposer, RenderPass, UnrealBloomPass) {
    THREE = T;

    const container = document.getElementById('particle-canvas-wrap');
    if (!container) { console.error('[particle-aura] #particle-canvas-wrap not found'); return; }

    scene  = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 6;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.id = 'particle-canvas';
    container.appendChild(renderer.domElement);

    // Post-processing: Bloom. Reduce strength on mobile for perf (#10)
    const mobile = isMobile();
    const renderScene = new RenderPass(scene, camera);
    const bloomPass   = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      mobile ? 0.45 : 0.9,  // bloom strength halved on mobile
      0.4, 0.75
    );
    bloomPass.threshold = 0.15;
    bloomPass.radius    = 1.1;

    composer = new EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);

    // Particles — fewer on mobile for performance (#10)
    const count     = mobile ? 3500 : 7000;
    const geometry  = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const randoms   = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(Math.random() * 2 - 1);
      const r     = 2.0;
      positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
      randoms[i] = Math.random();
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aRandom',  new THREE.BufferAttribute(randoms,   1));

    const cols = getColors('Neutral');
    material = new THREE.ShaderMaterial({
      uniforms: {
        uTime:   { value: 0 },
        uVolume: { value: 0 },
        uColor1: { value: new THREE.Color(cols.c1) },
        uColor2: { value: new THREE.Color(cols.c2) },
      },
      vertexShader,
      fragmentShader,
      transparent: true,
      blending:    THREE.AdditiveBlending,
      depthWrite:  false,
    });

    targetColor1 = new THREE.Color(cols.c1);
    targetColor2 = new THREE.Color(cols.c2);

    scene.add(new THREE.Points(geometry, material));
    _startLoop();
    window.addEventListener('resize', _onResize);

    console.log(`[particle-aura] Scene built — ${count} particles (mobile=${mobile})`);
  }

  /** Main render loop. */
  function _startLoop() {
    function animate() {
      rafId = requestAnimationFrame(animate);
      material.uniforms.uTime.value = performance.now() * 0.001;

      // Lerp volume toward target with throttle (#11)
      _lerpedVolume += (_volumeTarget - _lerpedVolume) * 0.14;
      const delta = Math.abs(_lerpedVolume - _lastVolumeSent);
      if (delta > VOLUME_THRESHOLD) {
        material.uniforms.uVolume.value = _lerpedVolume;
        _lastVolumeSent = _lerpedVolume;
      }

      // Lerp colors toward target
      material.uniforms.uColor1.value.lerp(targetColor1, 0.06);
      material.uniforms.uColor2.value.lerp(targetColor2, 0.06);

      scene.rotation.y += 0.0025;
      composer.render();
    }
    animate();
  }

  /** Handle window resize. */
  function _onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    composer.setSize(window.innerWidth, window.innerHeight);
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  const ParticleAura = {

    /**
     * Initialise the Three.js scene. Must be called from a module context
     * (importmap) after the DOM is ready.
     * @param {object} T             three module
     * @param {Function} EC          EffectComposer
     * @param {Function} RP          RenderPass
     * @param {Function} UBP         UnrealBloomPass
     */
    _init(T, EC, RP, UBP) { _buildScene(T, EC, RP, UBP); },

    /**
     * Transition the orb to a new emotion's color palette.
     * Delegates to window.EMOTION_PALETTE for color lookup.
     * @param {string} emotionName
     */
    setEmotion(emotionName) {
      if (emotionName === currentEmotion) return;
      currentEmotion = emotionName;
      if (!THREE) return;
      const cols = getColors(emotionName);
      targetColor1 = new THREE.Color(cols.c1);
      targetColor2 = new THREE.Color(cols.c2);
    },

    /**
     * Update the volume target for the explosion effect.
     * Changes smaller than VOLUME_THRESHOLD are ignored for performance (#11).
     * @param {number} v  0 (silence) to 1 (loud)
     */
    setVolume(v) {
      const clamped = Math.max(0, Math.min(1, v));
      if (Math.abs(clamped - _volumeTarget) > VOLUME_THRESHOLD) {
        _volumeTarget = clamped;
      }
    },

    /** Tear down WebGL and cancel animation. */
    destroy() {
      if (rafId) cancelAnimationFrame(rafId);
      window.removeEventListener('resize', _onResize);
      if (renderer) { renderer.dispose(); renderer.domElement.remove(); }
      scene = camera = renderer = composer = material = null;
    },
  };

  global.ParticleAura = ParticleAura;

})(window);

/**
 * emotion-config.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Single source of truth for ALL emotion color definitions and animation
 * parameters. Load this before particle-aura.js and main.js.
 *
 * Exposes:  window.EMOTION_PALETTE
 *   .colors           — full definition map
 *   .getEmotion(name) — safe lookup with Neutral fallback
 *   .isSarcastic(name)— sarcasm/passive-aggressive check
 *   .hexFor(name)     — primary hex color string
 *   .rgbFor(name)     — rgb() CSS string
 */
'use strict';

(function () {
  /** @type {Record<string, {c1:string,c2:string,r:number,g:number,b:number,freq:number,amp:number,base:number,glow:number}>} */
  const colors = {
    Neutral:              { c1:'#A0AAB5', c2:'#6b7787', r:160, g:170, b:181, freq:.50, amp:.040, base:.27, glow:.40 },
    Happy:                { c1:'#FFD166', c2:'#ff9a3c', r:255, g:209, b:102, freq:2.2, amp:.080, base:.32, glow:.72 },
    Sad:                  { c1:'#4A6FA5', c2:'#2a3f66', r: 74, g:111, b:165, freq:.25, amp:.025, base:.23, glow:.30 },
    Frustrated:           { c1:'#EF476F', c2:'#c42050', r:239, g: 71, b:111, freq:4.5, amp:.055, base:.30, glow:.82 },
    Agitated:             { c1:'#EF476F', c2:'#c42050', r:239, g: 71, b:111, freq:4.5, amp:.055, base:.30, glow:.82 },
    Angry:                { c1:'#EF476F', c2:'#8B0000', r:239, g: 71, b:111, freq:4.5, amp:.055, base:.30, glow:.82 },
    Surprised:            { c1:'#06D6A0', c2:'#00f2fe', r:  6, g:214, b:160, freq:1.0, amp:.120, base:.38, glow:.90 },
    Sarcastic:            { c1:'#9D4EDD', c2:'#6a1fad', r:157, g: 78, b:221, freq:1.0, amp:.040, base:.27, glow:.50 },
    'Passive-Aggressive': { c1:'#9D4EDD', c2:'#7b35c4', r:157, g: 78, b:221, freq:1.2, amp:.050, base:.28, glow:.52 },
  };

  window.EMOTION_PALETTE = {
    colors,

    /**
     * Get the full definition for an emotion, falling back to Neutral.
     * @param {string} name  Emotion label from backend.
     * @returns {object}
     */
    getEmotion(name) { return this.colors[name] || this.colors.Neutral; },

    /**
     * Returns true for sarcasm-family emotions that get special card treatment.
     * @param {string} name
     * @returns {boolean}
     */
    isSarcastic(name) { return name === 'Sarcastic' || name === 'Passive-Aggressive'; },

    /**
     * Get the primary hex color for an emotion.
     * @param {string} name
     * @returns {string}
     */
    hexFor(name) { return this.getEmotion(name).c1; },

    /**
     * Get a CSS rgb() string for an emotion.
     * @param {string} name
     * @returns {string}
     */
    rgbFor(name) {
      const e = this.getEmotion(name);
      return `rgb(${e.r},${e.g},${e.b})`;
    },
  };

  console.log('[emotion-config.js] Palette loaded —', Object.keys(colors).length, 'emotions.');
})();

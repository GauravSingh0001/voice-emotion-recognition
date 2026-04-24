/**
 * service-worker.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Aura PWA service worker — caches static assets for offline demo mode.
 * Strategy: Network-first on localhost (dev), Cache-first for deployed assets.
 * (#16)
 *
 * ⚠️  DEVELOPMENT NOTE:
 * Bump CACHE_NAME version whenever you update static assets so all browsers
 * receive the new files. The old cache is automatically deleted on activate.
 * Alternatively, unregister the service worker during active development.
 */
'use strict';

const CACHE_NAME    = 'aura-cache-v3';
const STATIC_ASSETS = [
  './',
  './index.html',
  './result.html',
  './style.css',
  './main.js',
  './livekit.js',
  './particle-aura.js',
  './emotion-config.js',
  './manifest.json',
  'https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;600&display=swap',
];

// ── Install: pre-cache static assets ─────────────────────────────────────────
self.addEventListener('install', (event) => {
  console.log('[SW] Installing cache:', CACHE_NAME);
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  // Immediately activate the new SW without waiting for old clients to close
  self.skipWaiting();
});

// ── Activate: delete ALL stale caches ────────────────────────────────────────
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== CACHE_NAME)
          .map((k) => {
            console.log('[SW] Deleting stale cache:', k);
            return caches.delete(k);
          })
      )
    )
  );
  // Take control of all open tabs immediately
  self.clients.claim();
});

// ── Fetch handler ─────────────────────────────────────────────────────────────
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Ignore non-HTTP(S) schemes (e.g., chrome-extension://)
  if (!url.protocol.startsWith('http')) {
    return;
  }

  // DEVELOPMENT: Always go to the network for localhost so JS/CSS edits
  // are picked up immediately without a manual "clear cache" step.
  const isLocalhost = url.hostname === 'localhost' || url.hostname === '127.0.0.1';

  // Also network-first for external APIs and CDN scripts that change often
  const isApiOrCdn =
    url.hostname.includes('supabase.co')     ||
    url.hostname.includes('livekit.cloud')   ||
    url.hostname.includes('cdn.jsdelivr.net')||
    url.hostname.includes('unpkg.com');

  // Network-first for first-party application files that change frequently
  const isFirstPartyAppFile =
    url.pathname.endsWith('/') ||
    url.pathname.endsWith('index.html') ||
    url.pathname.endsWith('result.html') ||
    url.pathname.endsWith('main.js') ||
    url.pathname.endsWith('style.css') ||
    url.pathname.endsWith('livekit.js') ||
    url.pathname.endsWith('particle-aura.js') ||
    url.pathname.endsWith('emotion-config.js');

  if (isLocalhost || isApiOrCdn || isFirstPartyAppFile) {
    // Network-first: fall back to cache if offline
    event.respondWith(
      fetch(event.request).then((response) => {
        // Update cache on successful fetch to ensure offline version stays fresh
        if (response && response.status === 200 && isFirstPartyAppFile) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
        }
        return response;
      }).catch(() => caches.match(event.request))
    );
    return;
  }

  // Cache-first for remote fonts and other stable external assets
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (response && response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
        }
        return response;
      });
    })
  );
});

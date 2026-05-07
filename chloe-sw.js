// Chloe — minimal service worker.
//
// Goal: make the PWA installable and let the shell load when the phone is
// briefly offline (e.g. between Wi-Fi and cellular, or as the page settles
// after launch from the Home Screen). We do NOT cache the WebSocket or any
// API responses — those are live data.
//
// Strategy:
//   - On install, pre-cache the static shell.
//   - On fetch, network-first (so updates roll out automatically), fall back
//     to cache for the shell only.

const CACHE_NAME = 'chloe-shell-v7';
const SHELL = [
  'chloe-mobile.html',
  'manifest.webmanifest',
  'chloe_icon.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(SHELL))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  const req = e.request;

  // Never intercept WebSocket upgrades (they go via a different mechanism,
  // but keep this guard explicit).
  if (req.headers.get('upgrade') === 'websocket') return;

  // Only handle same-origin GETs of shell assets. Everything else passes
  // through to the network untouched.
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== self.location.origin) return;

  e.respondWith(
    fetch(req)
      .then(res => {
        // Successful response — refresh cache for shell URLs
        if (res.ok && SHELL.some(s => url.pathname.endsWith(s))) {
          const copy = res.clone();
          caches.open(CACHE_NAME).then(c => c.put(req, copy));
        }
        return res;
      })
      .catch(() => caches.match(req))
  );
});

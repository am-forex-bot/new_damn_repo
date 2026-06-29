// Network-first service worker for Toddler Sleep Tracker.
// App files are always fetched fresh when online (no stale app), and fall
// back to cache only when offline. Firebase/CDN requests are left untouched.
const CACHE = "toddler-sleep-v2";
const ASSETS = [
  "./",
  "./index.html",
  "./config.js",
  "./manifest.webmanifest",
  "./icon.svg"
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  const url = new URL(e.request.url);
  // Only manage our own origin; let Firebase + gstatic CDN go straight to network.
  if (url.origin !== self.location.origin) return;

  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
        return res;
      })
      .catch(() =>
        caches.match(e.request).then((cached) => cached || caches.match("./index.html"))
      )
  );
});

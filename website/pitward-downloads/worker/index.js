const EVENT_KINDS = new Set(['page_view', 'download_installer', 'download_portable', 'download_checksums']);
const DAY = 86400000;
const TOKEN_AGE = 12 * 60 * 60 * 1000;
const TIME_ZONE = 'Europe/Istanbul';
const signingKeys = new WeakMap();
const lastCleanup = new WeakMap();
const botPattern = /bot\b|crawler|spider|headless|preview|facebookexternalhit|slurp/i;

function dayKey(time) {
  const parts = new Intl.DateTimeFormat('en-CA', { timeZone: TIME_ZONE, year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(new Date(time));
  const fields = Object.fromEntries(parts.map(part => [part.type, part.value]));
  return `${fields.year}-${fields.month}-${fields.day}`;
}

function encode(bytes) {
  return btoa(String.fromCharCode(...bytes)).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
}

function decode(value) {
  return Uint8Array.from(atob(value.replaceAll('-', '+').replaceAll('_', '/')), char => char.charCodeAt(0));
}

async function signingKey(db) {
  if (!signingKeys.has(db)) {
    const promise = (async () => {
      let row = await db.prepare('SELECT value FROM pitward_analytics_keys WHERE key = ?').bind('page-signing-v1').first();
      if (!row) {
        const value = encode(crypto.getRandomValues(new Uint8Array(32)));
        await db.prepare('INSERT OR IGNORE INTO pitward_analytics_keys (key, value, created_at) VALUES (?, ?, ?)').bind('page-signing-v1', value, Date.now()).run();
        row = await db.prepare('SELECT value FROM pitward_analytics_keys WHERE key = ?').bind('page-signing-v1').first();
      }
      if (!row?.value) throw new Error('Analytics signing key unavailable');
      return crypto.subtle.importKey('raw', decode(row.value), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);
    })();
    signingKeys.set(db, promise);
    promise.catch(() => signingKeys.delete(db));
  }
  return signingKeys.get(db);
}

export async function createPageToken(db, origin, { test = false, now = Date.now() } = {}) {
  const payload = { id: crypto.randomUUID(), issued: now, origin, test };
  const text = encode(new TextEncoder().encode(JSON.stringify(payload)));
  const signature = await crypto.subtle.sign('HMAC', await signingKey(db), new TextEncoder().encode(text));
  return `${text}.${encode(new Uint8Array(signature))}`;
}

export async function verifyPageToken(db, token, origin, now = Date.now()) {
  try {
    if (typeof token !== 'string' || token.length > 1024) return null;
    const pieces = token.split('.');
    if (pieces.length !== 2 || !pieces.every(piece => /^[A-Za-z0-9_-]+$/.test(piece))) return null;
    if (!await crypto.subtle.verify('HMAC', await signingKey(db), decode(pieces[1]), new TextEncoder().encode(pieces[0]))) return null;
    const payload = JSON.parse(new TextDecoder().decode(decode(pieces[0])));
    if (!/^[0-9a-f-]{36}$/.test(payload.id) || !Number.isSafeInteger(payload.issued) || payload.origin !== origin || typeof payload.test !== 'boolean') return null;
    if (payload.issued > now + 60000 || now - payload.issued > TOKEN_AGE) return null;
    return payload;
  } catch {
    return null;
  }
}

export function isOwner(request, env) {
  const id = request.headers.get('oai-authenticated-user-id');
  const email = request.headers.get('oai-authenticated-user-email');
  return Boolean(id && email && env.ANALYTICS_OWNER_EMAIL && email.trim().toLowerCase() === env.ANALYTICS_OWNER_EMAIL.trim().toLowerCase());
}

function optedOut(request) {
  return request.headers.get('sec-gpc') === '1' || request.headers.get('dnt') === '1';
}

export function normalizeSource(value, origin) {
  try {
    const url = new URL(value);
    if (!['https:', 'http:'].includes(url.protocol)) return 'direct';
    const host = url.hostname.toLowerCase().replace(/^www\./, '');
    const own = new URL(origin).hostname.toLowerCase().replace(/^www\./, '');
    if (host === own || host === 'pitward.com' || host === 'pitward.alpineanole2.chatgpt.site' || host === 'localhost' || /^[\d.:\[\]]+$/.test(host) || !host.includes('.') || host.length > 253) return 'direct';
    return host;
  } catch {
    return 'direct';
  }
}

function response(body, status = 200, type = 'application/json; charset=utf-8', extra = {}) {
  return new Response(body, { status, headers: { 'Content-Type': type, 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', ...extra } });
}

function json(body, status = 200) {
  return response(JSON.stringify(body), status);
}

function pruneHistory(db, force = false, now = Date.now()) {
  if (!force && now - (lastCleanup.get(db) || 0) <= 3600000) return Promise.resolve();
  lastCleanup.set(db, now);
  return db.prepare('DELETE FROM pitward_analytics_events WHERE visit_started_at < ?').bind(now - 90 * DAY).run().catch(error => { lastCleanup.delete(db); throw error; });
}

async function collect(request, env, ctx) {
  if (request.method !== 'POST') return response(null, 405, 'text/plain', { Allow: 'POST' });
  const origin = new URL(request.url).origin;
  if (request.headers.get('origin') !== origin || request.headers.get('sec-fetch-site') === 'cross-site') return json({ error: 'Origin rejected' }, 403);
  if (optedOut(request) || isOwner(request, env)) return new Response(null, { status: 204 });
  if (!env.DB) return json({ error: 'Analytics unavailable' }, 503);
  if (!request.headers.get('content-type')?.startsWith('application/json')) return json({ error: 'JSON required' }, 415);
  const length = Number(request.headers.get('content-length') || 0);
  if (length > 2048) return json({ error: 'Event too large' }, 413);
  const reader = request.body?.getReader();
  const chunks = [];
  let size = 0;
  if (reader) {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 2048) { await reader.cancel(); return json({ error: 'Event too large' }, 413); }
      chunks.push(value);
    }
  }
  const body = new Uint8Array(size);
  let position = 0;
  for (const chunk of chunks) { body.set(chunk, position); position += chunk.byteLength; }
  const raw = new TextDecoder().decode(body);
  let event;
  try { event = JSON.parse(raw); } catch { return json({ error: 'Invalid event' }, 400); }
  if (!event || !EVENT_KINDS.has(event.kind) || typeof event.id !== 'string' || !/^[0-9a-f-]{36}$/i.test(event.id)) return json({ error: 'Invalid event' }, 400);
  const page = await verifyPageToken(env.DB, event.token, origin);
  if (!page) return json({ error: 'Page authorization expired or invalid' }, 403);
  if (page.test) return json({ ok: true, test: true, kind: event.kind });
  const now = Date.now();
  const device = ['mobile', 'tablet', 'desktop'].includes(event.device) ? event.device : 'desktop';
  const source = normalizeSource(event.source, origin);
  const viewId = `${page.id}:view`;
  const view = env.DB.prepare('INSERT OR IGNORE INTO pitward_analytics_events (id, visit_id, kind, visit_started_at, recorded_at, day, source, device) VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
    .bind(viewId, page.id, 'page_view', page.issued, now, dayKey(page.issued), source, device);
  if (event.kind === 'page_view') {
    await view.run();
  } else {
    const click = env.DB.prepare('INSERT OR IGNORE INTO pitward_analytics_events (id, visit_id, kind, visit_started_at, recorded_at, day, source, device) SELECT ?, visit_id, ?, visit_started_at, ?, day, source, device FROM pitward_analytics_events WHERE id = ? AND (SELECT COUNT(*) FROM pitward_analytics_events WHERE visit_id = ?) < 64')
      .bind(event.id, event.kind, now, viewId, page.id);
    await env.DB.batch([view, click]);
  }
  const cleanup = pruneHistory(env.DB, false, now);
  if (ctx?.waitUntil) ctx.waitUntil(cleanup.catch(() => {}));
  else await cleanup;
  return new Response(null, { status: 204 });
}

const METRICS = `SUM(CASE WHEN kind = 'page_view' THEN 1 ELSE 0 END) AS visits,
 SUM(CASE WHEN kind = 'download_installer' THEN 1 ELSE 0 END) AS installer,
 SUM(CASE WHEN kind = 'download_portable' THEN 1 ELSE 0 END) AS portable,
 SUM(CASE WHEN kind = 'download_checksums' THEN 1 ELSE 0 END) AS checksums,
 COUNT(DISTINCT CASE WHEN kind IN ('download_installer', 'download_portable') THEN visit_id END) AS convertingVisits`;

function metrics(row = {}) {
  return Object.fromEntries(['visits', 'installer', 'portable', 'checksums', 'convertingVisits'].map(key => [key, Number(row[key]) || 0]));
}

async function statistics(request, env) {
  if (!isOwner(request, env)) return json({ error: 'Owner sign-in required' }, 401);
  if (!env.DB) return json({ error: 'Analytics unavailable' }, 503);
  await pruneHistory(env.DB, true);
  const chosen = Number(new URL(request.url).searchParams.get('days') || 30);
  if (![7, 30, 90].includes(chosen)) return json({ error: 'Choose 7, 30 or 90 days' }, 400);
  const now = Date.now();
  const first = dayKey(now - (chosen - 1) * DAY), last = dayKey(now);
  const queries = [
    `SELECT ${METRICS} FROM pitward_analytics_events WHERE day BETWEEN ? AND ?`,
    `SELECT day, ${METRICS} FROM pitward_analytics_events WHERE day BETWEEN ? AND ? GROUP BY day ORDER BY day`,
    `SELECT source, ${METRICS} FROM pitward_analytics_events WHERE day BETWEEN ? AND ? GROUP BY source ORDER BY visits DESC, source LIMIT 12`,
    `SELECT device, ${METRICS} FROM pitward_analytics_events WHERE day BETWEEN ? AND ? GROUP BY device ORDER BY visits DESC`,
  ].map(sql => env.DB.prepare(sql).bind(first, last));
  const result = await env.DB.batch(queries);
  const dailyRows = new Map((result[1].results || []).map(row => [row.day, row]));
  const daily = Array.from({ length: chosen }, (_, index) => {
    const day = dayKey(now - (chosen - index - 1) * DAY);
    return { day, ...metrics(dailyRows.get(day)) };
  });
  return json({ generatedAt: new Date(now).toISOString(), timeZone: TIME_ZONE, days: chosen,
    totals: metrics(result[0].results?.[0]), daily,
    sources: (result[2].results || []).map(row => ({ source: row.source, ...metrics(row) })),
    devices: (result[3].results || []).map(row => ({ device: row.device, ...metrics(row) })),
  });
}

function loginPage(denied = false) {
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pitward analytics — Sign in</title><link rel="icon" href="/assets/favicon.ico"><link rel="stylesheet" href="/analytics-dashboard.css"></head><body><main class="login-card"><img src="/assets/logo.svg" width="178" height="56" alt="Pitward"><span class="eyebrow">PRIVATE / OWNER ACCESS</span><h1>Pitward analytics</h1><p>${denied ? 'This account does not have access to the analytics dashboard.' : 'Sign in with the ChatGPT account that owns this website to view traffic and download clicks.'}</p><a class="primary-button" href="${denied ? '/signout-with-chatgpt?return_to=%2Fanalytics' : '/signin-with-chatgpt?return_to=%2Fanalytics'}" target="_top">${denied ? 'Switch account' : 'Sign in with ChatGPT'} ↗</a><a class="quiet-link" href="/">Back to Pitward</a></main></body></html>`;
}

export function createApp(assets) {
  const bytes = new Map();
  function assetBody(path) {
    const asset = assets[path];
    if (!asset) return null;
    if (typeof asset.text === 'string') return asset.text;
    if (!bytes.has(path)) {
      if (typeof Uint8Array.fromBase64 === 'function') bytes.set(path, Uint8Array.fromBase64(asset.base64));
      else {
        const binary = atob(asset.base64), value = new Uint8Array(binary.length);
        for (let index = 0; index < binary.length; index++) value[index] = binary.charCodeAt(index);
        bytes.set(path, value);
      }
    }
    return bytes.get(path);
  }
  return {
    async fetch(request, env = {}, ctx = {}) {
      const url = new URL(request.url);
      try {
        if (url.pathname === '/api/analytics/event') return await collect(request, env, ctx);
        if (url.pathname === '/api/analytics/stats') {
          if (request.method !== 'GET') return response(null, 405, 'text/plain', { Allow: 'GET' });
          return await statistics(request, env);
        }
        if (!['GET', 'HEAD'].includes(request.method)) return response(null, 405, 'text/plain', { Allow: 'GET, HEAD' });
        if (['/analytics', '/analytics/', '/analytics.html'].includes(url.pathname)) {
          if (!env.ANALYTICS_OWNER_EMAIL) return response('Analytics owner is not configured.', 503, 'text/plain; charset=utf-8');
          if (!isOwner(request, env)) return response(request.method === 'HEAD' ? null : loginPage(Boolean(request.headers.get('oai-authenticated-user-id'))), request.headers.get('oai-authenticated-user-id') ? 403 : 401, 'text/html; charset=utf-8');
          return response(request.method === 'HEAD' ? null : assetBody('/analytics.html'), 200, 'text/html; charset=utf-8');
        }
        const path = url.pathname === '/' ? '/index.html' : url.pathname === '/privacy' ? '/privacy.html' : url.pathname;
        const asset = assets[path];
        if (!asset) return response('Page not found.', 404, 'text/plain; charset=utf-8');
        if (path === '/index.html') {
          let bootstrap = '';
          const excluded = request.method === 'HEAD' || optedOut(request) || isOwner(request, env) || url.searchParams.get('analytics') === 'off' || (botPattern.test(request.headers.get('user-agent') || '') && url.searchParams.get('analytics') !== 'test');
          if (!excluded && env.DB) {
            try {
              const test = url.searchParams.get('analytics') === 'test';
              const token = await createPageToken(env.DB, url.origin, { test });
              bootstrap = `<meta name="pitward-analytics" content="${token}" data-test="${test}">`;
            } catch { /* The public download page stays available if analytics is offline. */ }
          }
          return response(request.method === 'HEAD' ? null : assetBody(path).replace('<!-- PITWARD_ANALYTICS -->', bootstrap), 200, asset.type);
        }
        if (request.headers.get('if-none-match') === `"${asset.hash}"`) return new Response(null, { status: 304, headers: { ETag: `"${asset.hash}"`, 'Cache-Control': 'public, max-age=300' } });
        const cache = globalThis.caches?.default;
        const cacheUrl = new URL(path, url.origin); cacheUrl.searchParams.set('__pitward_asset', asset.hash);
        const cacheKey = new Request(cacheUrl, { method: 'GET' });
        let cached;
        try { cached = cache && await cache.match(cacheKey); } catch { /* Cache failures do not affect public assets. */ }
        if (cached) return request.method === 'HEAD' ? new Response(null, { status: cached.status, headers: cached.headers }) : cached;
        const result = response(request.method === 'HEAD' ? null : assetBody(path), 200, asset.type, { 'Cache-Control': 'public, max-age=300', ETag: `"${asset.hash}"` });
        if (cache && ctx.waitUntil && request.method === 'GET') ctx.waitUntil(cache.put(cacheKey, result.clone()).catch(() => {}));
        return result;
      } catch {
        return json({ error: 'Analytics is temporarily unavailable' }, 503);
      }
    },
  };
}

export default createApp(typeof ASSET_BUNDLE === 'undefined' ? {} : ASSET_BUNDLE);

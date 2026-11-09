import test from 'node:test';
import assert from 'node:assert/strict';
import { createApp, createPageToken, normalizeSource } from '../worker/index.js';
import { createLocalDatabase } from '../scripts/local-db.mjs';

const origin = 'https://pitward.com';
const ownerHeaders = { 'oai-authenticated-user-id': 'owner-id', 'oai-authenticated-user-email': 'owner@example.com' };
const assets = {
  '/index.html': { text: '<html><head><!-- PITWARD_ANALYTICS --></head><body>Public download page</body></html>', type: 'text/html', hash: 'index-hash' },
  '/analytics.html': { text: '<h1>Private dashboard</h1>', type: 'text/html', hash: 'admin-hash' },
  '/privacy.html': { text: '<h1>Privacy</h1>', type: 'text/html', hash: 'privacy-hash' },
};

function setup(context) {
  const DB = createLocalDatabase(); context.after(() => DB.close());
  const env = { DB, ANALYTICS_OWNER_EMAIL: 'owner@example.com' };
  const app = createApp(assets), tasks = [];
  const request = (path, options = {}) => app.fetch(new Request(origin + path, options), env, { waitUntil(promise) { tasks.push(promise); } });
  const post = async (token, kind, extra = {}, headers = {}) => {
    const body = JSON.stringify({ id: crypto.randomUUID(), token, kind, source: 'https://www.reddit.com/r/simracing/?private=query', device: 'desktop', ...extra });
    const response = await request('/api/analytics/event', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json', ...headers }, body });
    await Promise.all(tasks.splice(0));
    return response;
  };
  const stats = async () => (await request('/api/analytics/stats?days=7', { headers: ownerHeaders })).json();
  const rowCount = async () => (await DB.prepare('SELECT COUNT(*) AS count FROM pitward_analytics_events').first()).count;
  return { DB, env, app, request, post, stats, rowCount };
}

test('real SQLite migration supports deduplicated visits, clicks and conversion cohorts', async context => {
  const fixture = setup(context), token = await createPageToken(fixture.DB, origin);
  assert.equal((await fixture.post(token, 'page_view')).status, 204);
  assert.equal((await fixture.post(token, 'page_view')).status, 204);
  const id = crypto.randomUUID();
  await fixture.post(token, 'download_installer', { id });
  await fixture.post(token, 'download_installer', { id });
  await fixture.post(token, 'download_portable');
  const data = await fixture.stats();
  assert.deepEqual(data.totals, { visits: 1, installer: 1, portable: 1, checksums: 0, convertingVisits: 1 });
  assert.equal(data.sources[0].source, 'reddit.com');
  assert.equal(data.timeZone, 'Europe/Istanbul');
  assert.equal(data.daily.length, 7);
  const rows = await fixture.DB.prepare('SELECT * FROM pitward_analytics_events').all();
  assert.equal(rows.results.length, 3);
  assert.ok(rows.results.every(row => row.source === 'reddit.com'));
  assert.ok(rows.results.every(row => !('ip' in row) && !('email' in row)));
});

test('a download arriving before its page-view still records one consistent visit', async context => {
  const fixture = setup(context), token = await createPageToken(fixture.DB, origin);
  await fixture.post(token, 'download_installer');
  await fixture.post(token, 'page_view');
  const data = await fixture.stats();
  assert.equal(data.totals.visits, 1); assert.equal(data.totals.installer, 1); assert.equal(data.totals.convertingVisits, 1);
});

test('dashboard HTML and statistics require the configured owner identity', async context => {
  const fixture = setup(context);
  for (const path of ['/analytics', '/analytics/', '/analytics.html']) {
    const anonymous = await fixture.request(path);
    assert.equal(anonymous.status, 401);
    assert.ok((await anonymous.text()).includes('/signin-with-chatgpt?return_to=%2Fanalytics'));
    assert.equal((await fixture.request(path, { headers: { ...ownerHeaders, 'oai-authenticated-user-email': 'someone@example.com' } })).status, 403);
    const owner = await fixture.request(path, { headers: ownerHeaders });
    assert.equal(owner.status, 200); assert.equal(owner.headers.get('cache-control'), 'no-store');
  }
  assert.equal((await fixture.request('/api/analytics/stats')).status, 401);
  assert.equal((await fixture.request('/api/analytics/stats', { headers: { 'oai-authenticated-user-email': 'owner@example.com' } })).status, 401);
  assert.equal((await fixture.request('/api/analytics/stats?days=10000', { headers: ownerHeaders })).status, 400);
});

test('event writes require same-origin signed, unexpired page authorization', async context => {
  const fixture = setup(context), token = await createPageToken(fixture.DB, origin);
  assert.equal((await fixture.post(token, 'page_view', {}, { Origin: 'https://other.example' })).status, 403);
  assert.equal((await fixture.post(token, 'page_view', {}, { 'Sec-Fetch-Site': 'cross-site' })).status, 403);
  const [body, signature] = token.split('.');
  const corrupt = body + '.' + (signature[0] === 'a' ? 'b' : 'a') + signature.slice(1);
  assert.equal((await fixture.post(corrupt, 'page_view')).status, 403);
  const expired = await createPageToken(fixture.DB, origin, { now: Date.now() - 13 * 3600000 });
  assert.equal((await fixture.post(expired, 'page_view')).status, 403);
  const anotherOrigin = await createPageToken(fixture.DB, 'https://other.example');
  assert.equal((await fixture.post(anotherOrigin, 'page_view')).status, 403);
  assert.equal(await fixture.rowCount(), 0);
});

test('privacy signals, signed-in owner traffic and explicit exclusions collect nothing', async context => {
  const fixture = setup(context), token = await createPageToken(fixture.DB, origin);
  for (const headers of [{ DNT: '1' }, { 'Sec-GPC': '1' }, ownerHeaders]) {
    const html = await (await fixture.request('/', { headers })).text();
    assert.ok(!html.includes('name="pitward-analytics"'));
    assert.equal((await fixture.post(token, 'page_view', {}, headers)).status, 204);
  }
  assert.ok(!(await (await fixture.request('/?analytics=off')).text()).includes('name="pitward-analytics"'));
  assert.ok(!(await (await fixture.request('/', { headers: { 'User-Agent': 'HeadlessChrome' } })).text()).includes('name="pitward-analytics"'));
  assert.equal(await fixture.rowCount(), 0);
});

test('test-mode validates collection without creating fake traffic', async context => {
  const fixture = setup(context);
  const html = await (await fixture.request('/?analytics=test', { headers: { 'User-Agent': 'HeadlessChrome' } })).text();
  const token = html.match(/name="pitward-analytics" content="([^"]+)"/)?.[1];
  assert.ok(token);
  for (const kind of ['page_view', 'download_installer', 'download_portable', 'download_checksums']) {
    const result = await fixture.post(token, kind);
    assert.equal(result.status, 200); assert.equal((await result.json()).test, true);
  }
  assert.equal(await fixture.rowCount(), 0);
});

test('invalid or oversized payloads and non-JSON writes are rejected', async context => {
  const fixture = setup(context), token = await createPageToken(fixture.DB, origin);
  assert.equal((await fixture.post(token, 'anything_else')).status, 400);
  assert.equal((await fixture.post(token, 'page_view', { id: 'invalid' })).status, 400);
  assert.equal((await fixture.post(token, 'page_view', { source: 'x'.repeat(2500) })).status, 413);
  assert.equal((await fixture.request('/api/analytics/event', { method: 'POST', headers: { Origin: origin, 'Content-Type': 'text/plain' }, body: 'no' })).status, 415);
  assert.equal((await fixture.request('/api/analytics/event')).status, 405);
  assert.equal(await fixture.rowCount(), 0);
});

test('one public page token has a bounded event budget', async context => {
  const fixture = setup(context), token = await createPageToken(fixture.DB, origin);
  for (let index = 0; index < 70; index++) await fixture.post(token, 'download_installer');
  assert.equal(await fixture.rowCount(), 64);
  const data = await fixture.stats();
  assert.equal(data.totals.visits, 1); assert.equal(data.totals.installer, 63);
});

test('dashboard refresh prunes events outside the 90-day reporting window', async context => {
  const fixture = setup(context), old = Date.now() - 91 * 86400000;
  await fixture.DB.prepare('INSERT INTO pitward_analytics_events (id, visit_id, kind, visit_started_at, recorded_at, day, source, device) VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
    .bind('old', 'old', 'page_view', old, old, '2000-01-01', 'direct', 'desktop').run();
  assert.equal(await fixture.rowCount(), 1);
  await fixture.stats();
  assert.equal(await fixture.rowCount(), 0);
});

test('analytics failures do not prevent the public download page from loading', async context => {
  const fixture = setup(context);
  const result = await fixture.app.fetch(new Request(origin), { DB: { prepare() { throw new Error('Unavailable'); } } });
  assert.equal(result.status, 200); assert.ok((await result.text()).includes('Public download page'));
});

test('source normalization never retains private paths, queries or addresses', () => {
  assert.equal(normalizeSource('https://news.ycombinator.com/item?id=123', origin), 'news.ycombinator.com');
  for (const value of ['', 'invalid', 'https://pitward.com/private', 'http://127.0.0.1/private', 'http://localhost:4000/', 'file:///private/path']) assert.equal(normalizeSource(value, origin), 'direct');
});

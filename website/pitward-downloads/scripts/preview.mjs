import { createServer } from 'node:http';
import { createLocalDatabase } from './local-db.mjs';
import worker from '../dist/server/index.js';

const database = createLocalDatabase();
const ownerPreview = process.argv.includes('--owner');
const env = { DB: database, ANALYTICS_OWNER_EMAIL: 'owner@example.com' };
const server = createServer(async (incoming, outgoing) => {
  const headers = new Headers();
  for (const [name, value] of Object.entries(incoming.headers)) {
    if (name.startsWith('oai-authenticated-user-')) continue;
    if (typeof value === 'string') headers.set(name, value);
  }
  const url = new URL(incoming.url, 'http://127.0.0.1:4175');
  if (ownerPreview && ['/analytics', '/analytics.html', '/api/analytics/stats'].includes(url.pathname)) {
    headers.set('oai-authenticated-user-id', 'local-preview-owner');
    headers.set('oai-authenticated-user-email', env.ANALYTICS_OWNER_EMAIL);
  }
  const chunks = []; let size = 0;
  for await (const chunk of incoming) {
    size += chunk.length;
    if (size > 16384) { outgoing.writeHead(413); outgoing.end(); return; }
    chunks.push(chunk);
  }
  const request = new Request(url, { method: incoming.method, headers, ...(chunks.length ? { body: Buffer.concat(chunks) } : {}) });
  const result = await worker.fetch(request, env, { waitUntil(promise) { promise.catch(() => {}); } });
  outgoing.writeHead(result.status, Object.fromEntries(result.headers));
  outgoing.end(Buffer.from(await result.arrayBuffer()));
});
server.listen(4175, '127.0.0.1', () => console.log(`Local: http://127.0.0.1:4175${ownerPreview ? '/analytics (local owner preview)' : ''}`));
process.on('SIGINT', () => { server.close(); database.close(); process.exit(); });

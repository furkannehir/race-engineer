import { createHash } from 'node:crypto';
import { lstatSync, mkdirSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon', '.ttf': 'font/ttf', '.txt': 'text/plain; charset=utf-8' };
const assets = {};
function collect(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const file = path.join(directory, entry.name);
    if (entry.isSymbolicLink()) throw new Error('Public assets must be regular files');
    if (entry.isDirectory()) { collect(file); continue; }
    const body = readFileSync(file), extension = path.extname(file);
    const name = '/' + path.relative('public', file).split(path.sep).join('/');
    const binary = ['.png', '.ico', '.ttf'].includes(extension);
    assets[name] = { type: types[extension] || 'application/octet-stream', hash: createHash('sha256').update(body).digest('hex'), ...(binary ? { base64: body.toString('base64') } : { text: body.toString('utf8') }) };
  }
}
collect('public');
const project = realpathSync(process.cwd());
const output = path.resolve(project, 'dist');
if (path.dirname(output) !== project || path.basename(output) !== 'dist' || lstatSync(output, { throwIfNoEntry: false })?.isSymbolicLink()) throw new Error('Refusing to replace an unexpected build directory');
rmSync(output, { recursive: true, force: true });
mkdirSync('dist/server', { recursive: true });
mkdirSync('dist/.openai', { recursive: true });
const worker = readFileSync('worker/index.js', 'utf8').replace(/^export (?=(?:async )?function\s)/gm, '');
writeFileSync('dist/server/index.js', `const ASSET_BUNDLE = ${JSON.stringify(assets)};\n${worker}`);
writeFileSync('dist/.openai/hosting.json', readFileSync('.openai/hosting.json'));
console.log(`Built analytics Worker with ${Object.keys(assets).length} embedded assets.`);

import { isOptedOut, setOptOut } from './analytics.js';

const status = document.querySelector('#dashboard-status');
const period = document.querySelector('#period');
const refresh = document.querySelector('#refresh');
const number = new Intl.NumberFormat('en');
const svgNamespace = 'http://www.w3.org/2000/svg';

function element(name, attributes = {}, text) {
  const node = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function drawChart(rows) {
  const root = document.querySelector('#chart');
  root.replaceChildren();
  if (!rows.some(row => row.visits || row.installer || row.portable)) {
    const message = document.createElement('p');
    message.className = 'empty-state';
    message.textContent = 'No traffic recorded yet. Visits will appear here after people reach the website.';
    root.append(message);
    return;
  }
  const width = 1050, height = 240, left = 42, right = width - 12, top = 15, bottom = height - 30;
  const peak = Math.max(1, ...rows.flatMap(row => [row.visits, row.installer + row.portable]));
  const steps = Math.min(4, peak), max = Math.ceil(peak / steps) * steps;
  const svg = element('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `Daily website visits and app download clicks over the last ${rows.length} days` });
  for (let step = 0; step <= steps; step++) {
    const y = bottom - (bottom - top) * step / steps;
    svg.append(element('line', { x1: left, x2: right, y1: y, y2: y, stroke: '#354032', 'stroke-width': 1 }));
    svg.append(element('text', { x: left - 10, y: y + 4, fill: '#9eae92', 'font-size': 11, 'text-anchor': 'end' }, String(max * step / steps)));
  }
  for (const [field, color] of [['visits', '#e1e8d5'], ['clicks', '#ff8000']]) {
    const points = rows.map((row, index) => `${left + (right - left) * index / Math.max(1, rows.length - 1)},${bottom - (field === 'visits' ? row.visits : row.installer + row.portable) / max * (bottom - top)}`).join(' ');
    svg.append(element('polyline', { points, fill: 'none', stroke: color, 'stroke-width': 2.5, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
  }
  for (const index of [...new Set([0, Math.floor((rows.length - 1) / 2), rows.length - 1])]) {
    const date = new Date(`${rows[index].day}T12:00:00Z`);
    const label = new Intl.DateTimeFormat('en', { day: 'numeric', month: 'short', timeZone: 'Europe/Istanbul' }).format(date);
    svg.append(element('text', { x: left + (right - left) * index / Math.max(1, rows.length - 1), y: height - 6, fill: '#9eae92', 'font-size': 11, 'text-anchor': index === 0 ? 'start' : index === rows.length - 1 ? 'end' : 'middle' }, label));
  }
  root.append(svg);
}

function table(selector, rows, key) {
  const body = document.querySelector(selector);
  body.replaceChildren();
  if (!rows.length) {
    const row = document.createElement('tr'), cell = document.createElement('td');
    cell.colSpan = 3; cell.className = 'empty-state'; cell.textContent = 'No traffic recorded yet.'; row.append(cell); body.append(row); return;
  }
  for (const value of rows) {
    const row = document.createElement('tr');
    const label = key === 'source' && value[key] === 'direct' ? 'Direct / unknown' : String(value[key]);
    for (const text of [label, number.format(value.visits), number.format(value.installer + value.portable)]) {
      const cell = document.createElement('td'); cell.textContent = text; row.append(cell);
    }
    body.append(row);
  }
}

let currentRequest;
async function load() {
  currentRequest?.abort();
  const controller = new AbortController(); currentRequest = controller;
  refresh.disabled = true; status.classList.remove('error'); status.textContent = 'Loading analytics…';
  try {
    const response = await fetch(`/api/analytics/stats?days=${period.value}`, { signal: controller.signal, cache: 'no-store' });
    if (response.status === 401) { location.assign('/analytics'); return; }
    if (!response.ok) throw new Error('Analytics could not be loaded. Try refreshing.');
    const data = await response.json(), totals = data.totals;
    document.querySelector('#visits').textContent = number.format(totals.visits);
    document.querySelector('#installer').textContent = number.format(totals.installer);
    document.querySelector('#portable').textContent = number.format(totals.portable);
    document.querySelector('#conversion').textContent = `${totals.visits ? (totals.convertingVisits / totals.visits * 100).toFixed(1) : '0'}%`;
    drawChart(data.daily); table('#sources', data.sources, 'source'); table('#devices', data.devices, 'device');
    const updated = new Intl.DateTimeFormat('en', { dateStyle: 'medium', timeStyle: 'short', timeZone: data.timeZone }).format(new Date(data.generatedAt));
    status.textContent = `Last ${data.days} days · Updated ${updated} · Europe/Istanbul`;
  } catch (error) {
    if (error.name === 'AbortError') return;
    status.classList.add('error'); status.textContent = error.message;
  } finally { if (currentRequest === controller) refresh.disabled = false; }
}

const exclude = document.querySelector('#exclude-browser');
try { if (localStorage.getItem('pitward.analytics.optout') === null) setOptOut(true); } catch { /* Preferences may be disabled by the browser. */ }
exclude.checked = isOptedOut();
exclude.addEventListener('change', () => {
  const saved = setOptOut(exclude.checked);
  document.querySelector('#preference-status').textContent = saved ? (exclude.checked ? 'This browser will stay excluded after sign-out.' : 'Signed-out visits from this browser may be counted.') : 'Your browser did not allow this preference to be saved.';
});
period.addEventListener('change', load); refresh.addEventListener('click', load); load();

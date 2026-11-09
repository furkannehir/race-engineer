const preferenceKey = 'pitward.analytics.optout';

export function isOptedOut() {
  try { return localStorage.getItem(preferenceKey) === '1'; } catch { return false; }
}

export function setOptOut(value) {
  try {
    localStorage.setItem(preferenceKey, value ? '1' : '0');
    return true;
  } catch { return false; }
}

const bootstrap = document.querySelector('meta[name="pitward-analytics"]');
const testMode = bootstrap?.dataset.test === 'true';
const privacySignal = navigator.globalPrivacyControl === true || navigator.doNotTrack === '1';

function send(kind) {
  if (!bootstrap?.content || privacySignal || isOptedOut() || (navigator.webdriver && !testMode)) return;
  const device = innerWidth < 768 ? 'mobile' : innerWidth < 1024 ? 'tablet' : 'desktop';
  let source = '';
  try { source = new URL(document.referrer).origin; } catch { /* Direct visit. */ }
  const body = JSON.stringify({ id: crypto.randomUUID(), kind, token: bootstrap.content, source, device });
  const data = new Blob([body], { type: 'application/json' });
  if (!navigator.sendBeacon?.('/api/analytics/event', data)) {
    fetch('/api/analytics/event', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body, keepalive: true }).catch(() => {});
  }
}

if (bootstrap) {
  send('page_view');
  const trackClick = event => {
    if ((event.type === 'click' && event.button !== 0) || (event.type === 'auxclick' && event.button !== 1)) return;
    const link = event.target.closest?.('a');
    if (!link || link.getAttribute('aria-disabled') === 'true' || !link.hasAttribute('href')) return;
    const kind = { 'installer-download': 'download_installer', 'portable-download': 'download_portable', 'checksums-download': 'download_checksums' }[link.id];
    if (kind) send(kind);
  };
  document.addEventListener('click', trackClick, { capture: true });
  document.addEventListener('auxclick', trackClick, { capture: true });
}

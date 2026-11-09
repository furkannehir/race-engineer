import { getReleaseDownloads } from './release-state.js';

const status = document.querySelector('#release-status');
async function checkRelease() {
  const configResponse = await fetch('release.json');
  if (!configResponse.ok) throw new Error('Release configuration unavailable');
  const config = await configResponse.json();
  document.querySelectorAll('[data-version]').forEach(node => { node.textContent = config.version; });
  document.querySelector('#component-size').textContent = `${(config.componentBytes / 1e9).toFixed(1)} GB`;
  const response = await fetch(`https://api.github.com/repos/${config.repository}/releases/tags/v${config.version}`, { headers: { Accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(10000) });
  if (response.status === 404) return;
  if (!response.ok) throw new Error('Release service unavailable');
  const release = await response.json();
  const downloads = getReleaseDownloads(release, config);
  if (!downloads) return;
  for (const [key, selector] of Object.entries({installer: '#installer-download', portable: '#portable-download', checksums: '#checksums-download'})) {
    const link = document.querySelector(selector);
    link.href = downloads[key];
    link.removeAttribute('aria-disabled');
    link.classList.remove('disabled');
  }
  document.querySelector('#download-label').textContent = 'Download for Windows';
  document.querySelector('#release-badge').textContent = 'AVAILABLE NOW';
  document.querySelector('#release-note-status').textContent = 'First public release';
  const date = new Date(release.published_at);
  status.textContent = Number.isNaN(date.getTime()) ? `Version ${config.version} is ready to download.` : `Published ${new Intl.DateTimeFormat('en', { dateStyle: 'medium' }).format(date)} · Downloads hosted on GitHub.`;
}
checkRelease().catch(() => {
  status.textContent = 'We couldn’t confirm download availability. Check GitHub releases for the latest published files.';
});

if ('IntersectionObserver' in window && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
  document.documentElement.classList.add('motion-ready');
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.remove('waiting');
        observer.unobserve(entry.target);
      }
    }
  }, { threshold: 0.08 });
  document.querySelectorAll('.reveal').forEach(element => {
    element.classList.add('waiting');
    observer.observe(element);
  });
}

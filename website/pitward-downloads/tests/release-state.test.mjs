import test from 'node:test';
import assert from 'node:assert/strict';
import { getReleaseDownloads } from '../public/release-state.js';

const config = {version: '0.1.0', repository: 'furkannehir/race-engineer'};
function release() {
  return {tag_name: 'v0.1.0', draft: false, assets: ['Pitward-Setup-0.1.0-windows-x64.exe', 'Pitward-0.1.0-windows-x64.zip', 'SHA256SUMS.txt'].map(name => ({name, size: 100, state: 'uploaded', browser_download_url: `https://github.com/${config.repository}/releases/download/v0.1.0/${name}`}))};
}
test('unpublished or incomplete releases never enable downloads', () => {
  assert.equal(getReleaseDownloads(null, config), null);
  assert.equal(getReleaseDownloads({...release(), draft: true}, config), null);
  assert.equal(getReleaseDownloads({...release(), tag_name: 'v0.2.0'}, config), null);
  assert.equal(getReleaseDownloads({...release(), assets: release().assets.slice(0, 2)}, config), null);
});
test('download links must identify the exact published GitHub assets', () => {
  const bad = release();
  bad.assets[0].browser_download_url = 'https://example.com/unverified.exe';
  assert.equal(getReleaseDownloads(bad, config), null);
  bad.assets[0].browser_download_url = release().assets[0].browser_download_url;
  bad.assets[0].size = 0;
  assert.equal(getReleaseDownloads(bad, config), null);
  assert.match(getReleaseDownloads(release(), config).installer, /Pitward-Setup-0\.1\.0-windows-x64\.exe$/);
});

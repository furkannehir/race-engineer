export function getReleaseDownloads(release, { version, repository }) {
  if (!release || release.draft || release.tag_name !== `v${version}`) return null;
  const prefix = `https://github.com/${repository}/releases/download/v${version}/`;
  const names = {
    installer: `Pitward-Setup-${version}-windows-x64.exe`,
    portable: `Pitward-${version}-windows-x64.zip`,
    checksums: 'SHA256SUMS.txt',
  };
  const result = {};
  for (const [key, name] of Object.entries(names)) {
    const asset = release.assets?.find(item => item.name === name);
    if (!asset || asset.state !== 'uploaded' || asset.size <= 0 || asset.browser_download_url !== prefix + name) return null;
    result[key] = asset.browser_download_url;
  }
  return result;
}

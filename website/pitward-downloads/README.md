# Pitward website

A cinematic, dependency-free static launch and download page for pitward.com. Source lives in
`public/`; `node scripts/build.mjs` copies it to `dist/`. Run
`node --test tests/release-state.test.mjs` to check download availability rules.

Preview locally from this directory:

```powershell
node scripts/build.mjs
python -m http.server 4173 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:4173`. No account or API secret is required. The dashboard
checks the public GitHub release API once per page load and links directly to
published GitHub assets. API failures leave downloads disabled and show a link
to GitHub releases. Without JavaScript, a GitHub releases link is shown.

`public/release.json` supplies the target app version, repository, and component
download size. Update it with the authoritative app version for subsequent releases.
The dashboard requires the exact installer, portable ZIP, and `SHA256SUMS.txt`
assets; a draft or incomplete release never activates download links.

The logo and favicon are copied from the repository's Pitward brand pack.
`public/assets/desktop-preview.png` is a preview capture of the actual Qt panel,
with no live telemetry, model, microphone, or playback access.
The generated racing hero and its final prompt are documented in `ASSETS.md`.
Motion respects the visitor's reduced-motion preference.

Sites identity is in `.openai/hosting.json`. A separate Sites source checkout may
be created under `.sites-checkout/` by the Sites workflow; it is ignored by the
application repository. Keep the local source here in sync when making changes.
DNS records returned by Sites must be configured at the domain's DNS provider
before `www.pitward.com` can serve this page.

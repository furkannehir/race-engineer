# Pitward website and first-party analytics

Public website: https://pitward.com. Private owner dashboard:
https://pitward.com/analytics. The dashboard uses the normal Sites **Sign in with
ChatGPT** flow; only the configured owner's identity can read its data.

## Architecture

- `public/`: the existing launch/download page, dashboard, tracker and privacy page.
- `worker/index.js`: a dependency-free Cloudflare Worker that serves the embedded
  assets, authorizes collection, records events in D1 and protects analytics reads.
- `db/schema.ts` and `drizzle/`: generated schema-only migrations and metadata.
- `.openai/hosting.json`: retained Site identity and logical `d1: "DB"` binding.
- `scripts/build.mjs`: builds one Worker ES module with embedded assets. Only
  `dist/server/index.js` and hosting metadata are emitted; legacy static output is
  removed from the verified project `dist` directory before each build.

Drizzle packages are build-time tools for generating migrations. The deployed
Worker imports no npm packages. Sites provisions D1 and applies the committed
migrations before uploading the Worker. Never alter applied migrations; append
new generated migrations for schema changes.

## Runtime configuration

Set `ANALYTICS_OWNER_EMAIL` as a secret using Sites runtime environment settings.
It must match the authenticated website owner's email. The real value is never
stored in source or the browser. `.env.example` documents the key only.

The public website stays public. Dashboard routes require both trusted Sites
identity headers and the configured owner email; stats never rely on client-side
access checks. Anonymous and different-account requests are denied. Platform
sign-in/callback routes are not implemented by this Worker.

## Metrics and data

The collector records page loads, installer clicks, portable ZIP clicks and
checksum clicks. Primary and middle-click activations are counted. An ephemeral,
server-signed page-load token authorizes a bounded set of events; event IDs make
retries idempotent. A click arriving before its page-view creates that view once.

Visits mean page loads, not identifiable people. Conversion is the percentage of
visits with an installer or ZIP click. Events are attributed to the visit's date
in Europe/Istanbul time. A click does not establish a completed download or install;
direct GitHub downloads are separate from these website metrics.

Stored fields are a random per-page visit ID, event kind, times, referring
hostname and broad viewport group. The browser sends only the referring origin;
paths, queries, names, emails, raw IP addresses, user-agent strings and persistent
cross-visit browser identifiers are not stored. There are no analytics cookies.
An optional localStorage opt-out contains only `0` or `1`, never an identifier.

Reports cover 7, 30 or 90 days. Older event records are pruned during analytics
activity and dashboard refreshes. If the site is inactive, pruning waits for the
next such activity; this is not a scheduled hard-deletion timer.

Signed-in owner traffic, browser privacy signals (GPC/DNT), recognizable bots,
automated browsers and explicit opt-outs are excluded. The dashboard offers a
browser opt-out for after sign-out; the public privacy page offers the same choice.

Analytics failures never prevent the public page or GitHub downloads from working.
The initial GitHub release counters cannot be backfilled as website visit events.

## Local development and checks

Use Node 24 or newer:

```text
npm ci
npm run db:generate
npm test
npm run build
npm run preview -- --owner
```

Open http://127.0.0.1:4175/analytics. Preview storage is in-memory SQLite and the
`--owner` flag injects a placeholder owner only into local dashboard requests.
Incoming identity headers are stripped by the local server first. None of this
preview server code is deployed.

`/?analytics=test` validates signed events without storing fake traffic. It also
allows automated browser collection checks. `/?analytics=off` omits the tracker
token entirely. Browser opt-outs and privacy signals are respected in test mode.

The tests use the actual generated migration and SQLite to check authorization,
deduplication, conversion cohorts, retention pruning, bounded inputs, test-mode
exclusion and graceful public-page fallback. Browser QA additionally verifies
real beacons, middle-clicks, opt-outs, GPC, responsive layouts and chart rendering.

## Publishing

Preserve the Site project ID and public audience. Build and publish through the
Sites source helper; its packager includes the generated `drizzle` migration files
and metadata. Keep source here and the application repository's website copy in
sync, excluding `.git`, `node_modules`, `dist` and ignored local state.

The download page checks GitHub's public target release from `public/release.json`
once per page load. It enables links only for the complete published asset set.
The [generated hero and its prompt](ASSETS.md), existing brand pack and actual Qt
panel preview are unchanged by the analytics addition.

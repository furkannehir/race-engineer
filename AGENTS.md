# Working rules for agents

These rules apply to every change in this repository, by people and by agents.

## Checks

Run these before every commit. CI runs the same three.

```
python -m pytest
python -m ruff check .
python -m mypy
```

The website under `website/pitward-downloads` has its own checks: `node --test tests/*.test.mjs`
and `node scripts/build.mjs`, run from that folder.

## Commits

- One logical change per commit.
- Tests go in the same commit as the code they cover.
- Refactors and behavior changes go in separate commits.
- Aim for under ~400 changed lines per commit, excluding fixtures and lockfiles.
- Use the existing prefixes: `feat:`, `fix:`, `refactor:`, `docs:`.
- One branch and one pull request per piece of work. Don't commit to `main` directly.

## Docs

- Never create a new doc for a feature, trial or investigation. Update the existing doc it
  belongs to, or put the explanation in the commit message or the pull request.
- Trial results and measurements go in the pull request, not in `docs/`.
- Docs describe the current state only. History lives in git.

## Model behavior

- A failure seen in a live session becomes a case in the eval question bank
  (`fixtures/intelligence-evaluation/`, run by `scripts/evaluate_ministral_dialogue.py`).
- Don't add a prompt sentence or a regex to fix a single failure. Only generic invariants
  live in code: unverified numbers, control markup in a reply, wrong language, empty reply,
  and session or telemetry changes.
- Python owns the facts; the model owns the wording.

## Tests

- A change to dialogue behavior needs a test that fails before the change.

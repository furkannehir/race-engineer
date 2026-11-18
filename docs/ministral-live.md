# Ministral in the desktop radio

Pitward 0.2.1 uses **Ministral 3 3B Instruct 2512 Q4_K_M**,
through the existing local llama.cpp b12088 server. `ministral-native-v1` is
selected in both the config schema and `config/default.toml`. No hosted API,
second language model, Python Transformers runtime or vision projector is used.
The panel, `intelligence-replay` and checked chat demo share packaged native
dialogue code. Published v0.2.0 installers and website downloads still use LFM;
the 0.2.1 migration does not replace any 0.2.0 release file.

## Source setup and test

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\setup_ministral.py --backend gpu
.\.venv\Scripts\python.exe scripts\smoke_ministral_live.py --backend gpu
.\start-panel.cmd
```

Setup verifies cached files before downloading anything. It reuses the pinned
CPU/CUDA runtimes under `data/lfm3-prototype` and the model under
`data/dialogue-comparison`; those directory names are historical, not the model
selected for inference. Both directories stay out of Git.

The smoke script owns a temporary loopback server on port 18201, uses replay
telemetry and exercises the **production factory and delivery refresh**. It
never opens the microphone or plays audio. Add `--synthesize` to run the same
replies through one Chatterbox worker without playback. Results go to ignored
`data/diagnostics/ministral-integration/`; a failed request makes the script
return a nonzero exit code. A completed request is not proof that its meaning
was correct; inspect the replies too.

For an explicit CPU test, set up and run with `--backend cpu`. This changes only
the test's owned server, not the GPU source default. CPU inference can exceed
racing deadlines. `scripts/start_conversation_model.py` plus
`python -m race_engineer intelligence-replay` also use the configured model.
Old context-plan evaluation/latency tools use their original JSON contracts;
select `config/lfm-legacy.toml` or `config/qwen-legacy.toml` explicitly for those
experiments. They are not Ministral dialogue benchmarks.

Rollback is explicit:

```powershell
.\start-panel.cmd --config config/lfm-legacy.toml
.\start-panel.cmd --config config/qwen-legacy.toml
```

The corresponding old assets must already exist. No failed Ministral request
silently starts LFM or Qwen.

## Runtime and delivery

1. The existing owned llama.cpp server loads the model once for the session.
   Loopback HTTP runs off the UI, telemetry and microphone threads.
2. Ministral selects one read-only operation. The adapter and core supply facts
   or deterministic calculations; Ministral composes a short spoken reply.
3. Validation binds measurements, units, categorical states and limitations to
   evidence. One repair is allowed. Invalid output is visibly rejected while
   telemetry and automatic announcements continue.
4. Once the radio queue gives the reply its slot, the selected operation runs
   again against current telemetry. It refreshes values without another LLM
   call. Changed availability, contradictory boolean verdicts, replaced requests
   and changed sessions fail closed.
5. Only successful playback commits the reply and its **actually delivered**
   readings to native dialogue/recall. Muted, cancelled, expired or unplayed
   replies do not become remembered calls. Reconnect generations and language
   changes reset that memory; history is bounded by `history_turns`.

Replies and automatic announcements still share Chatterbox and the existing
priority/cancellation/output-device path. PTT WAV cues are unchanged. Native
model stages, selected tool names and rejection reasons appear in correlated
radio logs; draft text follows `record_radio_text` and telemetry values retain
their separate privacy gate.

## Windows packages

Component set **4** selects the exact model/hash in
`src/race_engineer/assets/components.v1.json`. New builds download about **7.0 GB**
for a fresh setup, with the conversation model accounting for 2.15 GB. An upgrade
reuses unchanged speech assets and preserves profiles/settings. Missing Ministral
weights trigger the normal setup/restart flow. After the owned server successfully
loads verified Ministral weights, Pitward records the migration and removes the
verified obsolete LFM file from its managed cache, freeing about 1.10 GB. Failed
loads retain LFM; locked files defer cleanup until a later start. Custom model
files, source caches, speech assets and driver data are preserved.

The installer continues to distribute the verified **CPU** llama.cpp runtime.
Source CUDA support does not imply packaged CUDA support or proven CPU race
latency. The new model uses Apache-2.0 terms; see
[the component review](../distribution/REVIEW-MINISTRAL.md). Old release records
and website metadata describe the published assets, not this development set.

## Limits and acceptance

Local verification on 18 November 2026:

- 1000 tests passed; two optional NumPy-dependent tests were skipped in the main
  environment. Ruff and strict mypy passed (118 package files).
- Eight production-factory requests completed on CUDA and CPU. This is a
  completion count, not an accuracy score: the Turkish radio check was wrong.
  CUDA model latency was 190–509 ms (median 261 ms), excluding synthesis.
  CPU ranged from 1.68 to 19.47 seconds, including cold prompt processing.
- All eight generated replies synthesized through one reused Chatterbox worker
  without playback. Synthesis ranged from 0.77 to 2.87 seconds on this machine.
- Isolated PyInstaller panel/downloader builds passed. The offscreen frozen
  panel preview started and closed normally; archive inspection found the four
  native Ministral modules and no `scripts` package. The frozen downloader
  reported component set 4 and the correct missing Ministral model.

Evidence is under ignored `data/diagnostics/ministral-integration/`. These local
builds are development checks, not replacement v0.2.0 release artifacts.

English remains the primary validated language. Turkish position and lap-time
smoke replies work, but a Turkish radio check can be classified as unsupported
information. Treat Turkish social understanding as unfinished. This migration
does not close the existing ambiguous-question, unsupported-assessment or
radio-recall topic-selection issues. No car controls were added.

Tests cover delivery refresh, no history before playback, rejected boolean
changes, cancellation/no-play recall, session/language resets, zero history,
manifest/config alignment and the existing demo evidence checks. Before another
public release, exercise the rebuilt installer on clean Windows, speak through
PTT in iRacing/VR, check EN/TR audio and verify an upgrade from v0.2.0. Offline
synthesis cannot establish audible quality, microphone behavior or VR load.

Model retirement is defined explicitly in the component manifest's
`model_migrations` list. Future replacements must include the retired artifacts
from every supported upgrade path, including users who skip releases. No wildcard
model deletion is used. `.model-migrations.json` records successful load/cleanup;
verification and deletion run off the UI thread and honor Stop. Redirected paths,
unknown files and write failures defer cleanup and are logged without stopping
telemetry. Offline migration against the real cached weights freed 1,099,627,808
bytes in a temporary cache and the loaded model still answered a synthetic
position question; original development weights remained intact.

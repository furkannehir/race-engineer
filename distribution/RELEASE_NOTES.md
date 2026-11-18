# Pitward 0.2.1

Ministral 3 3B Instruct 2512 Q4_K_M is now the default conversation model in the
live desktop application and Windows component setup. The source chat preview
and desktop radio share the same checked native-tool implementation.

## Conversation and delivery

- Read position, gaps, classification, fuel/range/to-finish, lap number, last/best
  lap times, race progress, qualitative closing movement and bounded history.
- Refresh selected telemetry when the radio grants a reply its delivery slot.
  Only successfully delivered readings enter dialogue history and radio recall.
- Preserve Chatterbox's shared output queue, priorities, interruption, output
  device selection and separate PTT WAV cues.
- Source GPU smoke replies took about 0.19-0.51 seconds before speech generation.
  This is a small synthetic smoke run, not a general accuracy/performance guarantee.

## Upgrade from 0.1.0, 0.1.1 or 0.2.0

1. Install the update over the existing installation. Profiles, preferences,
   push-to-talk bindings and audio settings remain intact.
2. Select **Set up local AI components** when the new model is missing. Setup
   verifies and reuses existing speech assets, downloads the 2.15 GB Ministral
   weights, then restarts Pitward. Fresh setup totals about 7.0 GB.
3. Start the engineer. After its owned server successfully loads Ministral,
   Pitward verifies both models, records the migration and removes the known
   obsolete 1.10 GB LFM file from its managed component folder.

Failed setup or model startup retains LFM. Locked files are retained and cleanup
retries on a later start. Source/experimental/custom model files are preserved.
Cleanup does not remove speech assets, profiles, settings, logs or recordings.

Ministral weights use Apache-2.0 terms. The unchanged CPU llama.cpp runtime uses
MIT terms; Qwen3-ASR and Chatterbox retain their existing pinned assets/licenses.

## Known limitations

- Windows packages still use CPU conversation/speech runtimes. The source CUDA
  timings do not represent installed CPU performance; cold CPU questions can
  take about 19 seconds before speech. Packaged GPU support remains future work.
- English is the primary tested conversation language. Turkish position/lap-time
  smoke replies work, but Turkish radio checks may be misclassified as unavailable.
- Ambiguous requests, unsupported race assessments and recall topic selection
  still need quality work. Pit strategy, opponent lap pace and car controls
  remain unavailable.
- A single source Chatterbox startup OSError was observed in live testing; retry
  succeeded and all five subsequent replies played. Its exact cause is unknown.

The published v0.2.0 tag and assets remain unchanged.

# CE-04e routing decision

Decision recorded 2026-09-24: retain the existing Qwen-v1 live planner. Do not promote
MiniLM, Laya, Qwen-v2, or an automatic fallback chain. The full content-free screening
artifact is [here](evaluations/ce04-dialogue-screening-2026-09-24.json).

This means “best current production choice,” not “universally best model.” Qwen-v1 is the
only route with acceptable driver-reported live behavior. The stricter dialogue pilot was
designed primarily to reject unsafe replacements and is too small to qualify a release:
68 judged locked turns, correlated translations, typed text, one high-end CPU, and no
simulator/audio co-residency.

| Candidate | Complete turns | Accepted-plan accuracy | Answerable coverage | Clarification | Warm total p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniLM | 0/68 | no accepted plans | 0/46 | 0/8 | 590 ms |
| Laya multilingual | 0/68 | no accepted plans | 0/46 | 0/8 | 3,758 ms |
| Qwen-v2 full-context | 17/68 | 19/30 (63.3%) | 16/46 (34.8%) | 1/8 | 7,897 ms |

MiniLM and Laya abstained on every locked turn at their frozen diagnostic operating
points. That rejects these zero-shot full-plan head compositions; it does not prove the
models have no useful narrower or fine-tuned role. Qwen-v2 accepted more turns but missed
the frozen quality gates and recorded 38 inference/validation errors. Its prompt/contract
is not a production replacement.

The Windows report initially sampled the virtual-environment launcher PID, so its tiny RAM
numbers are invalid and deliberately excluded from the published decision. The dedicated
worker probe resolves the owned child PID; combined simulator/ASR/TTS residency is still a
CE-03/08 measurement. Timing above includes all candidate work and is usable as screening
evidence, though the machine was not an isolated benchmark host.

CE-05 therefore extends the retained local Qwen model conservatively: the default remains
the established v1 path, while an explicit preview adds only bounded dialogue fields and
deterministic composition. It is not a promotion of Qwen-v2. No small-model primary or
fallback remains resident, and critical calls continue to bypass conversation inference.

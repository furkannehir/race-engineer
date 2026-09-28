# CE-04 dialogue suite, revision 2026-09-24.1

Synthetic English/Turkish typed dialogue, reviewed against the CE-04a behavior contract.
Frozen before candidate locked runs; no real driver transcripts. The CE-02 fixtures and
scores are unchanged. This is an initial **small rejection/screening suite**, not enough
independent evidence to establish a universal 95% accuracy claim.

Calibration: 10 bilingual scenario families, 20 dialogues, 48 turns.
Locked: 15 bilingual scenario families, 30 dialogues, 70 turns (68 inference turns;
two stale-input guards excluded from model accuracy). Translations stay in the same
family/split. The family is a race/conversation situation, not a target label.

Split review: literal cross-split matches are rejected after Unicode/punctuation
normalization. Family IDs cannot use split/language prefixes. Semantic review groups
translations and equivalent narratives together: calibration's explicit lap recap,
rear-to-front trend correction, dirty-move/gap repeat and missing-fuel/lap combination
are distinct from locked interrupted-position replay, slot replacement, reconnection,
position/ambiguous-chaser partial answer, language switching and expired clarification.
Both splits deliberately share supported query labels and conversational acts. Some
elementary meanings necessarily recur (e.g. asking fuel quantity); this is not a claim
of topic-disjoint or statistically independent real-world generalization. Prompt
instructions use label definitions, not locked utterances. Broader independently
authored evidence is required before promotion. Exact checks cannot prove semantic
separation, and generated paraphrase volume must not inflate sample size.

Expected plans were annotated before inference. Only `stint-status/status` allows two
predeclared equivalent forms: reuse fuel topic or explicitly request fuel quantity.
Part IDs/order and adapter metadata do not affect semantic equality; query multiplicity,
references, language, mode, acts, clarification and abstention do. Inputs with the same
short wording may legitimately recur within a split and have context-specific meaning.
Local default language follows the bilingual scenario; explicit preference wins.

Metrics are frozen as follows:

- Accepted-plan accuracy: complete semantic matches / non-abstaining proposals. A genuine
  targeted clarification is a proposal. Abstentions/errors are separately counted.
- Answerable coverage: completely correct turns / judged turns with at least one
  independently available expected factual part. Wrong extra parts or dropped parts fail.
- Clarification accuracy: correct full turns / judged turns requiring clarification.
- Complete turn: acceptable full semantics AND expected outcome/answer-status multiset/
  clarification/language AND fresh fixture-grounded numbers/units/opponent identity.
- Whole dialogue: every turn including guard-only turns passes. Predicted state carries
  forward during locked evaluation; it is not repaired with expected labels.
- Category/language counts and Wilson intervals accompany rates. These are descriptive
  intervals at turn level; correlated turns/translations are not independent samples.
- Guard-only cases cannot inflate model accuracy. An always-abstain control must fail
  answerable coverage. Numeric truth checks are independent of production fact retrieval.

Encoder calibration alone uses teacher-forced state to isolate threshold choice from
earlier mistakes. Nine predeclared (threshold, margin) pairs use thresholds .5/.65/.8
and margins 0/.1/.2. Maximize correctly accepted plans after preferring configurations
meeting per-language 95% accuracy and 70% coverage. If none qualify, select a diagnostic
operating point and mark the candidate unqualified. These acceptance thresholds do
not turn raw model scores into calibrated probabilities. Locked runs use the frozen
calibration artifact and their own predicted state.

No threshold, prompt, fixture or expected outcome may be tuned after viewing a locked
result and then presented as the same held-out test. Corrections to infrastructure
must be disclosed; altered model behavior requires a new dataset revision/holdout.

This suite has no audio generation, playback, observed ASR confidence, simulator frame
timing or actual critical-radio interruption. Existing deterministic tests cover those
guard boundaries separately. These scores cannot qualify CE-05 live behavior by themselves.

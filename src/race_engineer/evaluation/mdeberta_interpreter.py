"""Experimental zero-shot NLI hypotheses and scoring; never a live radio router."""

import math
from collections.abc import Mapping, Sequence

from race_engineer.evaluation.radio_interpreter import PURPOSES, SCOPES, TOPICS

MODEL_REPO = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
MODEL_REVISION = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
MODEL_FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "spm.model", "README.md", "LICENSE",
)
TOPIC_THRESHOLD = 0.5

_EN = (
    "This message is a radio check asking whether the listener is there and can hear the speaker.",
    "This message is a request for racing information.",
    "This message expresses feelings or frustration without requesting information.",
    "This message combines a complaint with a request for information.",
    "This message clarifies or corrects a previous statement.",
    "This message is about something unrelated to racing or radio communication.",
    "This message is about the current situation.",
    "This message is about something that happened in the past.",
    "This message is about a future or hypothetical situation.",
    "This message has no reference to a moment in time.",
    "This message has an ambiguous reference to time.",
    "This message is about fuel.",
    "This message is about race position or standings.",
    "This message is about gaps between cars or relative pace.",
    "This message is about pit stops or pit strategy.",
    "This message is about car speed.",
    "This message is about how the race is going overall.",
)
_TR = (
    "Bu mesaj karşıdakinin orada olup olmadığını ve konuşanı duyup duymadığını "
    "soran bir telsiz kontrolü.",
    "Bu mesaj yarış hakkında bilgi isteyen bir soru.",
    "Bu mesaj bilgi istemeden duygu veya hayal kırıklığı ifade ediyor.",
    "Bu mesaj hem bir şikayet hem de bilgi isteyen bir soru içeriyor.",
    "Bu mesaj daha önce söylenen bir şeyi açıklıyor veya düzeltiyor.",
    "Bu mesaj yarış veya telsiz iletişimiyle ilgisiz bir konu hakkında.",
    "Bu mesaj şu anki durum hakkında.",
    "Bu mesaj geçmişte yaşanan bir olay hakkında.",
    "Bu mesaj gelecekteki veya varsayımsal bir durum hakkında.",
    "Bu mesaj belirli bir zamana atıfta bulunmuyor.",
    "Bu mesajın hangi zamana atıfta bulunduğu belirsiz.",
    "Bu mesaj yakıt hakkında.",
    "Bu mesaj yarıştaki sıra veya sıralama hakkında.",
    "Bu mesaj araçlar arasındaki fark veya göreceli tempo hakkında.",
    "Bu mesaj pit stopları veya pit stratejisi hakkında.",
    "Bu mesaj araç hızı hakkında.",
    "Bu mesaj yarışın genel olarak nasıl gittiği hakkında.",
)


def hypotheses(language: str) -> dict[str, dict[str, str]]:
    """Declarative hypotheses, not instructions or a bounded driver command grammar."""
    if language not in {"en", "tr"}:
        raise ValueError("unsupported hypothesis language")
    values = iter(_EN if language == "en" else _TR)
    return {
        group: {label: next(values) for label in labels}
        for group, labels in (("purpose", PURPOSES), ("time_scope", SCOPES), ("topics", TOPICS))
    }


def nli_indices(labels: Mapping[int, str]) -> tuple[int, int, int]:
    """Read checkpoint metadata; never assume the order of the three NLI logits."""
    normalized = {value.lower(): key for key, value in labels.items()}
    if set(normalized) != {"entailment", "neutral", "contradiction"}:
        raise ValueError("checkpoint does not declare the expected NLI labels")
    indices = (normalized["entailment"], normalized["neutral"], normalized["contradiction"])
    if set(indices) != {0, 1, 2}:
        raise ValueError("invalid NLI label indices")
    return indices


def decode_nli(
    rows: Sequence[Sequence[float]], language: str,
) -> dict[str, object]:
    """Rows are E/N/C probabilities. Independent full three-way E scores are uncalibrated.

    Purpose/scope use the highest entailment score (a forced diagnostic choice).
    Topics use E >= 0.5, retaining neutral rather than normalizing it away.
    No keyword correction, social/topic cleanup, or fitted abstention threshold is applied.
    """
    catalog = hypotheses(language)
    if len(rows) != sum(len(group) for group in catalog.values()):
        raise ValueError("unexpected number of NLI rows")
    scores: dict[str, dict[str, float]] = {}
    raw: dict[str, dict[str, dict[str, float]]] = {}
    row_iterator = iter(rows)
    for group, candidates in catalog.items():
        scores[group], raw[group] = {}, {}
        for label in candidates:
            row = tuple(next(row_iterator))
            if (len(row) != 3 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in row)
                    or not math.isclose(sum(row), 1, abs_tol=1e-5)):
                raise ValueError("invalid NLI probabilities")
            scores[group][label] = row[0]
            raw[group][label] = dict(zip(
                ("entailment", "neutral", "contradiction"), row, strict=True,
            ))
    return {
        "interpretation": {
            "purpose": max(scores["purpose"].items(), key=lambda item: item[1])[0],
            "time_scope": max(scores["time_scope"].items(), key=lambda item: item[1])[0],
            "topics": [label for label in TOPICS if scores["topics"][label] >= TOPIC_THRESHOLD],
        },
        "purpose_scores_uncalibrated": scores["purpose"],
        "scope_scores_uncalibrated": scores["time_scope"],
        "topic_scores_uncalibrated": scores["topics"],
        "nli_probabilities": raw,
    }

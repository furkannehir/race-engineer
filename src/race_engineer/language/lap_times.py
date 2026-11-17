"""Deterministic race-radio pronunciation of a lap duration stored in seconds."""

import math

_EN_SMALL = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_TR_SMALL = ("sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz")
_TR_TENS = ("", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan")


def _integer_words(value: int, language: str) -> str:
    small = _TR_SMALL if language == "tr" else _EN_SMALL
    if value < (10 if language == "tr" else 20):
        return small[value]
    if value < 100:
        tens = _TR_TENS if language == "tr" else _EN_TENS
        return tens[value // 10] + (" " + small[value % 10] if value % 10 else "")
    for scale, en, tr in ((1_000_000_000, "billion", "milyar"),
                          (1_000_000, "million", "milyon"), (1000, "thousand", "bin"),
                          (100, "hundred", "yüz")):
        if value >= scale:
            count, remainder = divmod(value, scale)
            prefix = "" if language == "tr" and count == 1 and scale in {100, 1000} else (
                _integer_words(count, language) + " "
            )
            return prefix + (tr if language == "tr" else en) + (
                " " + _integer_words(remainder, language) if remainder else ""
            )
    raise ValueError("invalid_integer")


def format_lap_time(seconds: int | float, language: str = "en") -> str:
    """Say 92.5 as 'one thirty two point five', preserving up to three decimals.

    Round before splitting minutes/seconds so a millisecond carry cannot produce
    a spoken sixty-second remainder. This is presentation, never race arithmetic.
    """
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("lap time must be positive and finite")
    if language not in {"en", "tr"}:
        raise ValueError("unsupported lap-time language")
    whole, fraction = f"{seconds:.3f}".split(".")
    minutes, remainder = divmod(int(whole), 60)
    fraction = fraction.rstrip("0")
    small = _TR_SMALL if language == "tr" else _EN_SMALL
    parts = []
    if minutes:
        parts.append(_integer_words(minutes, language))
        if remainder < 10:
            parts.append("oh" if language == "en" and remainder else small[0])
    parts.append(_integer_words(remainder, language))
    if fraction:
        parts.append("virgül" if language == "tr" else "point")
        parts.extend(small[int(digit)] for digit in fraction)
    return " ".join(parts)

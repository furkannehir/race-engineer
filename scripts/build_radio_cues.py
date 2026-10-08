"""Rebuild the selected soft-squelch/beep pair, reversing the beep for release."""

import argparse
import importlib
import math
import random
import struct
import wave
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATE = 48000
ASSETS = ROOT / "src/race_engineer/stt/assets"


def soft_squelch(*, closing: bool) -> bytes:
    """The selected soft variant: original filtered noise and a small switch transient."""
    duration = 0.46 if closing else 0.34
    count = round(duration * RATE)
    rng = random.Random(947 + (1 if closing else 0))
    low_alpha = math.exp(-2 * math.pi * 420 / RATE)
    high_alpha = math.exp(-2 * math.pi * 2850 / RATE)
    low = high = smooth = 0.0
    filtered = []
    for _ in range(count):
        white = rng.gauss(0, 1)
        low = low_alpha * low + (1 - low_alpha) * white
        high = high_alpha * high + (1 - high_alpha) * white
        smooth = high_alpha * smooth + (1 - high_alpha) * (high - low)
        filtered.append(smooth)
    dc = sum(filtered) / count
    values = []
    for index, sample in enumerate(filtered):
        elapsed = index / RATE
        progress = index / (count - 1)
        attack = min(1.0, elapsed / 0.005)
        release = min(1.0, (count - 1 - index) / (RATE * 0.024))
        edge = math.sin(attack * math.pi / 2) ** 2
        edge *= math.sin(release * math.pi / 2) ** 2
        if closing:
            body = 0.20 + 0.65 * math.sin(math.pi * progress) ** 2
            body += 0.12 * math.exp(-elapsed / 0.035)
        else:
            body = 0.13 + 0.85 * math.exp(-elapsed / 0.075)
        switch = 0.09 * math.exp(-elapsed / 0.0018) * (1 - elapsed / 0.0018)
        values.append(((sample - dc) * body + switch) * edge)
    rms = math.sqrt(sum(value * value for value in values) / count)
    peak = max(abs(value) for value in values)
    gain = min(0.042 / rms, 0.23 / peak)
    return pcm([value * gain for value in values])


def fade(samples: list[float], duration_s: float = 0.004) -> list[float]:
    result = samples.copy()
    count = min(round(RATE * duration_s), len(result) // 2)
    for index in range(count):
        gain = index / max(1, count - 1)
        result[index] *= gain
        result[-index - 1] *= gain
    return result


def decode_beep(path: Path) -> list[float]:
    try:
        soundfile = importlib.import_module("soundfile")
    except ImportError as error:
        raise RuntimeError("Run this script with the existing STT runtime's Python.") from error
    data, source_rate = soundfile.read(str(path), always_2d=True, dtype="float64")
    mono = [float(sum(frame) / len(frame)) for frame in data]
    if not mono or len(mono) / source_rate > 1:
        raise ValueError("beep must contain audio and be at most one second")
    # Preserve the onset: remove only the first 20 ms of this file's quiet preroll.
    mono = mono[min(round(source_rate * 0.020), len(mono) // 4) :]
    samples = []
    for index in range(round(len(mono) * RATE / source_rate)):
        position = index * source_rate / RATE
        left = min(int(position), len(mono) - 1)
        right = min(left + 1, len(mono) - 1)
        fraction = position - left
        samples.append(mono[left] * (1 - fraction) + mono[right] * fraction)
    return fade(samples)


def pcm(samples: list[float]) -> bytes:
    values = [round(value * 32767) for value in samples]
    return struct.pack(f"<{len(values)}h", *values)


def mixed(static: bytes, beep: list[float], offset_s: float) -> bytes:
    noise = struct.unpack(f"<{len(static) // 2}h", static)
    offset = round(RATE * offset_s)
    count = max(len(noise), offset + len(beep))
    if count > RATE:
        raise ValueError("mixed cue must stay within the player's one-second bound")
    peak = max(map(abs, beep))
    if not peak:
        raise ValueError("beep is silent")
    output = [0.0] * count
    for index, value in enumerate(noise):
        output[index] = value / 32768 * 0.60
    for index, value in enumerate(beep):
        output[offset + index] += value * (0.18 / peak)
    gain = min(1.0, 0.23 / max(map(abs, output)))
    return pcm(fade([value * gain for value in output]))


def write_wav(path: Path, audio: bytes) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(audio)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ASSETS / "radio_beep.mp3")
    parser.add_argument("--output", type=Path, default=ASSETS)
    parser.add_argument("--replace", action="store_true", help="replace an existing cue pair")
    args = parser.parse_args()
    targets = {part: args.output / f"radio-{part}.wav" for part in ("open", "close")}
    if not args.replace and any(path.exists() for path in targets.values()):
        parser.error("cue output already exists; choose another --output or use --replace")
    source_digest = sha256(args.source.read_bytes()).hexdigest()
    forward = decode_beep(args.source)
    opening = mixed(soft_squelch(closing=False), forward, 0.030)
    closing = mixed(soft_squelch(closing=True), forward[::-1], 0.120)
    if sha256(args.source.read_bytes()).hexdigest() != source_digest:
        raise RuntimeError("source MP3 changed during cue generation")
    args.output.mkdir(parents=True, exist_ok=True)
    write_wav(targets["open"], opening)
    write_wav(targets["close"], closing)
    print(f"Opening: {len(opening) / (RATE * 2):.3f}s; closing: {len(closing) / (RATE * 2):.3f}s")
    print(f"Original MP3 preserved: {source_digest}")
    print(f"Cue pair: {args.output}")


if __name__ == "__main__":
    main()

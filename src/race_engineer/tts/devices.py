"""Enumerate output devices without opening an audio stream."""

import importlib

from race_engineer.core.speech_output import SpeechOutputError


def output_devices() -> list[dict[str, object]]:
    try:
        sd = importlib.import_module("sounddevice")
        hosts = sd.query_hostapis()
        default = sd.default.device[1]
        return [
            {
                "index": index,
                "name": device["name"],
                "host_api": hosts[device["hostapi"]]["name"],
                "output_channels": device["max_output_channels"],
                "default_sample_rate_hz": device["default_samplerate"],
                "default": index == default,
            }
            for index, device in enumerate(sd.query_devices())
            if device["max_output_channels"] > 0
        ]
    except (ImportError, OSError) as error:
        raise SpeechOutputError("output_dependency_missing") from error
    except Exception as error:
        raise SpeechOutputError("output_enumeration_failed") from error

from types import SimpleNamespace

import pytest

from race_engineer.telemetry.iracing.source import PyIrSdkSource


class FakeSdk:
    def __init__(self) -> None:
        self.is_connected = True
        self.session_info_update = 7
        self.var_headers_names = [
            "SessionUniqueID",
            "SessionNum",
            "SessionTick",
            "SessionTime",
            "PlayerCarIdx",
            "Speed",
        ]
        self.values: dict[str, object] = {
            "SessionUniqueID": 123,
            "SessionNum": 1,
            "SessionTick": 9,
            "SessionTime": 4.5,
            "PlayerCarIdx": 0,
            "Speed": 12.5,
            "DriverInfo": {
                "Drivers": [
                    {
                        "CarIdx": 0,
                        "UserID": 99,
                        "UserName": "Must Not Be Retained",
                        "CarIsPaceCar": 0,
                    }
                ]
            },
            "SessionInfo": {"Sessions": [{"SessionNum": 1, "SessionType": "Race"}]},
        }
        self.freeze_calls = 0
        self.unfreeze_calls = 0
        self.shutdown_calls = 0

    def startup(self) -> bool:
        return True

    def freeze_var_buffer_latest(self) -> None:
        self.freeze_calls += 1

    def unfreeze_var_buffer_latest(self) -> None:
        self.unfreeze_calls += 1

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_sdk_source_freezes_sample_and_discards_personal_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = FakeSdk()
    monkeypatch.setattr(
        "race_engineer.telemetry.iracing.source.importlib.import_module",
        lambda name: SimpleNamespace(IRSDK=lambda: sdk),
    )
    source = PyIrSdkSource()
    assert source.connect() is True

    result = source.read()
    sample = result.sample
    assert sample.session_unique_id == 123
    assert sample.speed_mps == 12.5
    assert sample.session_type == "Race"
    assert sample.drivers[0].user_id == 99
    assert "Must Not Be Retained" not in sample.model_dump_json()
    assert sdk.freeze_calls == sdk.unfreeze_calls == 1
    assert result.metrics.metadata_refreshed is True
    assert result.metrics.total_ms >= 0

    second_result = source.read()
    assert second_result.metrics.metadata_refreshed is False

    source.close()
    assert sdk.shutdown_calls == 1


def test_sdk_source_unfreezes_buffer_when_read_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = FakeSdk()
    del sdk.values["Speed"]
    monkeypatch.setattr(
        "race_engineer.telemetry.iracing.source.importlib.import_module",
        lambda name: SimpleNamespace(IRSDK=lambda: sdk),
    )
    source = PyIrSdkSource()
    with pytest.raises(KeyError):
        source.read()
    assert sdk.freeze_calls == sdk.unfreeze_calls == 1

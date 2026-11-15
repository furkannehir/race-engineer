import importlib.metadata
import io
import json
from types import SimpleNamespace

import pytest

from race_engineer.tts import chatterbox_worker as worker
from race_engineer.tts_metadata import CHATTERBOX_VERSION


@pytest.fixture
def protocol(monkeypatch):
    # Production send() deliberately bypasses redirect_stdout for clean JSON-lines.
    output = io.StringIO()
    monkeypatch.setattr(worker.sys, "__stdout__", output)
    return output


def test_runtime_check_exercises_dependencies_without_loading_voice_weights(monkeypatch):
    calls = []
    modules = {
        "torch": SimpleNamespace(__version__="test-cpu"),
        "chatterbox.mtl_tts": SimpleNamespace(
            ChatterboxMultilingualTTS=SimpleNamespace(
                get_supported_languages=lambda: {"en": "English", "tr": "Turkish"}
            )
        ),
        "perth.perth_net.perth_net_implicit.perth_watermarker": SimpleNamespace(
            PerthImplicitWatermarker=lambda: None
        ),
    }

    def load(name):
        calls.append(name)
        return modules[name]

    monkeypatch.setattr(worker.importlib, "import_module", load)
    monkeypatch.setattr(
        worker.importlib.metadata,
        "version",
        lambda name: CHATTERBOX_VERSION if name == "chatterbox-tts" else "test-transformers",
    )
    result = worker.check_runtime()
    assert result["runtime_ready"] is True and result["torch_version"] == "test-cpu"
    assert calls == list(modules)


def test_runtime_check_cli_needs_no_model_and_keeps_protocol_output_clean(
    monkeypatch, capsys, protocol
):
    monkeypatch.setattr("sys.argv", ["worker", "--check-runtime"])

    def check():
        print("dependency startup output")
        return {"runtime_ready": True}

    monkeypatch.setattr(worker, "check_runtime", check)
    monkeypatch.setattr(
        worker, "load_model", lambda *_: pytest.fail("No voice weights in this check")
    )
    worker.main()
    output = capsys.readouterr()
    assert json.loads(protocol.getvalue()) == {"runtime_ready": True}
    assert output.out == ""
    assert "dependency startup output" in output.err


@pytest.mark.parametrize(
    "error,code",
    [
        (
            importlib.metadata.PackageNotFoundError("private package detail"),
            "tts_package_metadata_missing",
        ),
        (ModuleNotFoundError("private dependency detail"), "tts_dependency_missing"),
        (IndexError("private startup detail"), "tts_model_load_failed"),
    ],
)
def test_runtime_check_failure_has_nonzero_exit_and_sanitized_cause(
    monkeypatch, capsys, protocol, error, code
):
    monkeypatch.setattr("sys.argv", ["worker", "--check-runtime"])

    def fail():
        raise error

    monkeypatch.setattr(worker, "check_runtime", fail)
    with pytest.raises(SystemExit) as exit_error:
        worker.main()
    assert exit_error.value.code == 1
    output = capsys.readouterr()
    assert json.loads(protocol.getvalue()) == {
        "error": code,
        "exception_type": type(error).__name__,
    }
    assert "private" not in protocol.getvalue() + output.out + output.err


def test_explicit_startup_diagnostics_include_traceback_only_on_stderr(
    monkeypatch, capsys, protocol
):
    monkeypatch.setattr("sys.argv", ["worker", "--check-runtime", "--diagnostics"])
    monkeypatch.setattr(
        worker, "check_runtime", lambda: (_ for _ in ()).throw(IndexError("diagnostic cause"))
    )
    with pytest.raises(SystemExit):
        worker.main()
    output = capsys.readouterr()
    assert json.loads(protocol.getvalue())["error"] == "tts_model_load_failed"
    assert "IndexError: diagnostic cause" in output.err
    assert "diagnostic cause" not in protocol.getvalue() + output.out

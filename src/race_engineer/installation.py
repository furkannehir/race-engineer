"""Source and frozen-installation paths for Pitward."""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from race_engineer.config import AppConfig

_PRODUCT_DIRECTORY = "Pitward"


@dataclass(frozen=True)
class Installation:
    """All roots needed by the desktop application and packaged components."""

    frozen: bool
    install_root: Path
    bundle_root: Path
    data_root: Path

    @property
    def components_root(self) -> Path:
        return self.data_root / "components"

    @property
    def config_path(self) -> Path:
        return self.bundle_root / "config" / "default.toml"

    @property
    def settings_path(self) -> Path:
        return self.data_root / "control-panel.json"

    @property
    def logs_root(self) -> Path:
        return self.data_root / "logs" if self.frozen else self.install_root / "logs"

    @property
    def component_installer_path(self) -> Path:
        return self.install_root / "PitwardComponents.exe"

    @classmethod
    def discover(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        executable: Path | None = None,
        bundle_root: Path | None = None,
        frozen: bool | None = None,
    ) -> "Installation":
        env = os.environ if environ is None else environ
        is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
        working_root = (cwd or Path.cwd()).resolve()
        executable_path = (executable or Path(sys.executable)).resolve()
        install_root = executable_path.parent if is_frozen else working_root
        if bundle_root is None:
            detected = getattr(sys, "_MEIPASS", None) if is_frozen else None
            resources = Path(detected).resolve() if detected else install_root
        else:
            resources = bundle_root.resolve()
        override = env.get("PITWARD_DATA_DIR")
        if override:
            data_root = Path(override).expanduser().resolve()
        elif is_frozen:
            local = env.get("LOCALAPPDATA")
            if not local:
                raise OSError("LOCALAPPDATA is unavailable; set PITWARD_DATA_DIR")
            data_root = (Path(local) / _PRODUCT_DIRECTORY).resolve()
        else:
            data_root = working_root / "data"
        return cls(is_frozen, install_root, resources, data_root)

    def apply_to(self, config: AppConfig) -> AppConfig:
        """Resolve packaged runtime and writable paths without changing source-mode config."""
        if not self.frozen:
            return config
        components = self.components_root
        conversation = components / "conversation"
        stt = components / "stt"
        voices = components / "tts" / "voices"
        workers = self.install_root / "workers"
        runtime = config.conversation.runtime.model_copy(
            update={
                "cpu_executable_path": conversation
                / "llama-b10964-cpu"
                / "llama-server.exe",
                "model_path": conversation / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            }
        )
        return config.model_copy(
            update={
                "paths": config.paths.model_copy(
                    update={
                        "data_dir": self.data_root,
                        "database_path": self.data_root / "race_engineer.sqlite3",
                    }
                ),
                "conversation": config.conversation.model_copy(update={"runtime": runtime}),
                "stt": config.stt.model_copy(
                    update={
                        "worker_path": workers
                        / "PitwardSTTWorker"
                        / "PitwardSTTWorker.exe",
                        "model_path": stt / "Qwen3-ASR-0.6B-hf",
                    }
                ),
                "radio_tts": config.radio_tts.model_copy(
                    update={
                        "worker_path": workers
                        / "PitwardTTSWorker"
                        / "PitwardTTSWorker.exe",
                        "english_model_path": voices / "en_US-ljspeech-high.onnx",
                        "turkish_model_path": voices / "tr_TR-dfki-medium.onnx",
                    }
                ),
            }
        )

    def missing_required_components(self, config: AppConfig) -> tuple[Path, ...]:
        """Return packaged runtime files that must exist before a live session can start."""
        if not self.frozen:
            return ()
        candidates: tuple[Path, ...] = (
            config.conversation.runtime.cpu_executable_path,
            config.conversation.runtime.model_path,
            config.stt.worker_path
            or self.install_root / "workers/PitwardSTTWorker/PitwardSTTWorker.exe",
            config.stt.model_path / "config.json",
            config.stt.model_path / "model.safetensors",
            config.radio_tts.worker_path
            or self.install_root / "workers/PitwardTTSWorker/PitwardTTSWorker.exe",
            config.radio_tts.english_model_path,
            config.radio_tts.english_model_path.with_suffix(".onnx.json"),
            config.radio_tts.turkish_model_path,
            config.radio_tts.turkish_model_path.with_suffix(".onnx.json"),
        )
        return tuple(path for path in candidates if not path.is_file())

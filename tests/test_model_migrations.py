import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from race_engineer import model_migrations as migrations
from race_engineer.config import ConversationConfig
from race_engineer.installation import Installation

NEW = b"verified replacement weights"
OLD = b"verified obsolete weights"


def setup(tmp_path, monkeypatch):
    layout = Installation(True, tmp_path / "app", tmp_path / "bundle", tmp_path / "user")
    _, artifacts = migrations.load_manifest()
    definition = migrations.load_migrations()[0]
    replacement = next(a for a in artifacts if a.artifact_id == definition.replacement_id)
    replacement = replace(replacement, size=len(NEW), sha256=hashlib.sha256(NEW).hexdigest())
    retired = definition.retired[0].model_copy(
        update={
            "size": len(OLD),
            "sha256": hashlib.sha256(OLD).hexdigest(),
        }
    )
    definition = definition.model_copy(update={"retired": (retired,)})
    monkeypatch.setattr(migrations, "load_manifest", lambda: ("4", (replacement,)))
    monkeypatch.setattr(migrations, "load_migrations", lambda: (definition,))
    current = layout.components_root / replacement.destination
    old = layout.components_root / retired.destination
    current.parent.mkdir(parents=True)
    current.write_bytes(NEW)
    old.write_bytes(OLD)
    config = ConversationConfig()
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update={"model_path": current}),
        }
    )
    return layout, config, current, old, definition


def test_success_records_verified_migration_before_deleting_only_known_weights(
    tmp_path, monkeypatch
):
    layout, config, current, old, definition = setup(tmp_path, monkeypatch)
    unrelated = [
        layout.data_root / "profile.sqlite3",
        layout.data_root / "control-panel.json",
        old.with_name("custom-backup.gguf"),
        layout.components_root / "stt/weights.bin",
        layout.components_root / "tts/voice.bin",
    ]
    for path in unrelated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep")
    unlink = Path.unlink

    def observed_unlink(path, *args, **kwargs):
        if path == old:
            journal = json.loads((layout.components_root / migrations._JOURNAL).read_text())
            record = journal["migrations"][definition.migration_id]
            assert record["state"] == "cleanup_pending"
            assert record["replacement_sha256"] == hashlib.sha256(NEW).hexdigest()
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", observed_unlink)
    result = migrations.retire_loaded_model(layout, config, current)
    assert result.status == "completed" and result.bytes_freed == len(OLD)
    assert result.removed == (definition.retired[0].artifact_id,)
    assert not old.exists() and current.read_bytes() == NEW
    assert all(path.read_bytes() == b"keep" for path in unrelated)
    journal = json.loads((layout.components_root / migrations._JOURNAL).read_text())
    assert journal["migrations"][definition.migration_id]["state"] == "completed"
    assert migrations.retire_loaded_model(layout, config, current).status == "not_applicable"


@pytest.mark.parametrize("which", ["replacement", "retired"])
def test_changed_or_corrupt_weights_are_never_deleted(tmp_path, monkeypatch, which):
    layout, config, current, old, _ = setup(tmp_path, monkeypatch)
    path = current if which == "replacement" else old
    path.write_bytes(b"x" * path.stat().st_size)
    assert migrations.retire_loaded_model(layout, config, current).status == "deferred"
    assert old.exists() and current.exists()
    assert not (layout.components_root / migrations._JOURNAL).exists()


def test_missing_replacement_retains_old_weights(tmp_path, monkeypatch):
    layout, config, current, old, _ = setup(tmp_path, monkeypatch)
    current.unlink()
    assert migrations.retire_loaded_model(layout, config, current).status == "deferred"
    assert old.read_bytes() == OLD


@pytest.mark.parametrize("kind", ["source", "custom", "legacy"])
def test_source_launches_custom_paths_and_legacy_profiles_are_preserved(
    tmp_path, monkeypatch, kind
):
    layout, config, current, old, _ = setup(tmp_path, monkeypatch)
    if kind == "source":
        layout = replace(layout, frozen=False)
    elif kind == "custom":
        custom = tmp_path / "my-model.gguf"
        custom.write_bytes(NEW)
        config = config.model_copy(
            update={
                "runtime": config.runtime.model_copy(update={"model_path": custom}),
            }
        )
        current = custom
    else:
        config = config.model_copy(
            update={"model": "LiquidAI/LFM3-1.7B", "context_planner": "lfm3-native-v1"}
        )
    assert migrations.retire_loaded_model(layout, config, current).status == "not_applicable"
    assert old.read_bytes() == OLD


def test_locked_old_weights_are_retained_and_cleanup_retries(tmp_path, monkeypatch):
    layout, config, current, old, _ = setup(tmp_path, monkeypatch)
    unlink = Path.unlink

    def locked(path, *args, **kwargs):
        if path == old:
            raise PermissionError("test locked model")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", locked)
        assert migrations.retire_loaded_model(layout, config, current).status == "deferred"
    assert old.read_bytes() == OLD
    assert migrations.retire_loaded_model(layout, config, current).status == "completed"
    assert not old.exists()


def test_unwritable_migration_record_prevents_cleanup(tmp_path, monkeypatch):
    layout, config, current, old, _ = setup(tmp_path, monkeypatch)

    def failed(*args):
        raise PermissionError("test read-only migration record")

    monkeypatch.setattr(migrations, "_record", failed)
    assert migrations.retire_loaded_model(layout, config, current).status == "deferred"
    assert old.read_bytes() == OLD


def test_unknown_existing_migration_record_is_preserved(tmp_path, monkeypatch):
    layout, config, current, old, _ = setup(tmp_path, monkeypatch)
    journal = layout.components_root / migrations._JOURNAL
    journal.write_text('{"my_custom_file":true}')
    assert migrations.retire_loaded_model(layout, config, current).status == "deferred"
    assert journal.read_text() == '{"my_custom_file":true}'
    assert old.read_bytes() == OLD


@pytest.mark.parametrize("when", ["before", "after_record"])
def test_stop_during_migration_retains_old_model(tmp_path, monkeypatch, when):
    layout, config, current, old, _ = setup(tmp_path, monkeypatch)
    stopped = [when == "before"]
    record = migrations._record

    def stop_after_record(*args):
        record(*args)
        stopped[0] = True

    if when == "after_record":
        monkeypatch.setattr(migrations, "_record", stop_after_record)
    result = migrations.retire_loaded_model(layout, config, current, cancelled=lambda: stopped[0])
    assert result.status == "cancelled" and old.read_bytes() == OLD


def test_file_changed_after_verification_is_preserved(tmp_path, monkeypatch):
    import os

    layout, config, current, old, _ = setup(tmp_path, monkeypatch)
    record = migrations._record

    def changed(*args):
        record(*args)
        original = old.stat()
        os.utime(old, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))

    monkeypatch.setattr(migrations, "_record", changed)
    assert migrations.retire_loaded_model(layout, config, current).status == "deferred"
    assert old.read_bytes() == OLD


@pytest.mark.parametrize("target", ["root", "folder", "old", "journal"])
def test_redirected_paths_are_rejected_without_deleting_files(tmp_path, monkeypatch, target):
    layout, config, current, old, _ = setup(tmp_path, monkeypatch)
    redirected = {
        "root": layout.components_root,
        "folder": old.parent,
        "old": old,
        "journal": layout.components_root / migrations._JOURNAL,
    }[target]
    is_junction = Path.is_junction
    monkeypatch.setattr(Path, "is_junction", lambda path: path == redirected or is_junction(path))
    assert migrations.retire_loaded_model(layout, config, current).status == "deferred"
    assert current.read_bytes() == NEW and old.read_bytes() == OLD


def test_active_manifest_artifact_can_never_be_retired(tmp_path, monkeypatch):
    layout, config, current, old, _ = setup(tmp_path, monkeypatch)
    _, artifacts = migrations.load_manifest()
    protected = replace(
        artifacts[0],
        artifact_id="still-active",
        destination=Path("conversation/LFM3-1.7B-Q4_K_M.gguf"),
    )
    monkeypatch.setattr(migrations, "load_manifest", lambda: ("4", (*artifacts, protected)))
    assert migrations.retire_loaded_model(layout, config, current).status == "deferred"
    assert old.read_bytes() == OLD


def test_real_manifest_retirement_matches_the_previously_shipped_lfm_artifact():
    from scripts.setup_lfm3_trial import MODEL

    migration = migrations.load_migrations()[0]
    assert migration.retired[0].destination == "conversation/" + MODEL.name
    assert (migration.retired[0].size, migration.retired[0].sha256) == (MODEL.size, MODEL.sha256)
    assert migration.replacement_id in {a.artifact_id for a in migrations.load_manifest()[1]}

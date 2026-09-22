import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from race_engineer.cli import _history_action, _parse_profile_value, _profile_action
from race_engineer.core.contracts import PlaybackResult, PolicyDecision, PreferenceCommand
from race_engineer.core.enums import (
    PlaybackStatus,
    PolicyDecisionOutcome,
    PolicyDecisionReason,
    PreferenceScope,
    PreferenceSource,
)
from race_engineer.memory import (
    DriverMemoryError,
    DurableHistory,
    SqliteDriverProfileRepository,
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
ROOT = Path(__file__).parents[1]


def command(
    command_id: str,
    setting: str,
    value: object,
    *,
    scope: PreferenceScope = PreferenceScope.DRIVER,
    timestamp: datetime = NOW,
) -> PreferenceCommand:
    return PreferenceCommand(
        command_id=command_id,
        setting=setting,
        value=value,
        source=PreferenceSource.UI,
        timestamp=timestamp,
        scope=scope,
    )


def repository(tmp_path):
    return SqliteDriverProfileRepository(tmp_path / "driver.sqlite3", clock=lambda: NOW)


def decision(
    decision_id: str,
    *,
    session_id: str = "session-1",
    decided_at: datetime = NOW,
    approved: bool = True,
) -> PolicyDecision:
    return PolicyDecision(
        decision_id=decision_id,
        candidate_id=f"candidate-{decision_id}",
        session_id=session_id,
        source_sequence=1,
        decided_at=decided_at,
        outcome=(
            PolicyDecisionOutcome.APPROVED
            if approved
            else PolicyDecisionOutcome.SUPPRESSED
        ),
        reason=(
            PolicyDecisionReason.APPROVED
            if approved
            else PolicyDecisionReason.COOLDOWN
        ),
        priority=70,
        intent_id=f"intent-{decision_id}" if approved else None,
    )


def test_first_open_migrates_and_creates_one_safe_default_profile(tmp_path):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    assert profile.profile_id == "default"
    assert profile.display_name == "Default driver" and profile.is_default
    assert store.ensure_default_profile() == profile
    preferences = store.preferences(profile.profile_id)
    assert preferences.announce_position_changes
    assert not preferences.announce_pit_transitions
    assert preferences.reply_language == "auto" and preferences.updated_at is None

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0x52414345
        assert connection.execute("SELECT version FROM schema_migrations").fetchall() == [
            (1,),
            (2,),
        ]


def test_v1_database_migrates_without_changing_profile_preferences(tmp_path):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    expected = store.apply(
        profile.profile_id,
        command("turkish", "reply_language", "tr"),
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TABLE radio_outcomes")
        connection.execute("DROP TABLE policy_decisions")
        connection.execute("DELETE FROM schema_migrations WHERE version = 2")
        connection.execute("PRAGMA user_version = 1")

    migrated = repository(tmp_path)
    assert migrated.preferences(profile.profile_id) == expected
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name = 'policy_decisions'"
        ).fetchone() == (1,)


def test_explicit_preferences_survive_restart_with_source_audit(tmp_path):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    result = store.apply(
        profile.profile_id,
        command("position-off", "announce_position_changes", False),
    )
    result = store.apply(
        profile.profile_id,
        command(
            "turkish",
            "reply_language",
            "tr",
            timestamp=NOW + timedelta(seconds=1),
        ),
    )
    assert not result.announce_position_changes and result.reply_language == "tr"
    assert result.updated_at == NOW + timedelta(seconds=1)

    reopened = SqliteDriverProfileRepository(store.path, clock=lambda: NOW)
    assert reopened.preferences(profile.profile_id) == result
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute(
            "SELECT command_id, source, scope FROM preference_commands ORDER BY timestamp"
        ).fetchall()
    assert rows == [
        ("position-off", "ui", "driver"),
        ("turkish", "ui", "driver"),
    ]


def test_commands_are_idempotent_but_conflicting_reuse_is_rejected(tmp_path):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    first = command("same-command", "announce_pit_transitions", True)
    assert store.apply(profile.profile_id, first) == store.apply(profile.profile_id, first)
    with pytest.raises(DriverMemoryError, match="command_conflict"):
        store.apply(
            profile.profile_id,
            command("same-command", "announce_pit_transitions", False),
        )
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM preference_commands").fetchone()[0] == 1


def test_stale_command_cannot_overwrite_a_newer_preference(tmp_path):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    latest = command(
        "latest",
        "reply_language",
        "tr",
        timestamp=NOW + timedelta(seconds=2),
    )
    store.apply(profile.profile_id, latest)
    with pytest.raises(DriverMemoryError, match="command_stale"):
        store.apply(
            profile.profile_id,
            command("late-arrival", "reply_language", "en", timestamp=NOW),
        )
    assert store.preferences(profile.profile_id).reply_language == "tr"


def test_batch_validation_is_atomic(tmp_path):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    commands = (
        command("valid-first", "announce_pit_transitions", True),
        command("invalid-second", "reply_language", "de"),
    )
    with pytest.raises(ValueError):
        store.apply_many(profile.profile_id, commands)
    assert not store.preferences(profile.profile_id).announce_pit_transitions
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM preference_commands").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("announce_position_changes", 1),
        ("announce_pit_transitions", "yes"),
        ("reply_language", "de"),
        ("critical_calls", False),
    ],
)
def test_unknown_or_ill_typed_preferences_fail_before_writing(
    tmp_path, setting, value
):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    with pytest.raises(ValueError):
        store.apply(profile.profile_id, command("invalid", setting, value))
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM preference_commands").fetchone()[0] == 0


def test_non_driver_scope_is_explicitly_rejected_in_v1(tmp_path):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    with pytest.raises(ValueError, match="driver-scoped"):
        store.apply(
            profile.profile_id,
            command(
                "session-setting",
                "reply_language",
                "en",
                scope=PreferenceScope.SESSION,
            ),
        )


def test_multiple_profiles_keep_preferences_separate_and_one_default(tmp_path):
    store = repository(tmp_path)
    first = store.ensure_default_profile()
    second = store.create_profile("Endurance", profile_id="endurance")
    assert not second.is_default
    store.apply(second.profile_id, command("pit-on", "announce_pit_transitions", True))
    selected = store.set_default_profile(second.profile_id)
    assert selected.is_default
    profiles = store.list_profiles()
    assert [profile.profile_id for profile in profiles] == ["endurance", "default"]
    assert sum(profile.is_default for profile in profiles) == 1
    assert not store.preferences(first.profile_id).announce_pit_transitions
    assert store.preferences(second.profile_id).announce_pit_transitions


def test_rename_is_validated_and_reset_is_audited_as_explicit_defaults(tmp_path):
    ticks = iter(NOW + timedelta(seconds=value) for value in range(10))
    store = SqliteDriverProfileRepository(
        tmp_path / "driver.sqlite3", clock=lambda: next(ticks)
    )
    profile = store.ensure_default_profile()
    renamed = store.rename_profile(profile.profile_id, "Furkan")
    assert renamed.display_name == "Furkan"
    store.apply(
        profile.profile_id,
        command(
            "position-off",
            "announce_position_changes",
            False,
            timestamp=NOW + timedelta(seconds=3),
        ),
    )
    reset = store.reset_preferences(profile.profile_id, source=PreferenceSource.UI)
    assert reset.announce_position_changes and not reset.announce_pit_transitions
    assert reset.reply_language == "auto"
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM preference_commands").fetchone()[0] == 4


def test_policy_decisions_and_radio_outcomes_are_idempotent_and_linked(tmp_path):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    approved = decision("approved")
    store.record_policy_decision(profile.profile_id, approved)
    store.record_policy_decision(profile.profile_id, approved)
    result = PlaybackResult(
        intent_id="intent-approved",
        status=PlaybackStatus.COMPLETED,
        started_at=NOW + timedelta(seconds=1),
        finished_at=NOW + timedelta(seconds=2),
    )
    store.record_playback_result(profile.profile_id, result)
    store.record_playback_result(profile.profile_id, result)

    with pytest.raises(DriverMemoryError, match="decision_conflict"):
        store.record_policy_decision(
            profile.profile_id,
            approved.model_copy(update={"priority": 71}),
        )
    with pytest.raises(DriverMemoryError, match="outcome_conflict"):
        store.record_playback_result(
            profile.profile_id,
            result.model_copy(update={"status": PlaybackStatus.FAILED}),
        )
    with pytest.raises(DriverMemoryError, match="decision_not_found"):
        store.record_playback_result(
            profile.profile_id,
            PlaybackResult(intent_id="unknown", status=PlaybackStatus.FAILED),
        )

    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT count(*) FROM policy_decisions").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM radio_outcomes").fetchone() == (1,)


def test_history_summaries_and_retention_are_profile_scoped(tmp_path):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    older = decision("older", session_id="old-session", decided_at=NOW)
    recent = decision(
        "recent",
        session_id="new-session",
        decided_at=NOW + timedelta(days=10),
    )
    suppressed = decision(
        "suppressed",
        session_id="new-session",
        decided_at=NOW + timedelta(days=10, seconds=1),
        approved=False,
    )
    for item in (older, recent, suppressed):
        store.record_policy_decision(profile.profile_id, item)
    store.record_playback_result(
        profile.profile_id,
        PlaybackResult(
            intent_id="intent-older",
            status=PlaybackStatus.CANCELLED,
            error_code="muted",
        ),
    )
    store.record_playback_result(
        profile.profile_id,
        PlaybackResult(intent_id="intent-recent", status=PlaybackStatus.COMPLETED),
    )

    summaries = store.recent_session_history(profile.profile_id)
    assert [summary.session_id for summary in summaries] == ["new-session", "old-session"]
    assert summaries[0].decisions == 2
    assert summaries[0].approved == 1 and summaries[0].suppressed == 1
    assert summaries[0].radio_completed == 1 and summaries[0].radio_outcomes == 1
    assert summaries[1].radio_cancelled == 1

    assert store.prune_history(
        profile.profile_id, before=NOW + timedelta(days=1)
    ) == 1
    assert [item.session_id for item in store.recent_session_history(profile.profile_id)] == [
        "new-session"
    ]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM radio_outcomes WHERE intent_id = 'intent-older'"
        ).fetchone() == (0,)


def test_history_schema_cannot_store_content_payloads(tmp_path):
    store = repository(tmp_path)
    store.ensure_default_profile()
    with sqlite3.connect(store.path) as connection:
        columns = {
            row[1]
            for table in ("policy_decisions", "radio_outcomes")
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
    assert not {"facts", "text", "audio", "transcript", "prompt", "utterance"} & columns


def test_runtime_history_write_failures_never_escape(tmp_path, monkeypatch):
    store = repository(tmp_path)
    profile = store.ensure_default_profile()
    history = DurableHistory(store, profile.profile_id)

    def fail(*args, **kwargs):
        raise DriverMemoryError("driver_memory_write_failed")

    monkeypatch.setattr(store, "record_policy_decision", fail)
    monkeypatch.setattr(store, "record_playback_result", fail)
    history.record_decision(decision("isolated"))
    history.record_playback_result(
        PlaybackResult(intent_id="intent-isolated", status=PlaybackStatus.FAILED)
    )


def test_future_schema_is_rejected_without_modifying_it(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE schema_migrations("
            "version INTEGER PRIMARY KEY, name TEXT, applied_at TEXT)"
        )
        connection.execute(
            "INSERT INTO schema_migrations VALUES (99, 'future', '2026-09-21T12:00:00+00:00')"
        )
    with pytest.raises(DriverMemoryError, match="schema_too_new"):
        SqliteDriverProfileRepository(path, clock=lambda: NOW)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT version FROM schema_migrations").fetchall() == [(99,)]


def test_unrelated_sqlite_database_is_never_adopted(tmp_path):
    path = tmp_path / "other.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE precious_data(value TEXT)")
        connection.execute("INSERT INTO precious_data VALUES ('keep me')")
    with pytest.raises(DriverMemoryError, match="wrong_database"):
        SqliteDriverProfileRepository(path, clock=lambda: NOW)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT value FROM precious_data").fetchone() == ("keep me",)
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name = 'driver_profiles'"
        ).fetchone() == (0,)


def test_headless_profile_actions_use_the_configured_database(tmp_path, monkeypatch):
    path = tmp_path / "profiles.sqlite3"
    monkeypatch.setenv("RACE_ENGINEER_DATABASE_PATH", str(path))
    shown = _profile_action(ROOT / "config/default.toml", "show")
    assert shown["profile"]["profile_id"] == "default"
    renamed = _profile_action(
        ROOT / "config/default.toml", "rename", display_name="Furkan"
    )
    assert renamed["profile"]["display_name"] == "Furkan"
    updated = _profile_action(
        ROOT / "config/default.toml",
        "set",
        setting="reply_language",
        value="tr",
    )
    assert updated["preferences"]["reply_language"] == "tr"
    reset = _profile_action(ROOT / "config/default.toml", "reset")
    assert reset["preferences"]["reply_language"] == "auto"
    assert path.exists()


def test_headless_history_action_summarizes_configured_database(tmp_path, monkeypatch):
    path = tmp_path / "history.sqlite3"
    monkeypatch.setenv("RACE_ENGINEER_DATABASE_PATH", str(path))
    store = SqliteDriverProfileRepository(path, clock=lambda: NOW)
    profile = store.ensure_default_profile()
    store.record_policy_decision(profile.profile_id, decision("cli"))
    result = _history_action(ROOT / "config/default.toml", "recent", limit=1)
    assert result["retention_days"] == 30
    assert result["sessions"][0]["session_id"] == "session-1"
    assert result["sessions"][0]["decisions"] == 1


def test_headless_preference_parser_is_strict():
    assert _parse_profile_value("announce_position_changes", "TRUE") is True
    assert _parse_profile_value("reply_language", "tr") == "tr"
    with pytest.raises(ValueError):
        _parse_profile_value("announce_position_changes", "yes")
    with pytest.raises(ValueError):
        _parse_profile_value("reply_language", "Turkish")

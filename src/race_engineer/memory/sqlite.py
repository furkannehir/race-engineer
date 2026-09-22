"""Versioned SQLite storage for local driver profiles and explicit preferences."""

import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from race_engineer.core.contracts import PlaybackResult, PolicyDecision, PreferenceCommand
from race_engineer.core.enums import PreferenceScope, PreferenceSource
from race_engineer.memory.models import (
    CommunicationPreferences,
    DriverProfile,
    SessionHistorySummary,
    default_preferences,
    validate_preference_value,
)

_APPLICATION_ID = 0x52414345  # "RACE"
_SETTINGS = (
    "announce_position_changes",
    "announce_pit_transitions",
    "reply_language",
)


class DriverMemoryError(RuntimeError):
    """Stable failure code; database internals are kept out of user-facing messages."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


_MIGRATIONS = (
    Migration(
        1,
        "driver profiles and explicit communication preferences",
        (
            """
            CREATE TABLE driver_profiles (
                profile_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_default INTEGER NOT NULL CHECK (is_default IN (0, 1))
            ) STRICT
            """,
            """
            CREATE UNIQUE INDEX one_default_driver_profile
            ON driver_profiles(is_default) WHERE is_default = 1
            """,
            """
            CREATE TABLE preference_commands (
                command_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES driver_profiles(profile_id) ON DELETE CASCADE,
                setting TEXT NOT NULL CHECK (setting IN (
                    'announce_position_changes', 'announce_pit_transitions', 'reply_language'
                )),
                value_json TEXT NOT NULL,
                source TEXT NOT NULL CHECK (source IN ('config', 'cli', 'ui', 'voice')),
                scope TEXT NOT NULL CHECK (scope = 'driver'),
                timestamp TEXT NOT NULL
            ) STRICT
            """,
            """
            CREATE TABLE communication_preferences (
                profile_id TEXT NOT NULL REFERENCES driver_profiles(profile_id) ON DELETE CASCADE,
                setting TEXT NOT NULL CHECK (setting IN (
                    'announce_position_changes', 'announce_pit_transitions', 'reply_language'
                )),
                value_json TEXT NOT NULL,
                source TEXT NOT NULL CHECK (source IN ('config', 'cli', 'ui', 'voice')),
                updated_at TEXT NOT NULL,
                command_id TEXT NOT NULL REFERENCES preference_commands(command_id),
                PRIMARY KEY (profile_id, setting)
            ) STRICT
            """,
        ),
    ),
    Migration(
        2,
        "privacy-safe policy decisions and radio outcomes",
        (
            """
            CREATE TABLE policy_decisions (
                profile_id TEXT NOT NULL
                    REFERENCES driver_profiles(profile_id) ON DELETE CASCADE,
                decision_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                source_sequence INTEGER NOT NULL CHECK (source_sequence >= 0),
                decided_at TEXT NOT NULL,
                outcome TEXT NOT NULL CHECK (outcome IN ('approved', 'suppressed')),
                reason TEXT NOT NULL CHECK (reason IN (
                    'approved', 'disabled', 'expired', 'duplicate', 'cooldown', 'superseded'
                )),
                priority INTEGER NOT NULL CHECK (priority BETWEEN 0 AND 100),
                intent_id TEXT,
                PRIMARY KEY (profile_id, decision_id),
                UNIQUE (profile_id, intent_id),
                UNIQUE (profile_id, decision_id, intent_id),
                CHECK (
                    (outcome = 'approved' AND reason = 'approved' AND intent_id IS NOT NULL)
                    OR
                    (outcome = 'suppressed' AND reason != 'approved' AND intent_id IS NULL)
                )
            ) STRICT
            """,
            """
            CREATE INDEX policy_decisions_by_profile_session_time
            ON policy_decisions(profile_id, session_id, decided_at)
            """,
            """
            CREATE TABLE radio_outcomes (
                profile_id TEXT NOT NULL,
                decision_id TEXT NOT NULL,
                intent_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('completed', 'cancelled', 'expired', 'failed')
                ),
                error_code TEXT,
                started_at TEXT,
                finished_at TEXT,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY (profile_id, intent_id),
                FOREIGN KEY (profile_id, decision_id, intent_id)
                    REFERENCES policy_decisions(profile_id, decision_id, intent_id)
                    ON DELETE CASCADE
            ) STRICT
            """,
            """
            CREATE INDEX radio_outcomes_by_profile_recorded_time
            ON radio_outcomes(profile_id, recorded_at)
            """,
        ),
    ),
)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _utc_value(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


class SqliteDriverProfileRepository:
    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
        busy_timeout_ms: int = 5000,
    ) -> None:
        if not 0 <= busy_timeout_ms <= 60_000:
            raise ValueError("SQLite busy timeout must be between 0 and 60000 ms")
        self.path = path
        self._clock = clock or (lambda: datetime.now(UTC))
        self._busy_timeout_ms = busy_timeout_ms
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._migrate()
        except (OSError, sqlite3.Error) as error:
            raise DriverMemoryError("driver_memory_open_failed") from error

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            yield connection
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self._connection() as connection:
            application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
            if application_id not in {0, _APPLICATION_ID}:
                raise DriverMemoryError("driver_memory_wrong_database")
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
                if not str(row[0]).startswith("sqlite_")
            }
            if application_id == 0 and tables - {"schema_migrations"}:
                raise DriverMemoryError("driver_memory_wrong_database")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        applied_at TEXT NOT NULL
                    ) STRICT
                    """
                )
                applied = {
                    int(row["version"])
                    for row in connection.execute(
                        "SELECT version FROM schema_migrations ORDER BY version"
                    )
                }
                known = {migration.version for migration in _MIGRATIONS}
                if not applied.issubset(known):
                    raise DriverMemoryError("driver_memory_schema_too_new")
                if application_id == 0:
                    connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                for migration in _MIGRATIONS:
                    if migration.version in applied:
                        continue
                    for statement in migration.statements:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                        (migration.version, migration.name, _utc_text(self._clock())),
                    )
                connection.execute(f"PRAGMA user_version = {_MIGRATIONS[-1].version}")
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            connection.execute("PRAGMA journal_mode = WAL")

    def ensure_default_profile(self) -> DriverProfile:
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM driver_profiles WHERE is_default = 1"
                ).fetchone()
                if row is None:
                    now = _utc_text(self._clock())
                    connection.execute(
                        """
                        INSERT INTO driver_profiles(
                            profile_id, display_name, created_at, updated_at, is_default
                        ) VALUES (?, ?, ?, ?, 1)
                        """,
                        ("default", "Default driver", now, now),
                    )
                    row = connection.execute(
                        "SELECT * FROM driver_profiles WHERE profile_id = 'default'"
                    ).fetchone()
                connection.commit()
                assert row is not None
                return self._profile(row)
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_write_failed") from error

    def create_profile(
        self,
        display_name: str,
        *,
        profile_id: str | None = None,
        make_default: bool = False,
    ) -> DriverProfile:
        now = self._clock()
        candidate = DriverProfile(
            profile_id=profile_id or uuid.uuid4().hex,
            display_name=display_name,
            created_at=now,
            updated_at=now,
            is_default=make_default,
        )
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                count = int(
                    connection.execute("SELECT count(*) FROM driver_profiles").fetchone()[0]
                )
                is_default = make_default or count == 0
                if is_default:
                    connection.execute("UPDATE driver_profiles SET is_default = 0")
                connection.execute(
                    """
                    INSERT INTO driver_profiles(
                        profile_id, display_name, created_at, updated_at, is_default
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.profile_id,
                        candidate.display_name,
                        _utc_text(candidate.created_at),
                        _utc_text(candidate.updated_at),
                        int(is_default),
                    ),
                )
                connection.commit()
            result = self.get_profile(candidate.profile_id)
            assert result is not None
            return result
        except sqlite3.IntegrityError as error:
            raise DriverMemoryError("driver_profile_already_exists") from error
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_write_failed") from error

    def get_profile(self, profile_id: str) -> DriverProfile | None:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT * FROM driver_profiles WHERE profile_id = ?", (profile_id,)
                ).fetchone()
            return None if row is None else self._profile(row)
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_read_failed") from error

    def default_profile(self) -> DriverProfile:
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT * FROM driver_profiles WHERE is_default = 1"
                ).fetchone()
            if row is None:
                return self.ensure_default_profile()
            return self._profile(row)
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_read_failed") from error

    def list_profiles(self) -> tuple[DriverProfile, ...]:
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT * FROM driver_profiles ORDER BY is_default DESC, created_at, profile_id"
                ).fetchall()
            return tuple(self._profile(row) for row in rows)
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_read_failed") from error

    def rename_profile(self, profile_id: str, display_name: str) -> DriverProfile:
        current = self.get_profile(profile_id)
        if current is None:
            raise DriverMemoryError("driver_profile_not_found")
        candidate = current.model_copy(
            update={"display_name": display_name, "updated_at": self._clock()}
        )
        candidate = DriverProfile.model_validate(candidate.model_dump())
        try:
            with self._connection() as connection:
                result = connection.execute(
                    "UPDATE driver_profiles SET display_name = ?, updated_at = ? "
                    "WHERE profile_id = ?",
                    (candidate.display_name, _utc_text(candidate.updated_at), profile_id),
                )
            if result.rowcount != 1:
                raise DriverMemoryError("driver_profile_not_found")
            return candidate
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_write_failed") from error

    def set_default_profile(self, profile_id: str) -> DriverProfile:
        profile = self.get_profile(profile_id)
        if profile is None:
            raise DriverMemoryError("driver_profile_not_found")
        now = _utc_text(self._clock())
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("UPDATE driver_profiles SET is_default = 0")
                connection.execute(
                    "UPDATE driver_profiles SET is_default = 1, updated_at = ? "
                    "WHERE profile_id = ?",
                    (now, profile_id),
                )
                connection.commit()
            result = self.get_profile(profile_id)
            assert result is not None
            return result
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_write_failed") from error

    def preferences(self, profile_id: str) -> CommunicationPreferences:
        profile = self.get_profile(profile_id)
        if profile is None:
            raise DriverMemoryError("driver_profile_not_found")
        values: dict[str, Any] = default_preferences(profile_id).model_dump()
        latest: datetime | None = None
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT setting, value_json, updated_at
                    FROM communication_preferences WHERE profile_id = ?
                    """,
                    (profile_id,),
                ).fetchall()
            for row in rows:
                setting = str(row["setting"])
                value = validate_preference_value(setting, json.loads(row["value_json"]))
                values[setting] = value
                timestamp = _utc_value(str(row["updated_at"]))
                latest = timestamp if latest is None or timestamp > latest else latest
            values["updated_at"] = latest
            return CommunicationPreferences.model_validate(values)
        except (json.JSONDecodeError, sqlite3.Error, ValueError) as error:
            raise DriverMemoryError("driver_memory_read_failed") from error

    def has_explicit_preferences(self, profile_id: str) -> bool:
        if self.get_profile(profile_id) is None:
            raise DriverMemoryError("driver_profile_not_found")
        try:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT 1 FROM communication_preferences WHERE profile_id = ? LIMIT 1",
                    (profile_id,),
                ).fetchone()
            return row is not None
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_read_failed") from error

    def apply(self, profile_id: str, command: PreferenceCommand) -> CommunicationPreferences:
        return self.apply_many(profile_id, (command,))

    def apply_many(
        self,
        profile_id: str,
        commands: tuple[PreferenceCommand, ...],
    ) -> CommunicationPreferences:
        profile = self.get_profile(profile_id)
        if profile is None:
            raise DriverMemoryError("driver_profile_not_found")
        prepared: list[tuple[PreferenceCommand, str, str]] = []
        for command in commands:
            if command.scope is not PreferenceScope.DRIVER:
                raise ValueError("only driver-scoped preferences are persistent in schema v1")
            value = validate_preference_value(command.setting, command.value)
            encoded = json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
            prepared.append((command, encoded, _utc_text(command.timestamp)))
        if not prepared:
            return self.preferences(profile_id)
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                profile_updated_at = profile.updated_at
                for command, encoded, timestamp in prepared:
                    existing = connection.execute(
                        "SELECT profile_id, setting, value_json, source, scope, timestamp "
                        "FROM preference_commands WHERE command_id = ?",
                        (command.command_id,),
                    ).fetchone()
                    expected = (
                        profile_id,
                        command.setting,
                        encoded,
                        command.source.value,
                        command.scope.value,
                        timestamp,
                    )
                    if existing is not None:
                        if tuple(existing) != expected:
                            raise DriverMemoryError("preference_command_conflict")
                        continue
                    current = connection.execute(
                        "SELECT updated_at FROM communication_preferences "
                        "WHERE profile_id = ? AND setting = ?",
                        (profile_id, command.setting),
                    ).fetchone()
                    if current is not None and command.timestamp < _utc_value(
                        current["updated_at"]
                    ):
                        raise DriverMemoryError("preference_command_stale")
                    connection.execute(
                        """
                        INSERT INTO preference_commands(
                            command_id, profile_id, setting, value_json,
                            source, scope, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (command.command_id, *expected),
                    )
                    connection.execute(
                        """
                        INSERT INTO communication_preferences(
                            profile_id, setting, value_json, source, updated_at, command_id
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(profile_id, setting) DO UPDATE SET
                            value_json = excluded.value_json,
                            source = excluded.source,
                            updated_at = excluded.updated_at,
                            command_id = excluded.command_id
                        """,
                        (
                            profile_id,
                            command.setting,
                            encoded,
                            command.source.value,
                            timestamp,
                            command.command_id,
                        ),
                    )
                    profile_updated_at = max(profile_updated_at, command.timestamp)
                connection.execute(
                    "UPDATE driver_profiles SET updated_at = ? WHERE profile_id = ?",
                    (_utc_text(profile_updated_at), profile_id),
                )
                connection.commit()
            return self.preferences(profile_id)
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_write_failed") from error

    def reset_preferences(
        self,
        profile_id: str,
        *,
        source: PreferenceSource,
    ) -> CommunicationPreferences:
        defaults = default_preferences(profile_id)
        commands = tuple(
            PreferenceCommand(
                command_id=uuid.uuid4().hex,
                setting=setting,
                value=getattr(defaults, setting),
                source=source,
                timestamp=self._clock(),
                scope=PreferenceScope.DRIVER,
            )
            for setting in _SETTINGS
        )
        return self.apply_many(profile_id, commands)

    def record_policy_decision(
        self, profile_id: str, decision: PolicyDecision
    ) -> None:
        if self.get_profile(profile_id) is None:
            raise DriverMemoryError("driver_profile_not_found")
        expected = (
            profile_id,
            decision.decision_id,
            decision.candidate_id,
            decision.session_id,
            decision.source_sequence,
            _utc_text(decision.decided_at),
            decision.outcome.value,
            decision.reason.value,
            decision.priority,
            decision.intent_id,
        )
        try:
            with self._connection() as connection:
                existing = connection.execute(
                    """
                    SELECT profile_id, decision_id, candidate_id, session_id,
                           source_sequence, decided_at, outcome, reason, priority, intent_id
                    FROM policy_decisions
                    WHERE profile_id = ? AND decision_id = ?
                    """,
                    (profile_id, decision.decision_id),
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != expected:
                        raise DriverMemoryError("history_decision_conflict")
                    return
                connection.execute(
                    """
                    INSERT INTO policy_decisions(
                        profile_id, decision_id, candidate_id, session_id,
                        source_sequence, decided_at, outcome, reason, priority, intent_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    expected,
                )
        except sqlite3.IntegrityError as error:
            raise DriverMemoryError("history_decision_conflict") from error
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_write_failed") from error

    def record_playback_result(
        self, profile_id: str, result: PlaybackResult
    ) -> None:
        if self.get_profile(profile_id) is None:
            raise DriverMemoryError("driver_profile_not_found")
        try:
            with self._connection() as connection:
                decision = connection.execute(
                    """
                    SELECT decision_id FROM policy_decisions
                    WHERE profile_id = ? AND intent_id = ?
                    """,
                    (profile_id, result.intent_id),
                ).fetchone()
                if decision is None:
                    raise DriverMemoryError("history_decision_not_found")
                expected = (
                    profile_id,
                    str(decision["decision_id"]),
                    result.intent_id,
                    result.status.value,
                    result.error_code,
                    _utc_text(result.started_at) if result.started_at is not None else None,
                    _utc_text(result.finished_at) if result.finished_at is not None else None,
                )
                existing = connection.execute(
                    """
                    SELECT profile_id, decision_id, intent_id, status, error_code,
                           started_at, finished_at
                    FROM radio_outcomes
                    WHERE profile_id = ? AND intent_id = ?
                    """,
                    (profile_id, result.intent_id),
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != expected:
                        raise DriverMemoryError("history_outcome_conflict")
                    return
                connection.execute(
                    """
                    INSERT INTO radio_outcomes(
                        profile_id, decision_id, intent_id, status, error_code,
                        started_at, finished_at, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (*expected, _utc_text(self._clock())),
                )
        except sqlite3.IntegrityError as error:
            raise DriverMemoryError("history_outcome_conflict") from error
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_write_failed") from error

    def recent_session_history(
        self, profile_id: str, *, limit: int = 10
    ) -> tuple[SessionHistorySummary, ...]:
        if not 1 <= limit <= 100:
            raise ValueError("history limit must be between 1 and 100")
        if self.get_profile(profile_id) is None:
            raise DriverMemoryError("driver_profile_not_found")
        try:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        d.session_id,
                        min(d.decided_at) AS first_decided_at,
                        max(d.decided_at) AS last_decided_at,
                        count(*) AS decisions,
                        sum(CASE WHEN d.outcome = 'approved' THEN 1 ELSE 0 END) AS approved,
                        sum(CASE WHEN d.outcome = 'suppressed' THEN 1 ELSE 0 END) AS suppressed,
                        count(r.intent_id) AS radio_outcomes,
                        sum(CASE WHEN r.status = 'completed' THEN 1 ELSE 0 END)
                            AS radio_completed,
                        sum(CASE WHEN r.status = 'cancelled' THEN 1 ELSE 0 END)
                            AS radio_cancelled,
                        sum(CASE WHEN r.status = 'expired' THEN 1 ELSE 0 END)
                            AS radio_expired,
                        sum(CASE WHEN r.status = 'failed' THEN 1 ELSE 0 END)
                            AS radio_failed
                    FROM policy_decisions AS d
                    LEFT JOIN radio_outcomes AS r
                      ON r.profile_id = d.profile_id AND r.decision_id = d.decision_id
                    WHERE d.profile_id = ?
                    GROUP BY d.session_id
                    ORDER BY last_decided_at DESC, d.session_id
                    LIMIT ?
                    """,
                    (profile_id, limit),
                ).fetchall()
            return tuple(
                SessionHistorySummary(
                    profile_id=profile_id,
                    session_id=str(row["session_id"]),
                    first_decided_at=_utc_value(str(row["first_decided_at"])),
                    last_decided_at=_utc_value(str(row["last_decided_at"])),
                    decisions=int(row["decisions"]),
                    approved=int(row["approved"]),
                    suppressed=int(row["suppressed"]),
                    radio_outcomes=int(row["radio_outcomes"]),
                    radio_completed=int(row["radio_completed"]),
                    radio_cancelled=int(row["radio_cancelled"]),
                    radio_expired=int(row["radio_expired"]),
                    radio_failed=int(row["radio_failed"]),
                )
                for row in rows
            )
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_read_failed") from error

    def prune_history(self, profile_id: str, *, before: datetime) -> int:
        if before.tzinfo is None or before.utcoffset() is None:
            raise ValueError("history cutoff must be timezone-aware")
        if self.get_profile(profile_id) is None:
            raise DriverMemoryError("driver_profile_not_found")
        try:
            with self._connection() as connection:
                result = connection.execute(
                    "DELETE FROM policy_decisions WHERE profile_id = ? AND decided_at < ?",
                    (profile_id, _utc_text(before)),
                )
            return result.rowcount
        except sqlite3.Error as error:
            raise DriverMemoryError("driver_memory_write_failed") from error

    @staticmethod
    def _profile(row: sqlite3.Row) -> DriverProfile:
        return DriverProfile(
            profile_id=str(row["profile_id"]),
            display_name=str(row["display_name"]),
            created_at=_utc_value(str(row["created_at"])),
            updated_at=_utc_value(str(row["updated_at"])),
            is_default=bool(row["is_default"]),
        )

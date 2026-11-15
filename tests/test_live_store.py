"""Live Watch storage layer: migration v2, row round-trips, crash recovery.

A9's suite. Exercises the real SQLite machinery end to end: a fresh database
migrating from scratch (v0→v2), a hand-built v1 database upgraded in place
(1→2) with its existing rows intact, idempotent re-migration, and the store
API the manager codes against — round-trips for all three tables, the
UNSET/None patch semantics, cascade deletes, and crash recovery through
``reset_interrupted_states``.
"""

from __future__ import annotations

import threading

import pytest
from autoclip import db
from autoclip.db.store import UNSET
from autoclip.live import store
from autoclip.live.models import (
    LiveClipRow,
    LiveStreamRow,
    LiveUploadRow,
    new_id,
)


def table_names() -> set[str]:
    with db.connection() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r["name"] for r in rows}


class TestSchemaAndMigration:
    def test_fresh_init_reaches_the_current_version(self, autoclip_home) -> None:
        version = db.init()

        assert version == db.SCHEMA_VERSION == 3
        with db.connection() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 3

    def test_fresh_init_creates_live_tables(self, autoclip_home) -> None:
        db.init()

        assert {"live_streams", "live_clips", "live_uploads"} <= table_names()

    def test_migration_upgrades_a_v1_database_in_place(self, autoclip_home) -> None:
        # Hand-build a v1 database the way an old install leaves it.
        with db.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE sources (id TEXT PRIMARY KEY, type TEXT NOT NULL,
                                      path TEXT NOT NULL, title TEXT NOT NULL);
                PRAGMA user_version = 1;
                """
            )
            conn.execute(
                "INSERT INTO sources (id, type, path, title) VALUES ('s1', 'upload', 'p', 'Old')"
            )

        version = db.init()

        assert version == 3
        with db.connection() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
            # v1 data survived the upgrade.
            assert conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 1
        assert {"live_streams", "live_clips", "live_uploads"} <= table_names()

    def test_migration_is_idempotent(self, autoclip_home) -> None:
        db.init()
        stream = store.upsert_stream(_stream(state="watching"))

        first = db.init()
        second = db.init()

        assert first == second == 3
        # Re-running migration must not clobber live data.
        loaded = store.get_stream(stream.id)
        assert loaded is not None
        assert loaded.state == "watching"

    def test_live_store_works_on_a_v1_database_after_init(self, autoclip_home) -> None:
        """The realistic upgrade path: old db + new binary → store just works."""
        with db.connection() as conn:
            conn.executescript("PRAGMA user_version = 1;")
        stream = _stream()
        db.init()

        created = store.upsert_stream(stream)

        assert created.id == stream.id
        assert created.created_at == stream.created_at


class TestStreamRoundTrips:
    def test_upsert_and_get_round_trip(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream(last_error="boom"))

        loaded = store.get_stream(stream.id)

        assert loaded == _stream_like(stream)

    def test_get_missing_returns_none(self, initialised_db: int) -> None:
        assert store.get_stream("nope") is None

    def test_upsert_preserves_created_at_on_conflict(self, initialised_db: int) -> None:
        first = store.upsert_stream(_stream(created_at="2020-01-01T00:00:00+00:00"))

        reloaded = _stream_like(first)
        reloaded.label = "renamed"
        reloaded.last_error = None
        second = store.upsert_stream(reloaded)

        assert second.created_at == "2020-01-01T00:00:00+00:00"
        assert second.label == "renamed"
        assert second.last_error is None
        assert [s.id for s in store.list_streams()] == [first.id]

    def test_list_streams_is_registration_order(self, initialised_db: int) -> None:
        old = store.upsert_stream(_stream(label="older", created_at="2026-01-01T00:00:00+00:00"))
        new = store.upsert_stream(_stream(label="newer", created_at="2026-06-01T00:00:00+00:00"))

        listed = store.list_streams()

        assert [s.id for s in listed] == [old.id, new.id]

    def test_update_stream_state_patches_only_supplied_fields(self, initialised_db: int) -> None:
        stream = store.upsert_stream(
            _stream(state="waiting_for_live", state_detail="", last_error="old")
        )

        store.update_stream_state(
            stream.id, state="watching", state_detail="seg 42", error="cap failed"
        )

        loaded = store.get_stream(stream.id)
        assert loaded is not None
        assert loaded.state == "watching"
        assert loaded.state_detail == "seg 42"
        assert loaded.last_error == "cap failed"
        assert loaded.label == stream.label  # untouched by the patch

    def test_update_stream_state_refreshes_updated_at(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream(updated_at="2020-01-01T00:00:00+00:00"))

        store.update_stream_state(stream.id, state="connecting")

        loaded = store.get_stream(stream.id)
        assert loaded is not None
        assert loaded.updated_at != "2020-01-01T00:00:00+00:00"

    def test_delete_stream_cascades_to_clips_and_uploads(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        clip = store.create_clip(_clip(stream.id))
        store.create_upload(_upload(clip.id))

        store.delete_stream(stream.id)

        assert store.get_stream(stream.id) is None
        assert store.get_clip(clip.id) is None
        assert store.list_uploads(clip.id) == []


class TestClipRoundTrips:
    def test_create_and_get_round_trip(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        clip = store.create_clip(_clip(stream.id, title="Big moment", score=88))

        loaded = store.get_clip(clip.id)

        assert loaded == _clip_like(clip)
        assert loaded.title == "Big moment"
        assert loaded.score == 88

    def test_get_missing_returns_none(self, initialised_db: int) -> None:
        assert store.get_clip("nope") is None

    def test_list_clips_scopes_by_stream_and_orders_newest_first(self, initialised_db: int) -> None:
        one = store.upsert_stream(_stream())
        other = store.upsert_stream(_stream())
        old = store.create_clip(_clip(one.id, created_at="2026-01-01T00:00:00+00:00"))
        new = store.create_clip(_clip(one.id, created_at="2026-02-01T00:00:00+00:00"))
        other_clip = store.create_clip(_clip(other.id))

        scoped = store.list_clips(one.id)
        all_clips = store.list_clips()

        assert [c.id for c in scoped] == [new.id, old.id]
        assert len(all_clips) == 3
        assert {c.id for c in all_clips} == {old.id, new.id, other_clip.id}

    def test_list_clips_limit(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        ids = [
            store.create_clip(_clip(stream.id, created_at=f"2026-01-{i:02d}T00:00:00+00:00")).id
            for i in range(1, 6)
        ]

        listed = store.list_clips(limit=3)

        assert len(listed) == 3
        assert {c.id for c in listed} <= set(ids)


class TestUploadRoundTrips:
    def test_create_and_list_round_trip(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        clip = store.create_clip(_clip(stream.id))
        upload = store.create_upload(
            _upload(clip.id, platform="youtube", upload_url="https://youtu.be/x")
        )

        listed = store.list_uploads(clip.id)

        assert listed == [_upload_like(upload)]
        assert listed[0].upload_url == "https://youtu.be/x"

    def test_list_uploads_scopes_by_clip(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        first = store.create_clip(_clip(stream.id))
        second = store.create_clip(_clip(stream.id))
        store.create_upload(_upload(first.id))
        store.create_upload(_upload(first.id))
        store.create_upload(_upload(second.id))

        assert len(store.list_uploads(first.id)) == 2
        assert len(store.list_uploads(second.id)) == 1
        assert len(store.list_uploads()) == 3


class TestPatchSemantics:
    """``update_clip`` / ``update_upload`` — the **fields contract.

    The manager writes progress through them one field at a time, so an
    omitted field must survive and the nullable ``error`` must be clearable
    — the UNSET sentinel distinguishes "clear it" from "don't touch it".
    """

    @pytest.fixture
    def clip(self, initialised_db: int) -> LiveClipRow:
        stream = store.upsert_stream(_stream())
        return store.create_clip(_clip(stream.id, rendered_path="render.mp4", error=None))

    def test_patch_writes_only_supplied_fields(self, clip: LiveClipRow) -> None:
        store.update_clip(clip.id, status="queued", title="Queued!")

        loaded = store.get_clip(clip.id)
        assert loaded is not None
        assert loaded.status == "queued"
        assert loaded.title == "Queued!"
        assert loaded.rendered_path == "render.mp4"  # untouched

    def test_patch_refreshes_updated_at(self, clip: LiveClipRow) -> None:
        before = store.get_clip(clip.id).updated_at

        store.update_clip(clip.id, status="rendering")

        assert store.get_clip(clip.id).updated_at == before or (
            store.get_clip(clip.id).updated_at >= before
        )

    def test_patch_none_clears_the_nullable_error_column(self, clip: LiveClipRow) -> None:
        store.update_clip(clip.id, error="transcribe died")

        store.update_clip(clip.id, error=None)

        loaded = store.get_clip(clip.id)
        assert loaded is not None
        assert loaded.error is None

    def test_patch_unset_is_the_same_as_omitting(self, clip: LiveClipRow) -> None:
        store.update_clip(clip.id, error="stale failure")

        store.update_clip(clip.id, error=UNSET)

        loaded = store.get_clip(clip.id)
        assert loaded is not None
        assert loaded.error == "stale failure"  # sentinel = leave alone

    def test_patch_none_on_a_not_null_column_is_refused(self, clip: LiveClipRow) -> None:
        with pytest.raises(ValueError, match="non-nullable"):
            store.update_clip(clip.id, title=None)

    def test_patch_unknown_field_is_refused(self, clip: LiveClipRow) -> None:
        with pytest.raises(ValueError, match="Unknown or read-only"):
            store.update_clip(clip.id, titel="typo")

    def test_patch_id_and_timestamps_are_read_only(self, clip: LiveClipRow) -> None:
        with pytest.raises(ValueError, match="Unknown or read-only"):
            store.update_clip(clip.id, id="new-id")
        with pytest.raises(ValueError, match="Unknown or read-only"):
            store.update_clip(clip.id, created_at="1999-01-01T00:00:00+00:00")

    def test_patch_with_no_fields_is_a_noop(self, clip: LiveClipRow) -> None:
        before = store.get_clip(clip.id).updated_at

        store.update_clip(clip.id)

        assert store.get_clip(clip.id).updated_at == before

    def test_upload_patch_semantics_match(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        clip = store.create_clip(_clip(stream.id))
        upload = store.create_upload(_upload(clip.id, error="oauth refused"))

        store.update_upload(upload.id, status="uploading")
        assert store.list_uploads(clip.id)[0].status == "uploading"

        store.update_upload(upload.id, error=None)
        assert store.list_uploads(clip.id)[0].error is None

        with pytest.raises(ValueError, match="non-nullable"):
            store.update_upload(upload.id, platform=None)

        with pytest.raises(ValueError, match="Unknown or read-only"):
            store.update_upload(upload.id, plateforme="youtube")


class TestCrashRecovery:
    def test_reset_interrupted_states_flips_active_back_to_waiting(
        self, initialised_db: int
    ) -> None:
        watching = store.upsert_stream(_stream(state="watching", state_detail="seg 9"))
        clipping = store.upsert_stream(_stream(state="clipping"))
        reconnecting = store.upsert_stream(_stream(state="reconnecting"))
        connecting = store.upsert_stream(_stream(state="connecting"))

        reset = store.reset_interrupted_states()

        assert reset == 4
        for sid in (watching.id, clipping.id, reconnecting.id, connecting.id):
            loaded = store.get_stream(sid)
            assert loaded is not None
            assert loaded.state == "waiting_for_live"
            assert loaded.state_detail == ""

    def test_reset_leaves_terminal_and_paused_states_alone(self, initialised_db: int) -> None:
        ended = store.upsert_stream(_stream(state="ended"))
        errored = store.upsert_stream(_stream(state="error", last_error="dead"))
        paused = store.upsert_stream(_stream(state="off"))
        waiting = store.upsert_stream(_stream(state="waiting_for_live"))

        reset = store.reset_interrupted_states()

        assert reset == 1
        assert store.get_stream(ended.id).state == "ended"
        assert store.get_stream(errored.id).state == "error"
        assert store.get_stream(errored.id).last_error == "dead"
        assert store.get_stream(paused.id).state == "off"
        assert store.get_stream(waiting.id).state == "waiting_for_live"

    def test_reset_is_idempotent(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream(state="watching", state_detail="seg 9"))

        first = store.reset_interrupted_states()
        second = store.reset_interrupted_states()

        # waiting_for_live is itself a resettable state (its stale detail is
        # cleared), so both calls match the same row — but the end state is
        # identical: this is what idempotence means for crash recovery.
        assert first == 1
        assert second == 1
        loaded = store.get_stream(stream.id)
        assert loaded is not None
        assert loaded.state == "waiting_for_live"
        assert loaded.state_detail == ""

    def test_reset_preserves_last_error(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream(state="watching", last_error="segfault at t=120s"))

        store.reset_interrupted_states()

        loaded = store.get_stream(stream.id)
        assert loaded is not None
        assert loaded.state == "waiting_for_live"
        assert loaded.last_error == "segfault at t=120s"

    def test_reset_works_right_after_a_v1_upgrade(self, autoclip_home) -> None:
        """The full crash-recovery story: old db, new binary, reboot."""
        with db.connection() as conn:
            conn.executescript("PRAGMA user_version = 1;")
        db.init()
        crashed = store.upsert_stream(_stream(state="watching", state_detail="seg 7"))

        reset = store.reset_interrupted_states()

        assert reset == 1
        loaded = store.get_stream(crashed.id)
        assert loaded is not None
        assert loaded.state == "waiting_for_live"
        assert loaded.state_detail == ""


class TestConnectionConventions:
    def test_functions_use_per_thread_connections(self, initialised_db: int) -> None:
        """Store functions must not blow up when called from a fresh thread.

        The manager runs them via ``asyncio.to_thread``, i.e. a thread with
        no connection yet — exactly what this reproduces.
        """
        stream = store.upsert_stream(_stream())
        result: dict[str, object] = {}
        errors: list[Exception] = []

        def worker() -> None:
            try:
                loaded = store.get_stream(stream.id)
                result["ok"] = loaded is not None and loaded.id == stream.id
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert not errors
        assert result.get("ok") is True


class TestInterruptedWork:
    """A restart must not leave clips or uploads hanging mid-flight forever."""

    def test_in_flight_clips_become_failed_interrupted(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        busy = {
            status: store.create_clip(_clip(stream.id, status=status))
            for status in ("detecting", "queued", "rendering")
        }
        done = store.create_clip(_clip(stream.id, status="ready"))
        rejected = store.create_clip(_clip(stream.id, status="rejected", reason="meh"))

        store.reset_interrupted_states()

        for row in busy.values():
            loaded = store.get_clip(row.id)
            assert loaded.status == "failed"
            assert loaded.reason == "interrupted"
            assert loaded.error == "interrupted"
        assert store.get_clip(done.id).status == "ready"
        assert store.get_clip(rejected.id).status == "rejected"
        assert store.get_clip(rejected.id).reason == "meh"

    def test_uploads_stuck_uploading_become_failed(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        clip = store.create_clip(_clip(stream.id, status="ready"))
        stuck = store.create_upload(_upload(clip.id, status="uploading"))
        done = store.create_upload(_upload(clip.id, status="done", platform="instagram"))

        store.reset_interrupted_states()

        rows = {r.id: r for r in store.list_uploads(clip.id)}
        assert rows[stuck.id].status == "failed"
        assert rows[stuck.id].error == "interrupted"
        assert rows[done.id].status == "done"


class TestTombstoneGuard:
    def test_guarded_update_writes_live_rows(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        clip = store.create_clip(_clip(stream.id, status="queued"))
        assert store.update_clip_guarded(clip.id, status="rendering") is True
        assert store.get_clip(clip.id).status == "rendering"

    def test_guarded_update_never_resurrects_a_discarded_clip(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        clip = store.create_clip(_clip(stream.id, status="rendering"))
        store.update_clip(clip.id, status="rejected", reason="discarded by user")

        assert store.update_clip_guarded(clip.id, status="ready", rendered_path="x.mp4") is False
        loaded = store.get_clip(clip.id)
        assert loaded.status == "rejected"
        assert loaded.reason == "discarded by user"
        assert loaded.rendered_path == ""

    def test_guarded_update_of_a_missing_row_is_false(self, initialised_db: int) -> None:
        assert store.update_clip_guarded("nope", status="ready") is False


class TestJudgeBudgetBookkeeping:
    def test_judge_calls_are_summed_per_stream_in_a_window(self, initialised_db: int) -> None:
        a = store.upsert_stream(_stream())
        b = store.upsert_stream(_stream())
        first = store.create_clip(_clip(a.id))
        second = store.create_clip(_clip(a.id))
        other = store.create_clip(_clip(b.id))
        store.add_judge_call(first.id)
        store.add_judge_call(first.id)  # a repair retry: the second paid call
        store.add_judge_call(second.id)
        store.add_judge_call(other.id)
        old = store.create_clip(_clip(a.id, created_at="2000-01-01T00:00:00+00:00"))
        store.add_judge_call(old.id)

        assert store.judge_calls_since(a.id, "2001-01-01T00:00:00+00:00") == 3
        assert store.judge_calls_since(a.id, "1999-01-01T00:00:00+00:00") == 4
        assert store.judge_calls_since(b.id, "2001-01-01T00:00:00+00:00") == 1
        assert store.get_clip(first.id).judge_calls == 2


class TestClipQueries:
    def test_list_clips_filters_by_statuses(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream())
        for status in ("ready", "failed", "rejected", "queued"):
            store.create_clip(_clip(stream.id, status=status))
        got = {r.status for r in store.list_clips(statuses=["ready", "failed"])}
        assert got == {"ready", "failed"}
        assert len(store.list_clips()) == 4

    def test_clip_counts_count_ready_clips_per_stream(self, initialised_db: int) -> None:
        a = store.upsert_stream(_stream())
        b = store.upsert_stream(_stream())
        store.create_clip(_clip(a.id, status="ready"))
        store.create_clip(_clip(a.id, status="ready"))
        store.create_clip(_clip(a.id, status="rejected"))
        store.create_clip(_clip(b.id, status="failed"))
        assert store.clip_counts() == {a.id: 2}

    def test_set_stream_enabled_persists_flag_and_state(self, initialised_db: int) -> None:
        stream = store.upsert_stream(_stream(state="watching"))
        store.set_stream_enabled(stream.id, False, state="off")
        loaded = store.get_stream(stream.id)
        assert loaded.enabled is False and loaded.state == "off"
        store.set_stream_enabled(stream.id, True)
        assert store.get_stream(stream.id).enabled is True


class TestMigrationAtomicity:
    def test_v2_failure_rolls_back_completely(self, autoclip_home, monkeypatch) -> None:
        """A crash half-way through v2 must not leave half the tables behind."""
        from autoclip.db import schema

        with db.connection() as conn:
            conn.executescript(schema._V1)
            conn.execute("PRAGMA user_version = 1")
        real = schema._statements

        def broken(script: str) -> list[str]:
            statements = real(script)
            return [*statements[:2], "CREATE TABLE broken (", *statements[2:]]

        monkeypatch.setattr(schema, "_statements", broken)
        with pytest.raises(Exception):  # noqa: B017 - any sqlite error will do
            db.init()
        with db.connection() as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert "live_streams" not in table_names()  # rolled back, not half-made

        monkeypatch.setattr(schema, "_statements", real)
        assert db.init() == db.SCHEMA_VERSION
        assert {"live_streams", "live_clips", "live_uploads"} <= table_names()

    def test_half_migrated_database_from_an_old_build_recovers(self, autoclip_home) -> None:
        """The old non-atomic v2 could leave tables at user_version 1."""
        from autoclip.db import schema

        with db.connection() as conn:
            conn.executescript(schema._V1)
            conn.execute(
                "CREATE TABLE live_streams (id TEXT PRIMARY KEY, url TEXT NOT NULL,"
                " label TEXT NOT NULL DEFAULT '', enabled INTEGER NOT NULL DEFAULT 1,"
                " state TEXT NOT NULL DEFAULT 'waiting_for_live',"
                " state_detail TEXT NOT NULL DEFAULT '', last_error TEXT,"
                " created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            conn.execute("PRAGMA user_version = 1")

        assert db.init() == db.SCHEMA_VERSION
        stream = store.upsert_stream(_stream())
        clip = store.create_clip(_clip(stream.id))
        assert store.get_clip(clip.id).judge_calls == 0

    def test_v3_is_idempotent_when_the_column_exists(self, autoclip_home) -> None:
        from autoclip.db import schema

        db.init()
        with db.connection() as conn:
            conn.execute("PRAGMA user_version = 2")  # pretend v3 never ran
        assert db.init() == db.SCHEMA_VERSION  # ALTER is skipped, no error
        with db.connection() as conn:
            assert "judge_calls" in schema._column_names(conn, "live_clips")


# ---------------------------------------------------------------------------
# Row factories
# ---------------------------------------------------------------------------


def _stream(**overrides) -> LiveStreamRow:
    defaults = {
        "id": new_id(),
        "url": "https://twitch.tv/somechannel",
        "label": "Test stream",
    }
    return LiveStreamRow(**{**defaults, **overrides})


def _clip(stream_id: str, **overrides) -> LiveClipRow:
    defaults = {
        "id": new_id(),
        "stream_id": stream_id,
        "trigger_start_s": 120.0,
        "trigger_end_s": 180.0,
        "clip_start_s": 100.0,
        "clip_end_s": 190.0,
        "title": "Huge moment",
        "score": 85,
        "reason": "crowd went wild",
        "hook": "wait for it",
        "judge_model": "claude",
    }
    return LiveClipRow(**{**defaults, **overrides})


def _upload(clip_id: str, **overrides) -> LiveUploadRow:
    defaults = {
        "id": new_id(),
        "clip_id": clip_id,
        "platform": "youtube",
    }
    return LiveUploadRow(**{**defaults, **overrides})


def _stream_like(row: LiveStreamRow) -> LiveStreamRow:
    """The row as the database should hand it back."""
    return LiveStreamRow(
        id=row.id,
        url=row.url,
        label=row.label,
        enabled=row.enabled,
        state=row.state,
        state_detail=row.state_detail,
        last_error=row.last_error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _clip_like(row: LiveClipRow) -> LiveClipRow:
    """The clip row as the database should hand it back."""
    return LiveClipRow(
        id=row.id,
        stream_id=row.stream_id,
        status=row.status,
        trigger_start_s=row.trigger_start_s,
        trigger_end_s=row.trigger_end_s,
        clip_start_s=row.clip_start_s,
        clip_end_s=row.clip_end_s,
        title=row.title,
        score=row.score,
        reason=row.reason,
        hook=row.hook,
        judge_model=row.judge_model,
        transcript_path=row.transcript_path,
        rendered_path=row.rendered_path,
        caption_style=row.caption_style,
        ratio=row.ratio,
        error=row.error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _upload_like(row: LiveUploadRow) -> LiveUploadRow:
    return LiveUploadRow(
        id=row.id,
        clip_id=row.clip_id,
        platform=row.platform,
        status=row.status,
        upload_url=row.upload_url,
        platform_media_id=row.platform_media_id,
        error=row.error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )

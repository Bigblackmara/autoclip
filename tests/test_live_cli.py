"""The `autoclip live` command group (C1).

The CLI is deliberately a read-and-configure surface: it must touch nothing
but the persistent records (registry + store + config.json's live section),
so a forgotten terminal session can never be the thing keeping a 24/7 watch
alive. These tests exercise that contract end to end through Typer's
CliRunner against the real registry and store — both file/DB backed, so no
manager, watcher, or subprocess is ever in play.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from autoclip import db, paths
from autoclip.cli import app
from autoclip.live import registry
from autoclip.live import store as live_store
from autoclip.live.models import LiveClipRow, LiveStreamRow
from autoclip.live.settings import LiveStreamConfig
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add(url: str, label: str = "") -> str:
    """Run `live add` and return the stream id it printed."""
    args = ["live", "add", url]
    if label:
        args += ["--label", label]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    for line in result.output.splitlines():
        if line.startswith("  id:"):
            return line.split("  id:", 1)[1].strip()
    raise AssertionError(f"live add printed no id:\n{result.output}")


def _write_live_section(payload: dict) -> None:
    """Persist a config.json whose live section is ``payload``."""
    paths.ensure_layout()
    paths.config_path().write_text(json.dumps({"live": payload}), encoding="utf-8")


# ---------------------------------------------------------------------------
# status — the default view
# ---------------------------------------------------------------------------


class TestStatus:
    def test_bare_live_and_live_status_print_the_streams_table(self, initialised_db: int) -> None:
        _add("https://twitch.tv/channel_one", label="One")
        _add("https://twitch.tv/channel_two", label="Two")

        for argv in (["live"], ["live", "status"]):
            result = runner.invoke(app, argv)
            assert result.exit_code == 0, result.output
            assert "Live Watch streams" in result.output
            # The 80-column test terminal truncates the URL column, so the
            # stable assertions are the label column and the id column.
            assert "One" in result.output
            assert "Two" in result.output
            assert "2 registered" in result.output

    def test_status_lists_a_registry_entry_with_no_db_row_yet(self, initialised_db: int) -> None:
        """A registry entry the store has no row for still lists — its state
        is derived (enabled → waiting_for_live) rather than hidden, matching
        what serve will do for it on the next start."""
        cfg = registry.upsert(LiveStreamConfig(id="", url="https://twitch.tv/reg_only"))

        result = runner.invoke(app, ["live", "status"])

        assert result.exit_code == 0, result.output
        assert cfg.id in result.output
        assert "waiting_for_live" in result.output
        assert "1 registered" in result.output

    def test_status_shows_the_state_the_server_last_recorded(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/live_now")
        live_store.update_stream_state(sid, state="watching", state_detail="seg 42")

        result = runner.invoke(app, ["live", "status"])

        assert result.exit_code == 0, result.output
        assert "watching" in result.output
        assert "1 stream(s) actively watched" in result.output

    def test_status_marks_a_paused_stream_off(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/paused_one")

        assert runner.invoke(app, ["live", "pause", sid]).exit_code == 0

        listed = runner.invoke(app, ["live", "status"])
        assert listed.exit_code == 0, listed.output
        assert "off" in listed.output
        assert "0 stream(s) actively watched" in listed.output

    def test_status_prints_worker_caps_from_live_settings(self, initialised_db: int) -> None:
        result = runner.invoke(app, ["live", "status"])

        assert result.exit_code == 0, result.output
        assert "caps:" in result.output
        # The frozen LiveSettings defaults.
        assert "5 concurrent streams" in result.output
        assert "1 render worker(s)" in result.output
        assert "clip queue 10" in result.output

    def test_status_reports_the_feature_switch(self, initialised_db: int) -> None:
        _write_live_section({"enabled": False})

        result = runner.invoke(app, ["live", "status"])

        assert result.exit_code == 0, result.output
        assert "feature off" in result.output

    def test_status_on_a_fresh_home_is_not_an_error(self, autoclip_home) -> None:
        """No db.init() by the fixture here — the command must do it itself."""
        result = runner.invoke(app, ["live"])

        assert result.exit_code == 0, result.output
        assert "0 registered" in result.output


# ---------------------------------------------------------------------------
# The CLI-never-starts-a-manager contract
# ---------------------------------------------------------------------------


class TestNoManager:
    def test_live_source_never_imports_the_manager_or_capture_modules(self) -> None:
        """`autoclip live` must be pure records: no LiveManager, no watcher,
        no capture subprocess. Checked at the source level because the
        failure mode is an import that quietly starts the whole feature —
        a runtime spy cannot catch code paths that never run in-process
        during tests."""
        source = Path(db.__file__).parent.parent.joinpath("cli.py").read_text(encoding="utf-8")

        for forbidden in ("live.manager", "LiveManager", "live.watcher", "live.capture"):
            assert forbidden not in source, (
                f"'{forbidden}' must not appear in cli.py — the CLI never starts a manager"
            )

    def test_status_does_not_start_the_process_wide_manager(self, initialised_db: int) -> None:
        """The module-level `manager` singleton must stay unstarted after a
        status run — empty watcher registries prove nothing was launched,
        even though merely importing the module constructs the instance."""
        from autoclip.live import manager as manager_mod

        result = runner.invoke(app, ["live", "status"])

        assert result.exit_code == 0, result.output
        assert not manager_mod.manager._watchers
        assert not manager_mod.manager._tasks
        assert manager_mod.manager.started is False


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_registers_in_the_registry(self, initialised_db: int) -> None:
        _add("https://twitch.tv/new_channel", label="Brand New")

        listed = registry.list_streams()
        assert len(listed) == 1
        assert listed[0].url == "https://twitch.tv/new_channel"
        assert listed[0].label == "Brand New"
        assert listed[0].enabled is True

    def test_add_persists_a_store_row_a_serve_will_pick_up(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/db_row", label="Row")

        row = live_store.get_stream(sid)
        assert row is not None
        assert row.url == "https://twitch.tv/db_row"
        assert row.label == "Row"
        assert row.enabled is True

    def test_add_prints_the_id_and_url(self, initialised_db: int) -> None:
        result = runner.invoke(app, ["live", "add", "https://twitch.tv/printed"])

        assert result.exit_code == 0, result.output
        assert "Added." in result.output
        assert "https://twitch.tv/printed" in result.output
        assert "  id:" in result.output

    def test_add_without_a_label_sets_an_empty_one(self, initialised_db: int) -> None:
        cfg_id = _add("https://twitch.tv/no_label")

        cfg = registry.get(cfg_id)
        assert cfg is not None
        assert cfg.label == ""

    def test_add_refuses_a_whitespace_only_url(self, initialised_db: int) -> None:
        result = runner.invoke(app, ["live", "add", "   "])

        assert result.exit_code == 2
        assert "must not be empty" in result.output
        assert registry.list_streams() == []

    def test_add_mints_unique_ids(self, initialised_db: int) -> None:
        first = _add("https://twitch.tv/same_url")
        second = _add("https://twitch.tv/same_url")

        assert first != second

    def test_add_on_a_cold_home_inits_the_db_itself(self, autoclip_home) -> None:
        """No fixture ran db.init() — the command's own prologue must."""
        result = runner.invoke(app, ["live", "add", "https://twitch.tv/cold_start"])

        assert result.exit_code == 0, result.output
        listed = registry.list_streams()
        assert len(listed) == 1
        assert listed[0].url == "https://twitch.tv/cold_start"

    def test_status_merges_registry_entries_and_store_only_rows(self, initialised_db: int) -> None:
        """A store row with no registry entry (added while serve was down,
        or restored by crash recovery) still shows in status alongside the
        registry entries."""
        _add("https://twitch.tv/from_cli", label="From CLI")
        live_store.upsert_stream(LiveStreamRow(id="dbonly1", url="https://twitch.tv/from_db"))

        result = runner.invoke(app, ["live", "status"])

        assert result.exit_code == 0, result.output
        assert "From CLI" in result.output
        assert "dbonly1" in result.output  # long URL truncates; the id column does not
        assert "2 registered" in result.output


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_deletes_the_registry_entry_and_store_row(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/gone")

        result = runner.invoke(app, ["live", "remove", sid])

        assert result.exit_code == 0, result.output
        assert "Removed." in result.output
        assert registry.get(sid) is None
        assert live_store.get_stream(sid) is None

    def test_remove_cascades_its_clips(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/clipped")
        clip = live_store.create_clip(LiveClipRow(id="c1", stream_id=sid))

        runner.invoke(app, ["live", "remove", sid])

        assert live_store.get_clip(clip.id) is None  # ON DELETE CASCADE

    def test_remove_an_unknown_id_exits_1(self, initialised_db: int) -> None:
        result = runner.invoke(app, ["live", "remove", "does-not-exist"])

        assert result.exit_code == 1
        assert "does-not-exist" in result.output

    def test_remove_a_stream_known_only_in_the_store_still_works(self, initialised_db: int) -> None:
        row = live_store.upsert_stream(LiveStreamRow(id="storeonly1", url="https://twitch.tv/x"))

        result = runner.invoke(app, ["live", "remove", row.id])

        assert result.exit_code == 0, result.output
        assert live_store.get_stream(row.id) is None


# ---------------------------------------------------------------------------
# pause / resume
# ---------------------------------------------------------------------------


class TestPauseResume:
    def test_pause_disables_the_registry_entry(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/pauser")

        result = runner.invoke(app, ["live", "pause", sid])

        assert result.exit_code == 0, result.output
        assert "Paused." in result.output
        cfg = registry.get(sid)
        assert cfg is not None
        assert cfg.enabled is False

    def test_pause_records_state_off_in_the_store(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/offs")

        runner.invoke(app, ["live", "pause", sid])

        row = live_store.get_stream(sid)
        assert row is not None
        assert row.state == "off"

    def test_resume_reenables_and_returns_to_waiting_for_live(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/yo-yo")
        runner.invoke(app, ["live", "pause", sid])

        result = runner.invoke(app, ["live", "resume", sid])

        assert result.exit_code == 0, result.output
        assert "Resumed." in result.output
        cfg = registry.get(sid)
        assert cfg is not None
        assert cfg.enabled is True
        row = live_store.get_stream(sid)
        assert row is not None
        assert row.state == "waiting_for_live"

    def test_pause_then_resume_round_trip_preserves_the_row(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/roundtrip", label="Persisted")

        runner.invoke(app, ["live", "pause", sid])
        runner.invoke(app, ["live", "resume", sid])

        row = live_store.get_stream(sid)
        assert row is not None
        assert row.label == "Persisted"
        assert row.enabled is True

    def test_pause_an_unknown_id_exits_1(self, initialised_db: int) -> None:
        result = runner.invoke(app, ["live", "pause", "no-such-stream"])

        assert result.exit_code == 1

    def test_resume_an_unknown_id_exits_1(self, initialised_db: int) -> None:
        result = runner.invoke(app, ["live", "resume", "no-such-stream"])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# clips
# ---------------------------------------------------------------------------


class TestClips:
    def test_clips_lists_rows_newest_first(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/clippy")
        live_store.create_clip(
            LiveClipRow(
                id="old1", stream_id=sid, title="Older", created_at="2026-01-01T00:00:00+00:00"
            )
        )
        live_store.create_clip(
            LiveClipRow(
                id="new1", stream_id=sid, title="Newer", created_at="2026-02-01T00:00:00+00:00"
            )
        )

        result = runner.invoke(app, ["live", "clips"])

        assert result.exit_code == 0, result.output
        assert result.output.index("Newer") < result.output.index("Older")
        assert "old1" in result.output
        assert "new1" in result.output

    def test_clips_respects_the_limit_option(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/limited")
        for i in range(5):
            live_store.create_clip(
                LiveClipRow(
                    id=f"limclip{i}",
                    stream_id=sid,
                    title=f"Clip {i}",
                    created_at=f"2026-01-0{i + 1}T00:00:00+00:00",
                )
            )

        result = runner.invoke(app, ["live", "clips", "--limit", "3"])

        assert result.exit_code == 0, result.output
        # The three newest made the cut; the two oldest did not.
        newest = [c.id for c in live_store.list_clips(limit=3)]
        oldest = [f"limclip{i}" for i in range(5) if f"limclip{i}" not in newest]
        for clip_id in newest:
            assert clip_id in result.output
        for clip_id in oldest:
            assert clip_id not in result.output

    def test_clips_scopes_to_one_stream_with_the_stream_option(self, initialised_db: int) -> None:
        watched = _add("https://twitch.tv/scope_watched")
        other = _add("https://twitch.tv/scope_other")
        mine = live_store.create_clip(LiveClipRow(id="mine1", stream_id=watched, title="Mine"))
        theirs = live_store.create_clip(LiveClipRow(id="theirs1", stream_id=other, title="Theirs"))

        result = runner.invoke(app, ["live", "clips", "--stream", watched])

        assert result.exit_code == 0, result.output
        assert mine.id in result.output
        assert theirs.id not in result.output

    def test_clips_prints_a_friendly_empty_state(self, initialised_db: int) -> None:
        result = runner.invoke(app, ["live", "clips"])

        assert result.exit_code == 0, result.output
        assert "No live clips yet" in result.output

    def test_clips_shows_status_score_length_and_title(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/details")
        live_store.create_clip(
            LiveClipRow(
                id="det1",
                stream_id=sid,
                status="ready",
                score=88,
                title="Big moment",
                clip_start_s=100.0,
                clip_end_s=145.0,
            )
        )

        result = runner.invoke(app, ["live", "clips"])

        assert result.exit_code == 0, result.output
        assert "ready" in result.output
        assert "88" in result.output
        assert "Big moment" in result.output
        assert "45s" in result.output


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_settings_prints_the_persisted_live_section(self, initialised_db: int) -> None:
        _write_live_section({"enabled": False, "workers": {"clip_queue_workers": 3}})

        result = runner.invoke(app, ["live", "settings"])

        assert result.exit_code == 0, result.output
        printed = json.loads(result.stdout)
        assert printed["enabled"] is False
        assert printed["workers"]["clip_queue_workers"] == 3

    def test_settings_without_a_config_file_prints_defaults(self, initialised_db: int) -> None:
        result = runner.invoke(app, ["live", "settings"])

        assert result.exit_code == 0, result.output
        printed = json.loads(result.stdout)
        assert printed["enabled"] is True
        assert printed["workers"]["max_concurrent_streams"] == 5
        assert printed["rolling_buffer"]["segment_s"] == 6.0

    def test_settings_with_an_invalid_section_falls_back_to_defaults(
        self, initialised_db: int
    ) -> None:
        _write_live_section({"enabled": "not-a-bool", "unknown_field": 1})

        result = runner.invoke(app, ["live", "settings"])

        assert result.exit_code == 0, result.output
        assert "invalid" in result.output.lower()
        printed = json.loads(result.stdout[result.stdout.index("{") :])
        assert printed["enabled"] is True  # the frozen default, not the bad value

    def test_settings_prints_only_the_live_section(self, initialised_db: int) -> None:
        """Top-level settings, providers, and anything secret-bearing never
        appear — only the live tree."""
        paths.ensure_layout()
        paths.config_path().write_text(
            json.dumps(
                {
                    "active_provider": "anthropic",
                    "whisper": {"model": "small"},
                    "live": {"enabled": True},
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(app, ["live", "settings"])

        assert result.exit_code == 0, result.output
        printed = json.loads(result.stdout)
        assert set(printed) == {
            "enabled",
            "rolling_buffer",
            "rate_limit",
            "detection",
            "uploads",
            "workers",
        }


# ---------------------------------------------------------------------------
# Cross-command contracts
# ---------------------------------------------------------------------------


class TestContracts:
    def test_every_command_ensures_layout_and_inits_the_db(
        self, autoclip_home, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shared prologue: each command stands alone with no prior
        db.init() (the autoclip_home fixture here runs none), so each command
        itself must run ensure_layout + init."""
        import autoclip.cli as cli_mod
        import autoclip.live.manager  # noqa: F401  (prove importing is safe)

        called = {"ensure": 0, "init": 0}
        real_ensure, real_init = paths.ensure_layout, db.init

        def spy_ensure():
            called["ensure"] += 1
            return real_ensure()

        def spy_init():
            called["init"] += 1
            return real_init()

        monkeypatch.setattr(cli_mod.paths, "ensure_layout", spy_ensure)
        monkeypatch.setattr(db, "init", spy_init)

        sid = _add("https://twitch.tv/spied")
        for argv in (
            ["live", "status"],
            ["live", "clips"],
            ["live", "settings"],
            ["live", "pause", sid],
            ["live", "resume", sid],
            ["live", "remove", sid],
        ):
            result = runner.invoke(app, argv)
            assert result.exit_code == 0, result.output

        # add (the _add above) + the six commands above. Each also matches
        # db.init's own internal ensure_layout, hence the doubled count —
        # the point is every command ran the full prologue exactly once.
        assert called["ensure"] == 14
        assert called["init"] == 7

    def test_existing_commands_are_unchanged_at_the_top_level(self, initialised_db: int) -> None:
        """The append must not disturb the pre-existing command surface."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0, result.output
        for name in (
            "version",
            "doctor",
            "init",
            "clip",
            "jobs",
            "providers",
            "fetch-models",
            "update-ytdlp",
            "serve",
            "styles",
            "config",
        ):
            assert name in result.output, f"'{name}' vanished from the top-level CLI"
        assert "live" in result.output

    def test_a_full_workflow_add_status_pause_resume_remove(self, initialised_db: int) -> None:
        sid = _add("https://twitch.tv/workflow", label="Full Run")

        listed = runner.invoke(app, ["live"])
        assert listed.exit_code == 0, listed.output
        assert sid in listed.output
        assert "Full Run" in listed.output

        runner.invoke(app, ["live", "pause", sid])
        assert "off" in runner.invoke(app, ["live", "status"]).output

        runner.invoke(app, ["live", "resume", sid])
        assert "waiting_for_live" in runner.invoke(app, ["live", "status"]).output

        runner.invoke(app, ["live", "remove", sid])
        assert "0 registered" in runner.invoke(app, ["live", "status"]).output

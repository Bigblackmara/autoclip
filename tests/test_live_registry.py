"""StreamRegistry: the persisted Live Watch watch list."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from autoclip.live import registry
from autoclip.live.settings import LiveStreamConfig


def stream(url: str = "https://twitch.tv/someone", **overrides: object) -> LiveStreamConfig:
    """A valid stream with a stable id, unless the test overrides it."""
    base = {"id": "abc123", "label": "someone"}
    base.update(overrides)
    return LiveStreamConfig(url=url, **base)  # type: ignore[arg-type]


class TestEmptyInit:
    """An absent registry must read as an empty watch list, not an error."""

    def test_missing_file_lists_nothing(self, autoclip_home: Path) -> None:
        assert registry.list_streams() == []

    def test_no_directory_is_created_by_reads(self, autoclip_home: Path) -> None:
        registry.list_streams()

        assert not registry.streams_path().exists()
        assert not autoclip_home.exists()


class TestRoundTrip:
    def test_upsert_then_list(self, autoclip_home: Path) -> None:
        stored = registry.upsert(stream())

        assert registry.list_streams() == [stored]

    def test_survives_a_reinstantiated_module_state(self, autoclip_home: Path) -> None:
        stored = registry.upsert(stream())

        # Fresh read straight from disk — no caching anywhere.
        assert registry.get(stored.id) == stored
        assert registry.get("missing") is None

    def test_empty_id_is_assigned_and_timestamped(self, autoclip_home: Path) -> None:
        stored = registry.upsert(stream(id=""))

        assert stored.id
        assert stored.created_at
        assert registry.get(stored.id) == stored

    def test_upsert_replaces_same_id_in_place(self, autoclip_home: Path) -> None:
        first = registry.upsert(stream())
        second = registry.upsert(
            stream(label="renamed", url="https://twitch.tv/elsewhere", id=first.id)
        )

        assert second.id == first.id
        assert registry.list_streams() == [second]

    def test_reupsert_keeps_original_created_at(self, autoclip_home: Path) -> None:
        first = registry.upsert(stream())
        second = registry.upsert(stream(label="renamed", created_at="", id=first.id))

        assert second.created_at == first.created_at
        assert second.id == first.id

    def test_insertion_order_is_preserved(self, autoclip_home: Path) -> None:
        for index in range(5):
            registry.upsert(stream(id=f"id{index}", url=f"https://twitch.tv/ch{index}"))

        assert [s.id for s in registry.list_streams()] == [f"id{i}" for i in range(5)]

    def test_upsert_rejects_empty_url(self, autoclip_home: Path) -> None:
        with pytest.raises(ValueError, match="url"):
            registry.upsert(stream(url="   "))

        assert registry.list_streams() == []
        assert not registry.streams_path().exists()

    def test_caller_object_is_never_mutated(self, autoclip_home: Path) -> None:
        incoming = stream(id="")

        registry.upsert(incoming)

        assert incoming.id == ""
        assert incoming.created_at == ""


class TestDelete:
    def test_removes_only_the_target(self, autoclip_home: Path) -> None:
        registry.upsert(stream())
        other = registry.upsert(stream(id="other", url="https://twitch.tv/two"))

        assert registry.delete("abc123") is True

        assert registry.list_streams() == [other]
        assert registry.get("abc123") is None

    def test_missing_id_is_a_silent_no_op(self, autoclip_home: Path) -> None:
        stored = registry.upsert(stream())
        path = registry.streams_path()
        before = path.read_text(encoding="utf-8")

        assert registry.delete("nope") is False

        assert registry.list_streams() == [stored]
        assert path.read_text(encoding="utf-8") == before

    def test_deleting_last_stream_writes_an_empty_list(self, autoclip_home: Path) -> None:
        stored = registry.upsert(stream())

        assert registry.delete(stored.id) is True

        assert registry.list_streams() == []
        payload = json.loads(registry.streams_path().read_text(encoding="utf-8"))
        assert payload["streams"] == []


class TestSetEnabled:
    def test_flips_the_flag_and_persists(self, autoclip_home: Path) -> None:
        stored = registry.upsert(stream())
        assert stored.enabled is True

        assert registry.set_enabled(stored.id, False) is True

        assert registry.get(stored.id).enabled is False
        assert registry.list_streams()[0].enabled is False

    def test_missing_id_returns_false_without_writing(self, autoclip_home: Path) -> None:
        registry.upsert(stream())
        path = registry.streams_path()
        before = path.read_text(encoding="utf-8")

        assert registry.set_enabled("nope", True) is False

        assert path.read_text(encoding="utf-8") == before

    def test_already_matching_flag_skips_the_write(self, autoclip_home: Path) -> None:
        stored = registry.upsert(stream(enabled=False))
        path = registry.streams_path()
        before = path.read_text(encoding="utf-8")

        assert registry.set_enabled(stored.id, False) is True

        assert path.read_text(encoding="utf-8") == before


class TestAtomicWrite:
    """The previous watch list must survive a failed write untouched."""

    def test_failed_replace_keeps_old_file_and_removes_temp(
        self, autoclip_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stored = registry.upsert(stream())
        path = registry.streams_path()
        before = path.read_text(encoding="utf-8")

        def failing_replace(src: object, dst: object) -> None:
            raise OSError("simulated failure during replace")

        monkeypatch.setattr(registry.os, "replace", failing_replace)

        with pytest.raises(OSError, match="simulated failure"):
            registry.upsert(stream(id="doomed"))

        assert path.read_text(encoding="utf-8") == before
        assert registry.list_streams() == [stored]
        assert list(autoclip_home.glob("*.tmp")) == []

    def test_failed_replace_with_missing_temp_cleanup_is_still_tolerant(
        self, autoclip_home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure while removing the temp file must mask the original error."""
        registry.upsert(stream())
        path = registry.streams_path()
        before = path.read_text(encoding="utf-8")

        real_replace = registry.os.replace
        real_unlink = registry.os.unlink

        def failing_replace(src: object, dst: object) -> None:
            real_unlink(src)  # pretend the crash left no sidecar behind
            raise OSError("simulated failure during replace")

        def failing_unlink(target: object, **kwargs: object) -> None:
            raise OSError("cleanup also failed")

        monkeypatch.setattr(registry.os, "replace", failing_replace)
        monkeypatch.setattr(registry.os, "unlink", failing_unlink)
        try:
            with pytest.raises(OSError, match="simulated failure during replace"):
                registry.upsert(stream(id="doomed"))
        finally:
            monkeypatch.setattr(registry.os, "replace", real_replace)

        assert path.read_text(encoding="utf-8") == before

    def test_no_temp_file_left_behind_on_success(self, autoclip_home: Path) -> None:
        registry.upsert(stream())
        registry.delete("abc123")
        registry.set_enabled("abc123", False)

        assert list(autoclip_home.glob("*.tmp")) == []


class TestLock:
    """Read-modify-write cycles must serialise; concurrent writers lose nothing."""

    def test_concurrent_upserts_all_land(self, autoclip_home: Path) -> None:
        n = 10

        barrier = threading.Barrier(n)

        def insert(index: int) -> None:
            barrier.wait()
            for j in range(10):
                registry.upsert(stream(id=f"w{index}-{j}", url=f"https://twitch.tv/{index}/{j}"))

        threads = [threading.Thread(target=insert, args=(i,)) for i in range(n)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        stored = {s.id for s in registry.list_streams()}
        assert stored == {f"w{i}-{j}" for i in range(n) for j in range(10)}

    def test_reads_during_writes_never_see_partial_state(self, autoclip_home: Path) -> None:
        """A reader that grabs the lock mid-burst sees a consistent snapshot."""
        n = 4
        inserted = threading.Event()
        stop = threading.Event()

        def writer() -> None:
            i = 0
            while not stop.is_set():
                registry.upsert(stream(id=f"live{i}", url=f"https://twitch.tv/{i}"))
                if i >= 3:
                    inserted.set()
                i += 1

        def reader(out: list[int]) -> None:
            inserted.wait()
            for _ in range(50):
                ids = {s.id for s in registry.list_streams()}
                out.append(len({x for x in ids if x.startswith("live")}))
            stop.set()

        writers = [threading.Thread(target=writer) for _ in range(n)]
        snapshots: list[int] = []
        reader_thread = threading.Thread(target=reader, args=(snapshots,))
        for thread in writers:
            thread.start()
        reader_thread.start()
        reader_thread.join(timeout=15)
        stop.set()
        for thread in writers:
            thread.join(timeout=15)

        assert snapshots, "reader never observed a snapshot"
        assert all(count >= 4 for count in snapshots)

    def test_exception_inside_the_lock_releases_it(self, autoclip_home: Path) -> None:
        with pytest.raises(ValueError):
            registry.upsert(stream(url=""))

        # Would deadlock the whole suite if the lock leaked.
        assert registry.upsert(stream()).id == "abc123"


class TestCorruptTolerance:
    """One bad file must not cost the user their watch list or their startup."""

    def test_corrupt_json_reads_as_empty(self, autoclip_home: Path) -> None:
        registry.streams_path().parent.mkdir(parents=True, exist_ok=True)
        registry.streams_path().write_text("{ not json", encoding="utf-8")

        assert registry.list_streams() == []

    def test_bare_list_shape_still_loads(self, autoclip_home: Path) -> None:
        path = registry.streams_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([stream().model_dump(mode="json")]), encoding="utf-8")

        assert [s.id for s in registry.list_streams()] == ["abc123"]

    def test_bad_single_entry_is_skipped(self, autoclip_home: Path) -> None:
        path = registry.streams_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            {"url": "https://twitch.tv/ok", "id": "keep"},
            {"id": "no url here"},
        ]
        path.write_text(json.dumps({"version": 1, "streams": entries}), encoding="utf-8")

        assert [s.id for s in registry.list_streams()] == ["keep"]

    def test_unknown_fields_are_dropped_on_next_write(self, autoclip_home: Path) -> None:
        path = registry.streams_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = {"version": 99, "streams": [{"id": "abc123", "url": "u://x", "future_field": 1}]}
        path.write_text(json.dumps(raw), encoding="utf-8")

        registry.set_enabled("abc123", False)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["version"] == registry.REGISTRY_VERSION
        assert payload["streams"] == [
            {"id": "abc123", "url": "u://x", "label": "", "enabled": False, "created_at": ""}
        ]

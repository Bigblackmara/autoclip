"""Small JSON state files under the AutoClip home, written atomically.

Posting and Autopilot keep their state here instead of in SQLite (the
database schema is owned elsewhere). Every change is a read-modify-write
under two locks:

* a process-wide ``threading.RLock`` per file (the API's worker threads and
  the background loops share one process), and
* an OS file lock on a ``.lock`` sidecar, so a second AutoClip process (the
  CLI next to the server) can't interleave a write.

Writes go to a temp file in the same directory, are flushed to disk, then
``os.replace``-d over the target: a crash leaves the old file or the new
one, never half of either. A file that can't be parsed is kept aside as
``<name>.corrupt-<n>`` and treated as empty, so one bad write never stops
the app.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .. import paths

log = logging.getLogger(__name__)

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_REPLACE_ATTEMPTS = 20


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.RLock()
        return lock


@contextlib.contextmanager
def _os_lock(path: Path) -> Iterator[None]:
    """An exclusive OS lock on ``<path>.lock`` (blocks until acquired)."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")  # noqa: SIM115 - held for the with-block
    try:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:  # LK_LOCK gives up after ~10 s; keep waiting
                    time.sleep(0.05)
            try:
                yield
            finally:
                fh.seek(0)
                with contextlib.suppress(OSError):
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def write_atomic(path: Path, data: Any) -> None:
    """Write ``data`` as JSON to ``path`` atomically (temp file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1, ensure_ascii=False, sort_keys=False)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                # Windows: a reader holds the target open for a moment.
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(0.05)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class JsonFile:
    """One JSON document under ``<home>/<folder>/<name>``.

    The path is resolved on every access (``AUTOCLIP_HOME`` can change
    between tests). ``default`` builds the empty document.
    """

    def __init__(self, folder: str, name: str, default: Callable[[], Any]) -> None:
        self.folder = folder
        self.name = name
        self._default = default

    @property
    def path(self) -> Path:
        return paths.root() / self.folder / self.name

    def _read_unlocked(self, path: Path) -> Any:
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._default()
        except OSError as exc:
            log.warning("Could not read %s: %s", path, exc)
            return self._default()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._quarantine(path)
            return self._default()
        default = self._default()
        if not isinstance(data, type(default)):
            self._quarantine(path)
            return default
        return data

    def _quarantine(self, path: Path) -> None:
        for n in range(1, 100):
            target = path.with_name(f"{path.name}.corrupt-{n}")
            if not target.exists():
                with contextlib.suppress(OSError):
                    os.replace(path, target)
                log.warning("%s was unreadable; kept it as %s and started empty.", path, target)
                return

    def read(self) -> Any:
        """A private copy of the current document."""
        path = self.path
        with _thread_lock(path):
            return copy.deepcopy(self._read_unlocked(path))

    @contextlib.contextmanager
    def edit(self) -> Iterator[Any]:
        """``with file.edit() as data: ...`` - saved atomically on a clean exit."""
        path = self.path
        with _thread_lock(path), _os_lock(path):
            data = self._read_unlocked(path)
            yield data
            write_atomic(path, data)

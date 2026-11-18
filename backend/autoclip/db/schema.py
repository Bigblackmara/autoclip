"""SQLite schema and migrations.

Migrations are driven by SQLite's ``user_version`` pragma: each entry in
:data:`MIGRATIONS` moves the database forward one version. Applying them is
idempotent, so :func:`migrate` runs safely on every startup.

We use raw ``sqlite3`` rather than an ORM. The schema is six small tables read
by one local process, so an ORM would add a dependency and a layer of
indirection without buying anything.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

_V1 = """
CREATE TABLE sources (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL CHECK (type IN ('youtube', 'upload')),
    url         TEXT,
    filename    TEXT,
    path        TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    channel     TEXT,
    duration_s  REAL NOT NULL DEFAULT 0,
    width       INTEGER,
    height      INTEGER,
    fps         REAL,
    has_audio   INTEGER NOT NULL DEFAULT 1,
    has_video   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);

CREATE TABLE jobs (
    id            TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    status        TEXT NOT NULL CHECK (status IN ('queued','running','failed','done','cancelled')),
    current_stage TEXT NOT NULL DEFAULT '',
    progress      REAL NOT NULL DEFAULT 0,
    error         TEXT,
    provider      TEXT NOT NULL DEFAULT '',
    settings_json TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);
CREATE INDEX idx_jobs_source ON jobs(source_id);
CREATE INDEX idx_jobs_status ON jobs(status);

CREATE TABLE transcripts (
    job_id          TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    json_path       TEXT NOT NULL,
    language        TEXT NOT NULL DEFAULT '',
    model           TEXT NOT NULL DEFAULT '',
    has_diarization INTEGER NOT NULL DEFAULT 0,
    word_count      INTEGER NOT NULL DEFAULT 0,
    source          TEXT NOT NULL DEFAULT 'whisper',
    created_at      TEXT NOT NULL
);

CREATE TABLE clips (
    id           TEXT PRIMARY KEY,
    job_id       TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    rank         INTEGER NOT NULL DEFAULT 0,
    start_s      REAL NOT NULL,
    end_s        REAL NOT NULL,
    start_word   INTEGER NOT NULL DEFAULT 0,
    end_word     INTEGER NOT NULL DEFAULT 0,
    title        TEXT NOT NULL DEFAULT '',
    hook         TEXT NOT NULL DEFAULT '',
    score        INTEGER NOT NULL DEFAULT 0,
    reason       TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'candidate'
                 CHECK (status IN ('candidate','kept','discarded','exported')),
    user_trimmed INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);
CREATE INDEX idx_clips_job ON clips(job_id, rank);

CREATE TABLE clip_edits (
    clip_id           TEXT PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
    edited_words_json TEXT,
    caption_style     TEXT NOT NULL DEFAULT 'bold_pop',
    ratio             TEXT NOT NULL DEFAULT '9:16',
    updated_at        TEXT NOT NULL
);

CREATE TABLE exports (
    id         TEXT PRIMARY KEY,
    clip_id    TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    ratio      TEXT NOT NULL DEFAULT '9:16',
    style      TEXT NOT NULL DEFAULT 'bold_pop',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_exports_clip ON exports(clip_id);
"""


def _migration_v1(conn: sqlite3.Connection) -> None:
    conn.executescript(_V1)


_V2 = """
-- Live Watch (work order A9). Columns match the frozen row dataclasses in
-- autoclip/live/models.py exactly; from_row() reads them by name.
-- Status columns deliberately carry NO CHECK constraints: statuses evolve
-- with the feature and the store layer validates them in Python, so a new
-- status must never require a migration.
-- Every statement is IF NOT EXISTS: a database left half-migrated by an
-- older build (which ran this script outside a transaction) re-runs cleanly.
CREATE TABLE IF NOT EXISTS live_streams (
    id           TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    label        TEXT NOT NULL DEFAULT '',
    enabled      INTEGER NOT NULL DEFAULT 1,
    state        TEXT NOT NULL DEFAULT 'waiting_for_live',
    state_detail TEXT NOT NULL DEFAULT '',
    last_error   TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_live_streams_state ON live_streams(state);

CREATE TABLE IF NOT EXISTS live_clips (
    id              TEXT PRIMARY KEY,
    stream_id       TEXT NOT NULL REFERENCES live_streams(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'detecting',
    trigger_start_s REAL NOT NULL DEFAULT 0,
    trigger_end_s   REAL NOT NULL DEFAULT 0,
    clip_start_s    REAL NOT NULL DEFAULT 0,
    clip_end_s      REAL NOT NULL DEFAULT 0,
    title           TEXT NOT NULL DEFAULT '',
    score           INTEGER NOT NULL DEFAULT 0,
    reason          TEXT NOT NULL DEFAULT '',
    hook            TEXT NOT NULL DEFAULT '',
    judge_model     TEXT NOT NULL DEFAULT '',
    transcript_path TEXT NOT NULL DEFAULT '',
    rendered_path   TEXT NOT NULL DEFAULT '',
    caption_style   TEXT NOT NULL DEFAULT 'bold_pop',
    ratio           TEXT NOT NULL DEFAULT '9:16',
    error           TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_live_clips_stream ON live_clips(stream_id, created_at);

CREATE TABLE IF NOT EXISTS live_uploads (
    id                TEXT PRIMARY KEY,
    clip_id           TEXT NOT NULL REFERENCES live_clips(id) ON DELETE CASCADE,
    platform          TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending',
    upload_url        TEXT NOT NULL DEFAULT '',
    platform_media_id TEXT NOT NULL DEFAULT '',
    error             TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_live_uploads_clip ON live_uploads(clip_id);
"""


def _statements(script: str) -> list[str]:
    """Split a DDL script into statements (``--`` comment lines dropped).

    Safe for this module's scripts only: they hold no string literal or
    trigger body containing a semicolon.
    """
    lines = [line for line in script.splitlines() if not line.lstrip().startswith("--")]
    return [stmt.strip() for stmt in "\n".join(lines).split(";") if stmt.strip()]


def _run_in_transaction(conn: sqlite3.Connection, statements: list[str]) -> None:
    """Execute ``statements`` inside the caller's transaction.

    ``executescript`` is deliberately avoided: it COMMITs first and then runs
    in autocommit mode, so a crash half-way left some tables created and
    ``user_version`` unchanged — and the next start died on "table already
    exists". An explicit BEGIN makes the DDL and the ``user_version`` bump
    that :func:`migrate` issues afterwards one atomic unit (SQLite DDL and
    the user_version header are both transactional).
    """
    if not conn.in_transaction:
        conn.execute("BEGIN")
    for statement in statements:
        conn.execute(statement)


def _migration_v2(conn: sqlite3.Connection) -> None:
    _run_in_transaction(conn, _statements(_V2))


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migration_v3(conn: sqlite3.Connection) -> None:
    """Live Watch: per-clip LLM judge call count (restart-proof judge budgets).

    The per-stream minute/hour/day judge budgets are derived from
    ``live_clips`` timestamps; ``judge_calls`` makes a repair retry count as
    the second paid call it is. Idempotent: the column is only added when
    missing.
    """
    statements: list[str] = []
    if "judge_calls" not in _column_names(conn, "live_clips"):
        statements.append(
            "ALTER TABLE live_clips ADD COLUMN judge_calls INTEGER NOT NULL DEFAULT 0"
        )
    _run_in_transaction(conn, statements)


_V4 = """
-- GTA Mode: queued stream links, the moments judged in them, and the
-- "what's working" reports. Statuses carry no CHECK constraints for the same
-- reason as the live tables: the store validates them in Python.
CREATE TABLE IF NOT EXISTS gta_streams (
    id            TEXT PRIMARY KEY,
    url           TEXT NOT NULL,
    platform      TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    channel       TEXT NOT NULL DEFAULT '',
    duration_s    REAL NOT NULL DEFAULT 0,
    range_start_s REAL,
    range_end_s   REAL,
    status        TEXT NOT NULL DEFAULT 'queued',
    stage         TEXT NOT NULL DEFAULT '',
    progress      REAL NOT NULL DEFAULT 0,
    detail        TEXT NOT NULL DEFAULT '',
    error         TEXT,
    signals       TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gta_streams_status ON gta_streams(status, created_at);

CREATE TABLE IF NOT EXISTS gta_clips (
    id            TEXT PRIMARY KEY,
    stream_id     TEXT NOT NULL REFERENCES gta_streams(id) ON DELETE CASCADE,
    status        TEXT NOT NULL DEFAULT 'candidate',
    start_s       REAL NOT NULL,
    end_s         REAL NOT NULL,
    chunk_index   INTEGER NOT NULL DEFAULT 0,
    hook_title    TEXT NOT NULL DEFAULT '',
    hook_quote    TEXT NOT NULL DEFAULT '',
    moment_type   TEXT NOT NULL DEFAULT '',
    summary       TEXT NOT NULL DEFAULT '',
    hype_score    REAL NOT NULL DEFAULT 0,
    score         INTEGER NOT NULL DEFAULT 0,
    rank          INTEGER,
    verdict       TEXT NOT NULL DEFAULT '{}',
    reject_reason TEXT NOT NULL DEFAULT '',
    caption       TEXT NOT NULL DEFAULT '',
    hashtags      TEXT NOT NULL DEFAULT '[]',
    path          TEXT NOT NULL DEFAULT '',
    qa            TEXT NOT NULL DEFAULT '{}',
    error         TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gta_clips_stream ON gta_clips(stream_id, start_s);
CREATE INDEX IF NOT EXISTS idx_gta_clips_status ON gta_clips(status);

CREATE TABLE IF NOT EXISTS gta_reports (
    id          TEXT PRIMARY KEY,
    channel_url TEXT NOT NULL,
    report      TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
"""


def _migration_v4(conn: sqlite3.Connection) -> None:
    """GTA Mode tables (additive; the other modes never read them)."""
    _run_in_transaction(conn, _statements(_V4))


_V5 = """
-- GTA Mode posting schedule: at most one YouTube upload row per clip. A row
-- holds its publish slot while uploading/scheduled/interrupted; skipped rows
-- are clips the user excluded from auto-posting.
CREATE TABLE IF NOT EXISTS gta_uploads (
    id         TEXT PRIMARY KEY,
    clip_id    TEXT NOT NULL REFERENCES gta_clips(id) ON DELETE CASCADE,
    status     TEXT NOT NULL,
    slot_at    TEXT,
    video_id   TEXT NOT NULL DEFAULT '',
    url        TEXT NOT NULL DEFAULT '',
    title      TEXT NOT NULL DEFAULT '',
    error      TEXT,
    attempts   INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_gta_uploads_clip ON gta_uploads(clip_id);
CREATE INDEX IF NOT EXISTS idx_gta_uploads_slot ON gta_uploads(slot_at);
"""


def _migration_v5(conn: sqlite3.Connection) -> None:
    """GTA Mode YouTube posting queue (additive)."""
    _run_in_transaction(conn, _statements(_V5))


def _migration_v6(conn: sqlite3.Connection) -> None:
    """GTA Mode posting to Instagram (via Zernio) next to YouTube.

    One upload row per clip PER PLATFORM. ``media_url`` = the uploaded file's
    public URL at Zernio, recorded before the post is created so an unclear
    outcome can be looked up. Idempotent: columns are only added when missing.
    """
    columns = _column_names(conn, "gta_uploads")
    statements: list[str] = []
    if "platform" not in columns:
        statements.append(
            "ALTER TABLE gta_uploads ADD COLUMN platform TEXT NOT NULL DEFAULT 'youtube'"
        )
    if "media_url" not in columns:
        statements.append("ALTER TABLE gta_uploads ADD COLUMN media_url TEXT NOT NULL DEFAULT ''")
    statements += [
        "DROP INDEX IF EXISTS idx_gta_uploads_clip",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_gta_uploads_clip_platform "
        "ON gta_uploads(clip_id, platform)",
    ]
    _run_in_transaction(conn, statements)


#: Columns added to ``clips`` by v7, with their DDL.
_V7_CLIP_COLUMNS: tuple[tuple[str, str], ...] = (
    # Up to three on-screen headline options from detection (JSON list).
    ("hook_options", "TEXT NOT NULL DEFAULT '[]'"),
    # The headline actually rendered: one of the options or free text.
    ("hook_text", "TEXT NOT NULL DEFAULT ''"),
    # Post text: description and hashtags (JSON list of "#tag").
    ("description", "TEXT NOT NULL DEFAULT ''"),
    ("hashtags", "TEXT NOT NULL DEFAULT '[]'"),
    # The punchiest line the model flagged for a flash-forward opening, in
    # absolute source seconds. NULL = none flagged (a heuristic picks one).
    ("teaser_start_s", "REAL"),
    ("teaser_end_s", "REAL"),
    # Per-clip flash-forward: NULL follows the export setting, 0/1 override it.
    ("flash_forward", "INTEGER"),
    # Variants: the clip this one re-cuts, and its zoom-pattern seed.
    ("variant_of", "TEXT"),
    ("motion_seed", "INTEGER NOT NULL DEFAULT 0"),
)


def _migration_v7(conn: sqlite3.Connection) -> None:
    """Clip editor: hook options, post text, flash-forward and variants.

    Additive and idempotent: each column is only added when missing. A
    database without a ``clips`` table (hand-built in tests) is left alone.
    """
    columns = _column_names(conn, "clips")
    if not columns:
        _run_in_transaction(conn, [])
        return
    statements = [
        f"ALTER TABLE clips ADD COLUMN {name} {ddl}"
        for name, ddl in _V7_CLIP_COLUMNS
        if name not in columns
    ]
    _run_in_transaction(conn, statements)


#: Columns added to ``gta_clips`` by v8, with their DDL.
_V8_GTA_CLIP_COLUMNS: tuple[tuple[str, str], ...] = (
    # The judge's three on-screen hook title options (JSON list, best first).
    ("hook_options", "TEXT NOT NULL DEFAULT '[]'"),
    # The peak reaction the judge quoted, in stream seconds (flash-forward).
    ("teaser_start_s", "REAL"),
    ("teaser_end_s", "REAL"),
    # Per-clip flash-forward: NULL follows the GTA setting, 0/1 override it.
    ("flash_forward", "INTEGER"),
    # Variants: the original clip this one re-cuts, its zoom-pattern seed and
    # its caption look ('' = the GTA setting).
    ("variant_of", "TEXT"),
    ("motion_seed", "INTEGER NOT NULL DEFAULT 0"),
    ("caption_style", "TEXT NOT NULL DEFAULT ''"),
    # Hook + post text as rendered in the caption language (JSON object).
    ("localised", "TEXT NOT NULL DEFAULT '{}'"),
)


def _migration_v8(conn: sqlite3.Connection) -> None:
    """GTA Mode: hook options, flash-forward, variants and caption language.

    Additive and idempotent: each column is only added when missing. A
    database without a ``gta_clips`` table (hand-built in tests) is left alone.
    """
    columns = _column_names(conn, "gta_clips")
    if not columns:
        _run_in_transaction(conn, [])
        return
    statements = [
        f"ALTER TABLE gta_clips ADD COLUMN {name} {ddl}"
        for name, ddl in _V8_GTA_CLIP_COLUMNS
        if name not in columns
    ]
    _run_in_transaction(conn, statements)


#: Ordered migrations. Index + 1 is the resulting ``user_version``.
#: Append only — never edit a migration that has shipped.
MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    _migration_v1,
    _migration_v2,
    _migration_v3,
    _migration_v4,
    _migration_v5,
    _migration_v6,
    _migration_v7,
    _migration_v8,
]

SCHEMA_VERSION = len(MIGRATIONS)


def migrate(conn: sqlite3.Connection) -> int:
    """Apply any outstanding migrations. Returns the resulting schema version."""
    current: int = conn.execute("PRAGMA user_version").fetchone()[0]

    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {current} is newer than this AutoClip "
            f"build supports ({SCHEMA_VERSION}). Upgrade AutoClip."
        )

    for version in range(current, SCHEMA_VERSION):
        migration = MIGRATIONS[version]
        with conn:
            migration(conn)
            # PRAGMA doesn't accept bound parameters; version is loop-controlled.
            conn.execute(f"PRAGMA user_version = {version + 1}")

    return SCHEMA_VERSION

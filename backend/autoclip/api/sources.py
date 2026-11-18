"""Source ingestion endpoints."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ..config import load as load_settings
from ..db import store
from ..pipeline import drivefolder, ingest
from .schemas import SourceOut, YouTubeIngestIn

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sources", tags=["sources"])

#: Uploads stream to disk in chunks rather than being buffered whole — source
#: videos routinely exceed available RAM.
UPLOAD_CHUNK = 4 * 1024 * 1024


@router.get("", response_model=list[SourceOut])
async def list_sources(limit: int = 50) -> list[SourceOut]:
    sources = await asyncio.to_thread(store.list_sources, limit)
    return [SourceOut.of(s) for s in sources]


@router.get("/{source_id}", response_model=SourceOut)
async def get_source(source_id: str) -> SourceOut:
    source = await asyncio.to_thread(store.get_source, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found.")
    return SourceOut.of(source)


@router.post("/youtube", response_model=SourceOut, status_code=201)
async def ingest_youtube(payload: YouTubeIngestIn) -> SourceOut:
    """Download a YouTube video and register it as a source.

    File links (Drive / Dropbox / OneDrive / direct video URLs) are accepted
    here too and take the file-link path, so every client that could paste a
    YouTube URL can paste those as well.
    """
    if not ingest.is_youtube_url(payload.url):
        if ingest.is_file_link(payload.url):
            return await ingest_link(LinkIngestIn(url=payload.url))
        raise HTTPException(status_code=400, detail="That is not a YouTube URL.")

    settings = load_settings().ingest
    if payload.cookies_from_browser is not None:
        settings = settings.model_copy(
            update={"cookies_from_browser": payload.cookies_from_browser}
        )

    try:
        source = await asyncio.to_thread(ingest.ingest_youtube, payload.url, settings)
    except ingest.IngestError as exc:
        # 422 rather than 500: the request was well-formed, but the content
        # can't be fetched, and the hint tells the user what to change.
        raise HTTPException(
            status_code=422, detail={"message": str(exc), "hint": exc.hint}
        ) from exc

    await asyncio.to_thread(store.create_source, source)
    return SourceOut.of(source)


@router.post("/upload", response_model=SourceOut, status_code=201)
async def upload_source(file: UploadFile = File(...)) -> SourceOut:
    """Accept a media upload and register it as a source."""
    filename = Path(file.filename or "upload")
    suffix = filename.suffix.lower()
    if suffix not in ingest.ACCEPTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail={
                "message": f"{suffix or 'This file'} is not a supported format.",
                "hint": f"Accepted: {', '.join(sorted(ingest.ACCEPTED_SUFFIXES))}",
            },
        )

    # Stage to a temp file first so a failed or abandoned upload never leaves a
    # half-written file in the media store.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as staging:
        staging_path = Path(staging.name)
        try:
            while chunk := await file.read(UPLOAD_CHUNK):
                staging.write(chunk)
        except Exception as exc:
            staging_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Upload failed.") from exc

    try:
        source = await asyncio.to_thread(
            ingest.ingest_file, staging_path, move=True, title=filename.stem
        )
    except ingest.IngestError as exc:
        staging_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422, detail={"message": str(exc), "hint": exc.hint}
        ) from exc
    finally:
        shutil.rmtree(staging_path.parent / staging_path.name, ignore_errors=True)

    await asyncio.to_thread(store.create_source, source)
    return SourceOut.of(source)


# --------------------------------------------------------------------------
# File links (Google Drive, Dropbox, OneDrive, direct URLs)
# --------------------------------------------------------------------------


class LinkIngestIn(BaseModel):
    url: str
    #: Client-chosen id to poll ``GET /api/sources/link/progress/{id}`` with.
    progress_id: str | None = Field(default=None, max_length=64)


class DriveFolderIn(BaseModel):
    url: str


#: Download progress by client-chosen id. In memory: it only has to live as
#: long as the request that fills it, plus a short grace period.
_progress: dict[str, dict[str, Any]] = {}
_progress_lock = threading.Lock()
_PROGRESS_KEEP_S = 600.0
_PROGRESS_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _set_progress(progress_id: str, **values: Any) -> None:
    with _progress_lock:
        now = time.monotonic()
        for key in [k for k, v in _progress.items() if now - v["at"] > _PROGRESS_KEEP_S]:
            del _progress[key]
        entry = _progress.setdefault(progress_id, {})
        entry.update(values, at=now)


def _ingest_error(exc: ingest.IngestError) -> HTTPException:
    # 422 rather than 500: the request was well-formed, but the content
    # can't be fetched, and the hint tells the user what to change.
    return HTTPException(status_code=422, detail={"message": str(exc), "hint": exc.hint})


@router.post("/link", response_model=SourceOut, status_code=201)
async def ingest_link(payload: LinkIngestIn) -> SourceOut:
    """Download a Drive / Dropbox / OneDrive / direct file link as a source."""
    progress_id = payload.progress_id
    if progress_id is not None and not _PROGRESS_ID.fullmatch(progress_id):
        raise HTTPException(status_code=400, detail="Invalid progress id.")
    if drivefolder.folder_id(payload.url):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "That is a Google Drive folder.",
                "hint": "List the folder first and pick the videos to clip.",
            },
        )

    def on_progress(done: int, total: int | None) -> None:
        if progress_id:
            _set_progress(progress_id, done_bytes=done, total_bytes=total, state="downloading")

    if progress_id:
        _set_progress(progress_id, done_bytes=0, total_bytes=None, state="starting")
    settings = load_settings().ingest
    try:
        source = await asyncio.to_thread(
            ingest.ingest_link, payload.url, settings, on_progress=on_progress
        )
    except ingest.IngestError as exc:
        if progress_id:
            _set_progress(progress_id, state="failed")
        raise _ingest_error(exc) from exc
    if progress_id:
        _set_progress(progress_id, state="done")

    await asyncio.to_thread(store.create_source, source)
    return SourceOut.of(source)


@router.get("/link/progress/{progress_id}")
async def link_progress(progress_id: str) -> dict[str, Any]:
    with _progress_lock:
        entry = dict(_progress.get(progress_id) or {})
    if not entry:
        raise HTTPException(status_code=404, detail="No download with that id.")
    entry.pop("at", None)
    return entry


@router.post("/drive-folder")
async def list_drive_folder(payload: DriveFolderIn) -> dict[str, Any]:
    """List the videos in a public Google Drive folder, for the user to pick."""
    folder = drivefolder.folder_id(payload.url)
    if folder is None:
        raise HTTPException(status_code=400, detail="That is not a Google Drive folder link.")
    try:
        listing = await asyncio.to_thread(drivefolder.list_folder, folder)
    except ingest.IngestError as exc:
        raise _ingest_error(exc) from exc
    return {
        "id": listing.id,
        "name": listing.name,
        "truncated": listing.truncated,
        "files": [
            {"id": f.id, "name": f.name, "mime_type": f.mime_type, "folder": f.folder, "url": f.url}
            for f in listing.files
        ],
    }

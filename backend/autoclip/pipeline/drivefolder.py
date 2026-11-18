"""Public Google Drive folders: list the videos inside so the user can pick.

No API key and no sign-in: the folder's public "embedded view" page
(``drive.google.com/embeddedfolderview?id=<id>``) is read and parsed. That
page is not a documented API, so parsing is best effort; when it cannot be
read the user gets a clear error rather than an empty list. Subfolders are
walked breadth-first, with caps so a huge share cannot stall the request.

Picked files are then ingested one by one as ordinary Drive file links
(:mod:`autoclip.pipeline.filelink`).
"""

from __future__ import annotations

import html as html_lib
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, unquote, urlparse

import httpx

from . import filelink
from .ingest import ACCEPTED_SUFFIXES, IngestError

log = logging.getLogger(__name__)

MAX_DEPTH = 3
MAX_FOLDERS = 25
MAX_FILES = 500

FOLDER_MIME = "application/vnd.google-apps.folder"


@dataclass(frozen=True)
class DriveEntry:
    id: str
    name: str
    mime_type: str
    is_folder: bool = False


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str
    mime_type: str
    #: Subfolder path inside the shared folder ("" = top level).
    folder: str = ""

    @property
    def url(self) -> str:
        """The share link the normal Drive-file download understands."""
        return f"https://drive.google.com/file/d/{self.id}/view"


@dataclass
class DriveFolder:
    id: str
    name: str
    files: list[DriveFile] = field(default_factory=list)
    #: True when the walk stopped early (too many files or subfolders).
    truncated: bool = False


def folder_id(raw: str) -> str | None:
    """The folder id in a Drive folder link, or None when it isn't one."""
    try:
        parsed = urlparse(raw.strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host not in ("drive.google.com", "www.drive.google.com"):
        return None
    match = re.search(r"/folders/([\w-]{10,})", parsed.path)
    found = match.group(1) if match else None
    if found is None and parsed.path == "/embeddedfolderview":
        found = dict(parse_qsl(parsed.query)).get("id")
    return found if found and re.fullmatch(r"[\w-]{10,}", found) else None


def is_media(name: str, mime_type: str) -> bool:
    """A file AutoClip can ingest: video or audio, by type or extension."""
    mime = mime_type.lower()
    if mime.startswith(("video/", "audio/")):
        return True
    dot = name.rfind(".")
    return dot >= 0 and name[dot:].lower() in ACCEPTED_SUFFIXES


def parse_embedded_view(page: str) -> tuple[str, list[DriveEntry]]:
    """Parse one embedded-view page into its title and entries.

    Each entry is a ``<div class="flip-entry" id="entry-<ID>">`` holding a link
    (``/folders/`` for subfolders), a type icon whose URL ends in the MIME type
    (``.../type/video/mp4``) and a ``flip-entry-title`` div with the name.
    """
    title_match = re.search(r"<title>([^<]*)</title>", page, re.IGNORECASE)
    title = html_lib.unescape(title_match.group(1)).strip() if title_match else ""
    title = re.sub(r"\s*-\s*Google Drive\s*$", "", title, flags=re.IGNORECASE)

    entries: list[DriveEntry] = []
    chunks = re.split(r"<div\s+class=\"flip-entry\"\s+id=\"entry-", page)[1:]
    for chunk in chunks:
        entry_id = chunk[: chunk.find('"')]
        if not re.fullmatch(r"[\w-]{10,}", entry_id):
            continue
        href_match = re.search(r"<a\s[^>]*href=\"([^\"]+)\"", chunk)
        href = html_lib.unescape(href_match.group(1)) if href_match else ""
        name_match = re.search(r"class=\"flip-entry-title\"[^>]*>([^<]*)<", chunk)
        name = html_lib.unescape(name_match.group(1)).strip() if name_match else ""
        if not name:
            continue
        icon_match = re.search(r"<img\s[^>]*src=\"([^\"]*)\"", chunk)
        icon = html_lib.unescape(icon_match.group(1)) if icon_match else ""
        is_folder = "/folders/" in href
        if is_folder:
            mime = FOLDER_MIME
        else:
            mime = unquote(icon.split("/type/", 1)[1]) if "/type/" in icon else ""
        entries.append(DriveEntry(entry_id, name, mime, is_folder))
    return title, entries


def _looks_private(page: str, final_url: str) -> bool:
    lowered = page.lower()
    return (
        "accounts.google.com" in (urlparse(final_url).hostname or "")
        or "accounts.google.com/servicelogin" in lowered
        or "sign in" in lowered
    )


def _private_error() -> IngestError:
    return IngestError(
        "This Google Drive folder is private.",
        hint='In Drive, set the folder\'s sharing to "Anyone with the link", then try again.',
    )


def read_folder_page(client: httpx.Client, folder: str) -> tuple[str, list[DriveEntry]]:
    url = f"https://drive.google.com/embeddedfolderview?id={folder}"
    try:
        response = filelink.open_url(client, url)
        try:
            status = response.status_code
            final_url = str(response.url)
            if status == 404:
                raise IngestError(
                    "Google Drive says this folder does not exist (or is not shared)."
                )
            if status in (401, 403):
                raise _private_error()
            if status >= 400:
                raise IngestError(f"Could not open the Google Drive folder (HTTP {status}).")
            page = filelink.read_text(response)
        finally:
            response.close()
    except httpx.HTTPError as exc:
        raise IngestError(f"Could not reach Google Drive: {exc}") from exc

    # Even an empty folder renders the (empty) flip-entries list, so a page
    # without it is a sign-in page or a layout this parser does not know.
    if "flip-entries" not in page:
        if _looks_private(page, final_url):
            raise _private_error()
        raise IngestError(
            "Could not read that Google Drive folder.",
            hint="Google may have changed the folder page. Share the video files "
            "one by one instead (Share, then Copy link on each file).",
        )
    return parse_embedded_view(page)


def list_folder(folder: str, *, client: httpx.Client | None = None) -> DriveFolder:
    """List the media files in a public folder and its subfolders."""
    own_client = client is None
    client = client or filelink.new_client()
    try:
        title, root_entries = read_folder_page(client, folder)
        result = DriveFolder(id=folder, name=title or "Google Drive folder")
        queue: list[tuple[str, str, int, list[DriveEntry] | None]] = [(folder, "", 0, root_entries)]
        folders_seen = 0
        while queue:
            current, path, depth, entries = queue.pop(0)
            if entries is None:
                try:
                    _, entries = read_folder_page(client, current)
                except IngestError as exc:
                    log.info("Skipping Drive subfolder %s: %s", path, exc)
                    continue
            for entry in entries:
                if entry.is_folder:
                    folders_seen += 1
                    if depth + 1 > MAX_DEPTH or folders_seen > MAX_FOLDERS:
                        result.truncated = True
                    else:
                        sub = f"{path}/{entry.name}" if path else entry.name
                        queue.append((entry.id, sub, depth + 1, None))
                    continue
                if not is_media(entry.name, entry.mime_type):
                    continue
                if len(result.files) >= MAX_FILES:
                    result.truncated = True
                    break
                result.files.append(DriveFile(entry.id, entry.name, entry.mime_type, path))
        return result
    finally:
        if own_client:
            client.close()

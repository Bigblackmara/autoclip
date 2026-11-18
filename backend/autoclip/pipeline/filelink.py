"""File links as sources: Google Drive, Dropbox, OneDrive and direct video URLs.

These point at a video FILE in cloud storage, not at a video website, so they
need no yt-dlp: one plain HTTP download, after which the file goes through the
same flow as an upload (:func:`autoclip.pipeline.ingest.ingest_file`).

Share links are rewritten to their documented direct-download form:

* Google Drive: ``drive.google.com/uc?export=download&id=<id>``. Files too big
  for Drive's virus scan come back as an HTML "can't scan this file" page; its
  download form (or, on older pages, a ``confirm=`` link or a
  ``download_warning`` cookie) is followed to get the file itself.
* Dropbox: the share link with ``dl=1``.
* OneDrive: personal links (1drv.ms, onedrive.live.com) through the documented
  ``api.onedrive.com/v1.0/shares/u!<base64url>/root/content`` endpoint;
  SharePoint / OneDrive for Business links with ``download=1``.
* Direct links (``.mp4``/``.mov``/``.mkv``/``.webm``/...): as they are.

Downloads stream to disk with a size cap and a content-type check (a host that
answers with a web page instead of the file is caught before anything is
saved). Every request, redirects included, must go to a public http(s) host:
loopback, link-local (cloud metadata) and other special addresses are refused.
"""

from __future__ import annotations

import base64
import html as html_lib
import ipaddress
import logging
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlparse, urlunparse

import httpx

from .ingest import ACCEPTED_SUFFIXES, IngestError

log = logging.getLogger(__name__)

LinkKind = Literal["gdrive", "dropbox", "onedrive", "direct"]

LABELS: dict[str, str] = {
    "gdrive": "Google Drive",
    "dropbox": "Dropbox",
    "onedrive": "OneDrive",
    "direct": "the link",
}

#: Path endings that make an arbitrary URL a "direct video link".
DIRECT_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
MAX_REDIRECTS = 8
CHUNK_BYTES = 1024 * 1024
#: Anything smaller is an error page or an empty file, not a video.
MIN_BYTES = 50_000
#: Most HTML read from a host (Drive's confirm page, a folder listing).
MAX_HTML_BYTES = 5 * 1024 * 1024
PROGRESS_EVERY_S = 0.25

_DRIVE_HOSTS = ("drive.google.com", "docs.google.com", "drive.usercontent.google.com")
_DROPBOX_HOSTS = ("dropbox.com", "dl.dropboxusercontent.com", "dl.dropbox.com")

#: Content types that mean "the host sent a page, not the file".
_PAGE_TYPES = ("text/html", "application/json", "application/xhtml", "text/plain", "text/xml")

_SUFFIX_BY_TYPE = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
    "video/x-m4v": ".m4v",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
}

ProgressFn = Callable[[int, int | None], None]


@dataclass(frozen=True)
class FileLink:
    kind: LinkKind
    url: str
    #: Google Drive's file id.
    file_id: str | None = None

    @property
    def label(self) -> str:
        return LABELS[self.kind]


def _host(parsed) -> str:
    host = (parsed.hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def classify(raw: str) -> FileLink | None:
    """Recognise a file link. None when ``raw`` is not one.

    Raises :class:`IngestError` for links that are recognisably the wrong
    thing (a Drive folder or document, a Dropbox folder), so the user hears
    what to share instead of a generic "unsupported link".
    """
    try:
        parsed = urlparse(raw.strip())
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    host = _host(parsed)
    path = parsed.path or ""
    query = dict(parse_qsl(parsed.query))

    if host in _DRIVE_HOSTS:
        if "/folders/" in path or path == "/embeddedfolderview":
            raise IngestError(
                "That is a Google Drive folder link.",
                hint="Paste it into the link field: AutoClip lists the videos in the "
                "folder so you can pick which to clip.",
            )
        if re.search(r"/(document|spreadsheets|presentation|forms)/", path):
            raise IngestError(
                "That Google link is not a video file.",
                hint="Share the video file itself (Share, then Copy link).",
            )
        match = re.search(r"/(?:file/)?d/([\w-]{10,})", path)
        file_id = match.group(1) if match else query.get("id")
        if not file_id or not re.fullmatch(r"[\w-]{10,}", file_id):
            raise IngestError(
                "Could not find the file in that Google Drive link.",
                hint="In Drive, use Share, then Copy link on the video file.",
            )
        return FileLink("gdrive", raw.strip(), file_id)

    if host in _DROPBOX_HOSTS:
        if path.startswith(("/home", "/sh/", "/scl/fo/")):
            raise IngestError(
                "That is a Dropbox folder link.",
                hint="Share the video file itself, not the folder.",
            )
        return FileLink("dropbox", raw.strip())

    if host in ("1drv.ms", "onedrive.live.com") or host.endswith(".sharepoint.com"):
        return FileLink("onedrive", raw.strip())

    if path.lower().endswith(DIRECT_SUFFIXES):
        return FileLink("direct", raw.strip())
    return None


def is_file_link(raw: str) -> bool:
    """True for any link :func:`classify` recognises, usable or not."""
    try:
        return classify(raw) is not None
    except IngestError:
        return True


def _with_query(url: str, **updates: str) -> str:
    parsed = urlparse(url)
    pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k not in updates]
    pairs.extend(updates.items())
    return urlunparse(parsed._replace(query=urlencode(pairs)))


def drive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def direct_url(link: FileLink) -> str:
    """The URL that downloads the file itself."""
    if link.kind == "gdrive":
        assert link.file_id
        return drive_download_url(link.file_id)
    if link.kind == "dropbox":
        if _host(urlparse(link.url)) == "dl.dropboxusercontent.com":
            return link.url
        parsed = urlparse(link.url)
        pairs = [(k, v) for k, v in parse_qsl(parsed.query) if k not in ("dl", "raw")]
        pairs.append(("dl", "1"))
        return urlunparse(parsed._replace(query=urlencode(pairs)))
    if link.kind == "onedrive":
        host = _host(urlparse(link.url))
        if host.endswith(".sharepoint.com"):
            return _with_query(link.url, download="1")
        # The documented "shares" API: base64url of the share link, unpadded.
        token = base64.urlsafe_b64encode(link.url.encode("utf-8")).decode("ascii").rstrip("=")
        return f"https://api.onedrive.com/v1.0/shares/u!{token}/root/content"
    return link.url


# --------------------------------------------------------------------------
# Google Drive's "can't scan this file for viruses" page
# --------------------------------------------------------------------------


def _attr(tag: str, name: str) -> str | None:
    match = re.search(rf'\b{name}\s*=\s*"([^"]*)"', tag, re.IGNORECASE) or re.search(
        rf"\b{name}\s*=\s*'([^']*)'", tag, re.IGNORECASE
    )
    return html_lib.unescape(match.group(1)) if match else None


def drive_confirm_url(
    page: str, *, base_url: str, file_id: str | None, cookies: dict[str, str] | None = None
) -> str | None:
    """Where Drive's large-file warning page says the file really is.

    Three generations of the page are understood: the current download form
    (``<form id="download-form" action=...>`` with hidden inputs), the older
    ``href="/uc?export=download&confirm=TOKEN&id=..."`` link, and the oldest
    ``download_warning_*`` cookie carrying the token. None when the page is
    none of them (a sign-in page, a quota page, anything else).
    """
    form = re.search(
        r"<form\b[^>]*\bid\s*=\s*[\"']download-form[\"'][^>]*>(.*?)</form>",
        page,
        re.IGNORECASE | re.DOTALL,
    )
    if form:
        opening = form.group(0)[: form.group(0).index(">") + 1]
        action = _attr(opening, "action")
        if action:
            params: list[tuple[str, str]] = []
            for tag in re.findall(r"<input\b[^>]*>", form.group(1), re.IGNORECASE):
                if (_attr(tag, "type") or "").lower() != "hidden":
                    continue
                name = _attr(tag, "name")
                if name:
                    params.append((name, _attr(tag, "value") or ""))
            target = urljoin(base_url, action)
            joiner = "&" if "?" in target else "?"
            return f"{target}{joiner}{urlencode(params)}" if params else target

    for href in re.findall(r"href\s*=\s*\"([^\"]*confirm=[^\"]*)\"", page, re.IGNORECASE):
        href = html_lib.unescape(href)
        if "export=download" in href or "/uc?" in href or "/download" in href:
            return urljoin(base_url, href)

    for name, value in (cookies or {}).items():
        if name.startswith("download_warning") and value and file_id:
            return f"{drive_download_url(file_id)}&confirm={value}"
    return None


_PRIVATE_MARKERS = (
    "accounts.google.com/servicelogin",
    "accounts.google.com/v3/signin",
    "you need access",
    "request access",
    "you need permission",
)
_QUOTA_MARKERS = ("too many users have viewed or downloaded", "download quota", "quota exceeded")


def _drive_page_error(page: str) -> IngestError:
    lowered = page.lower()
    if any(marker in lowered for marker in _QUOTA_MARKERS):
        return IngestError(
            "Google Drive's download limit for this file was reached.",
            hint="Too many people downloaded it recently. Try again in a few hours, or "
            "make a copy in your own Drive and share that.",
        )
    if any(marker in lowered for marker in _PRIVATE_MARKERS):
        return IngestError(
            "This Google Drive file is private.",
            hint='In Drive, set sharing to "Anyone with the link", then try again.',
        )
    return IngestError(
        "Google Drive did not send the video.",
        hint='Check that the file is a video shared with "Anyone with the link".',
    )


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def resolve_host(host: str) -> list[str]:
    """Every address ``host`` resolves to. Patched out in tests."""
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [str(info[4][0]) for info in infos]


def assert_public_url(url: str) -> None:
    """Refuse anything but http(s) to an ordinary network address.

    Private LAN addresses are allowed (a NAS share is a legitimate source);
    loopback, link-local (where cloud metadata services live), multicast and
    unspecified addresses are not.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise IngestError("Only http(s) links can be downloaded.")
    try:
        addresses = resolve_host(parsed.hostname)
    except OSError as exc:
        raise IngestError(f"Could not find the host {parsed.hostname}: {exc}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            raise IngestError(
                f"{parsed.hostname} points at a local or special address; "
                "AutoClip only downloads from public hosts."
            )


def new_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(30.0, read=120.0),
        follow_redirects=False,
    )


def open_url(client: httpx.Client, url: str) -> httpx.Response:
    """GET ``url`` as a stream, following redirects by hand.

    By hand so every hop is checked by :func:`assert_public_url`, not just
    the first. The caller closes the returned response.
    """
    for _ in range(MAX_REDIRECTS + 1):
        assert_public_url(url)
        response = client.send(client.build_request("GET", url), stream=True)
        location = response.headers.get("location")
        if response.is_redirect and location:
            response.close()
            url = urljoin(str(response.url), location)
            continue
        return response
    raise IngestError("The link redirected too many times.")


def read_text(response: httpx.Response, limit: int = MAX_HTML_BYTES) -> str:
    """Read a (small) text body from a streamed response, capped."""
    data = bytearray()
    for chunk in response.iter_bytes():
        data.extend(chunk)
        if len(data) > limit:
            break
    return bytes(data).decode(response.encoding or "utf-8", errors="replace")


def _content_type(response: httpx.Response) -> str:
    return response.headers.get("content-type", "").split(";")[0].strip().lower()


def _is_page(response: httpx.Response) -> bool:
    return _content_type(response).startswith(_PAGE_TYPES)


def _check_status(response: httpx.Response, where: str) -> None:
    status = response.status_code
    if status in (401, 403):
        raise IngestError(
            f"{where} refused the download.",
            hint='Make sure the file is shared publicly ("Anyone with the link").',
        )
    if status == 404:
        raise IngestError(f"{where} says this file does not exist (or is not shared).")
    if status >= 400:
        raise IngestError(f"Could not download from {where} (HTTP {status}).")


def filename_from(response: httpx.Response, fallback: str) -> str:
    """The file's own name: Content-Disposition first, then the URL path."""
    disposition = response.headers.get("content-disposition", "")
    name = ""
    star = re.search(r"filename\*\s*=\s*(?:[\w-]+'[^']*')?\"?([^\";]+)\"?", disposition, re.I)
    plain = re.search(r"filename\s*=\s*\"?([^\";]+)\"?", disposition, re.I)
    if star:
        name = unquote(star.group(1))
    elif plain:
        name = plain.group(1)
    if not name:
        name = unquote(Path(urlparse(str(response.url)).path).name)
    # Never trust a remote name as a path: keep the last component only.
    name = re.split(r"[\\/]", name.strip())[-1].strip().strip(".")
    if not name or name.lower() in ("download", "content", "uc"):
        return fallback
    return name[:200]


def choose_suffix(name: str, content_type: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in ACCEPTED_SUFFIXES:
        return suffix
    return _SUFFIX_BY_TYPE.get(content_type, ".mp4")


def _mb(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def download(
    link: FileLink,
    dest_dir: Path,
    *,
    max_bytes: int,
    on_progress: ProgressFn | None = None,
    client: httpx.Client | None = None,
) -> tuple[Path, str]:
    """Download the linked file into ``dest_dir``.

    Returns the saved path and the file's own name (for the title). The file
    is written as ``*.part`` and renamed only once complete, so a failure
    never leaves something that looks like a finished download.
    """
    where = link.label
    own_client = client is None
    client = client or new_client()
    response: httpx.Response | None = None
    part: Path | None = None
    try:
        try:
            response = open_url(client, direct_url(link))
            if link.kind == "gdrive" and response.status_code < 400 and _is_page(response):
                page = read_text(response)
                response.close()
                next_url = drive_confirm_url(
                    page,
                    base_url=str(response.url),
                    file_id=link.file_id,
                    cookies=dict(client.cookies.items()),
                )
                if next_url is None:
                    raise _drive_page_error(page)
                response = open_url(client, next_url)
                if response.status_code < 400 and _is_page(response):
                    raise _drive_page_error(read_text(response))

            _check_status(response, where)
            content_type = _content_type(response)
            if _is_page(response):
                raise IngestError(
                    f"{where} sent a web page instead of a video.",
                    hint="Check that the link points at a video file shared with "
                    '"Anyone with the link".',
                )

            total = int(response.headers.get("content-length") or 0) or None
            if total and total > max_bytes:
                raise IngestError(
                    f"This file is {_mb(total)}; the limit is {_mb(max_bytes)}.",
                    hint="Raise the link size limit in Settings to download it.",
                )

            name = filename_from(response, "video.mp4")
            suffix = choose_suffix(name, content_type)
            dest_dir.mkdir(parents=True, exist_ok=True)
            part = dest_dir / f"download{suffix}.part"
            done = 0
            last_report = 0.0
            with part.open("wb") as handle:
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    done += len(chunk)
                    if done > max_bytes:
                        raise IngestError(
                            f"This file is bigger than the {_mb(max_bytes)} limit.",
                            hint="Raise the link size limit in Settings to download it.",
                        )
                    handle.write(chunk)
                    now = time.monotonic()
                    if on_progress and now - last_report >= PROGRESS_EVERY_S:
                        last_report = now
                        on_progress(done, total)
        except httpx.HTTPError as exc:
            raise IngestError(f"Could not download from {where}: {exc}") from exc

        if done < MIN_BYTES:
            raise IngestError(
                f"{where} sent an empty or tiny file.",
                hint="Is the link right, and is the file shared publicly?",
            )
        if total and done < total:
            raise IngestError(f"The download from {where} stopped early ({_mb(done)}).")
        if on_progress:
            on_progress(done, total or done)
        final = dest_dir / f"download{suffix}"
        part.replace(final)
        part = None
        log.info("Downloaded %s from %s (%s).", name, where, _mb(done))
        return final, name
    finally:
        if response is not None:
            response.close()
        if part is not None:
            part.unlink(missing_ok=True)
        if own_client:
            client.close()

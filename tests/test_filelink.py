"""File links as sources: link conversion, Drive's confirm page, folders.

No real network: every HTTP exchange goes through ``httpx.MockTransport`` and
host resolution is patched to a public address.
"""

# Canned Drive pages keep their real, long lines.
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from autoclip.app import ENV_NO_WORKER, create_app
from autoclip.db import store
from autoclip.db.models import Source, new_id
from autoclip.pipeline import drivefolder, ffmpeg, filelink, ingest
from autoclip.pipeline.ingest import IngestError
from fastapi.testclient import TestClient

VIDEO = b"\x00\x00\x00\x18ftypmp42" + b"v" * 200_000
DRIVE_ID = "1AbCdEfGhIjKlMnOpQrStUv"


@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every host resolves to a public address (no real DNS)."""
    monkeypatch.setattr(filelink, "resolve_host", lambda host: ["93.184.216.34"])


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


# --------------------------------------------------------------------------
# Classification and direct URLs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("link", "kind", "direct"),
    [
        (
            f"https://drive.google.com/file/d/{DRIVE_ID}/view?usp=sharing",
            "gdrive",
            f"https://drive.google.com/uc?export=download&id={DRIVE_ID}",
        ),
        (
            f"https://drive.google.com/open?id={DRIVE_ID}",
            "gdrive",
            f"https://drive.google.com/uc?export=download&id={DRIVE_ID}",
        ),
        (
            f"https://drive.google.com/uc?id={DRIVE_ID}&export=download",
            "gdrive",
            f"https://drive.google.com/uc?export=download&id={DRIVE_ID}",
        ),
        (
            f"https://docs.google.com/file/d/{DRIVE_ID}/edit",
            "gdrive",
            f"https://drive.google.com/uc?export=download&id={DRIVE_ID}",
        ),
        (
            "https://www.dropbox.com/s/abc123/talk.mp4?dl=0",
            "dropbox",
            "https://www.dropbox.com/s/abc123/talk.mp4?dl=1",
        ),
        (
            "https://www.dropbox.com/scl/fi/xyz/talk.mov?rlkey=k1&dl=0",
            "dropbox",
            "https://www.dropbox.com/scl/fi/xyz/talk.mov?rlkey=k1&dl=1",
        ),
        (
            "https://www.dropbox.com/scl/fi/xyz/talk.mov?rlkey=k1&raw=1",
            "dropbox",
            "https://www.dropbox.com/scl/fi/xyz/talk.mov?rlkey=k1&dl=1",
        ),
        (
            "https://dl.dropboxusercontent.com/s/abc/talk.mp4",
            "dropbox",
            "https://dl.dropboxusercontent.com/s/abc/talk.mp4",
        ),
        (
            # Microsoft's documented example share link and its encoding.
            "https://onedrive.live.com/redir?resid=1231244193912!12&authKey=1201919!12921!1",
            "onedrive",
            "https://api.onedrive.com/v1.0/shares/"
            "u!aHR0cHM6Ly9vbmVkcml2ZS5saXZlLmNvbS9yZWRpcj9yZXNpZD0xMjMxMjQ0MTkzOTEyITEy"
            "JmF1dGhLZXk9MTIwMTkxOSExMjkyMSEx/root/content",
        ),
        (
            "https://contoso.sharepoint.com/:v:/g/personal/me/EaBcD?e=xYz",
            "onedrive",
            "https://contoso.sharepoint.com/:v:/g/personal/me/EaBcD?e=xYz&download=1",
        ),
        (
            "https://cdn.example.com/media/keynote.MP4?sig=1",
            "direct",
            "https://cdn.example.com/media/keynote.MP4?sig=1",
        ),
        ("https://example.com/v/clip.webm", "direct", "https://example.com/v/clip.webm"),
        ("https://example.com/v/clip.mkv", "direct", "https://example.com/v/clip.mkv"),
        ("https://example.com/v/clip.mov", "direct", "https://example.com/v/clip.mov"),
    ],
)
def test_share_links_become_direct_download_urls(link: str, kind: str, direct: str) -> None:
    parsed = filelink.classify(link)

    assert parsed is not None
    assert parsed.kind == kind
    assert filelink.direct_url(parsed) == direct


def test_1drv_short_links_use_the_shares_api() -> None:
    parsed = filelink.classify("https://1drv.ms/v/s!AbCdEf")

    assert parsed is not None and parsed.kind == "onedrive"
    assert filelink.direct_url(parsed).startswith("https://api.onedrive.com/v1.0/shares/u!")
    assert filelink.direct_url(parsed).endswith("/root/content")


@pytest.mark.parametrize(
    "link",
    [
        "https://www.youtube.com/watch?v=abc",
        "https://example.com/page.html",
        "ftp://example.com/video.mp4",
        "not a url",
        "",
    ],
)
def test_other_links_are_not_file_links(link: str) -> None:
    assert filelink.classify(link) is None
    assert not filelink.is_file_link(link)


@pytest.mark.parametrize(
    ("link", "message"),
    [
        (f"https://drive.google.com/drive/folders/{DRIVE_ID}", "folder"),
        (f"https://docs.google.com/document/d/{DRIVE_ID}/edit", "not a video"),
        ("https://drive.google.com/file/view", "Could not find the file"),
        ("https://www.dropbox.com/sh/abc/AAA?dl=0", "Dropbox folder"),
        ("https://www.dropbox.com/scl/fo/abc/xyz?rlkey=1", "Dropbox folder"),
    ],
)
def test_recognised_but_unusable_links_explain_themselves(link: str, message: str) -> None:
    with pytest.raises(IngestError, match=message):
        filelink.classify(link)
    assert filelink.is_file_link(link)


# --------------------------------------------------------------------------
# Drive's large-file confirm page
# --------------------------------------------------------------------------

CONFIRM_FORM = f"""<!DOCTYPE html><html><head><title>Google Drive - Virus scan warning</title>
</head><body><div class="uc-main"><p class="uc-warning-subcaption">Google Drive can't scan
this file for viruses.</p><form id="download-form"
action="https://drive.usercontent.google.com/download" method="get">
<input type="submit" id="uc-download-link" value="Download anyway">
<input type="hidden" name="id" value="{DRIVE_ID}">
<input type="hidden" name="export" value="download">
<input type="hidden" name="confirm" value="t">
<input type="hidden" name="uuid" value="5f1e&amp;x"></form></div></body></html>"""

CONFIRM_HREF = f"""<html><body><a id="uc-download-link" class="goog-inline-block jfk-button"
href="/uc?export=download&amp;confirm=AbCd&amp;id={DRIVE_ID}">Download anyway</a>
</body></html>"""

SIGN_IN = """<html><head><title>Google Drive: Sign-in</title></head><body>
<a href="https://accounts.google.com/ServiceLogin?service=wise">Sign in</a></body></html>"""


def test_confirm_url_from_the_download_form() -> None:
    url = filelink.drive_confirm_url(
        CONFIRM_FORM, base_url="https://drive.google.com/uc", file_id=DRIVE_ID
    )

    assert url == (
        "https://drive.usercontent.google.com/download?"
        f"id={DRIVE_ID}&export=download&confirm=t&uuid=5f1e%26x"
    )


def test_confirm_url_from_the_older_confirm_link() -> None:
    url = filelink.drive_confirm_url(
        CONFIRM_HREF, base_url="https://drive.google.com/uc?export=download", file_id=DRIVE_ID
    )

    assert url == f"https://drive.google.com/uc?export=download&confirm=AbCd&id={DRIVE_ID}"


def test_confirm_url_from_the_download_warning_cookie() -> None:
    url = filelink.drive_confirm_url(
        "<html>nothing here</html>",
        base_url="https://drive.google.com/uc",
        file_id=DRIVE_ID,
        cookies={"NID": "1", "download_warning_13058876669334088843_abc": "Tok3n"},
    )

    assert url == f"https://drive.google.com/uc?export=download&id={DRIVE_ID}&confirm=Tok3n"


def test_no_confirm_url_on_a_sign_in_page() -> None:
    assert (
        filelink.drive_confirm_url(SIGN_IN, base_url="https://drive.google.com", file_id=DRIVE_ID)
        is None
    )


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------


def video_response(request: httpx.Request, **headers: str) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "video/mp4", "content-length": str(len(VIDEO)), **headers},
        content=VIDEO,
        request=request,
    )


def test_direct_download_streams_to_disk_with_progress(tmp_path: Path) -> None:
    seen: list[tuple[int, int | None]] = []
    client = mock_client(lambda request: video_response(request))
    link = filelink.classify("https://cdn.example.com/talk.mp4")
    assert link is not None

    path, name = filelink.download(
        link, tmp_path, max_bytes=10**9, on_progress=lambda d, t: seen.append((d, t)), client=client
    )

    assert path.read_bytes() == VIDEO
    assert path.suffix == ".mp4"
    assert name == "talk.mp4"
    assert seen[-1] == (len(VIDEO), len(VIDEO))
    assert not list(tmp_path.glob("*.part"))


def test_drive_confirm_page_is_followed_with_its_cookies(tmp_path: Path) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.host == "drive.google.com":
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/html; charset=utf-8",
                    "set-cookie": "drive_session=abc; Path=/; Domain=.google.com",
                },
                text=CONFIRM_FORM,
                request=request,
            )
        assert request.url.host == "drive.usercontent.google.com"
        assert request.url.params["confirm"] == "t"
        return video_response(
            request, **{"content-disposition": 'attachment; filename="My Talk.mov"'}
        )

    link = filelink.classify(f"https://drive.google.com/file/d/{DRIVE_ID}/view")
    assert link is not None
    path, name = filelink.download(link, tmp_path, max_bytes=10**9, client=mock_client(handler))

    assert len(requests) == 2
    assert name == "My Talk.mov"
    assert path.suffix == ".mov"
    assert path.read_bytes() == VIDEO


def test_private_drive_file_says_so(tmp_path: Path) -> None:
    client = mock_client(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/html"}, text=SIGN_IN, request=request
        )
    )
    link = filelink.classify(f"https://drive.google.com/file/d/{DRIVE_ID}/view")
    assert link is not None

    with pytest.raises(IngestError, match="private") as caught:
        filelink.download(link, tmp_path, max_bytes=10**9, client=client)
    assert "Anyone with the link" in caught.value.hint


def test_a_web_page_instead_of_the_file_is_refused(tmp_path: Path) -> None:
    client = mock_client(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/html"}, text="<html>hi</html>", request=request
        )
    )
    link = filelink.classify("https://www.dropbox.com/s/abc/talk.mp4?dl=0")
    assert link is not None

    with pytest.raises(IngestError, match="web page"):
        filelink.download(link, tmp_path / "dl", max_bytes=10**9, client=client)
    assert not (tmp_path / "dl").exists()  # refused before anything was written


def test_size_cap_from_content_length(tmp_path: Path) -> None:
    client = mock_client(lambda request: video_response(request))
    link = filelink.classify("https://cdn.example.com/talk.mp4")
    assert link is not None

    with pytest.raises(IngestError, match="limit"):
        filelink.download(link, tmp_path, max_bytes=1000, client=client)


def test_size_cap_while_streaming_without_content_length(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            content=iter([VIDEO[:100_000], VIDEO[100_000:]]),
            request=request,
        )

    link = filelink.classify("https://cdn.example.com/talk.mkv")
    assert link is not None
    with pytest.raises(IngestError, match="bigger than"):
        filelink.download(link, tmp_path, max_bytes=150_000, client=mock_client(handler))
    assert not list(tmp_path.glob("*.part"))


def test_tiny_files_are_refused(tmp_path: Path) -> None:
    client = mock_client(
        lambda request: httpx.Response(
            200, headers={"content-type": "video/mp4"}, content=b"tiny", request=request
        )
    )
    link = filelink.classify("https://cdn.example.com/talk.mp4")
    assert link is not None

    with pytest.raises(IngestError, match="tiny"):
        filelink.download(link, tmp_path, max_bytes=10**9, client=client)


@pytest.mark.parametrize(("status", "message"), [(403, "refused"), (404, "does not exist")])
def test_http_errors_are_explained(tmp_path: Path, status: int, message: str) -> None:
    client = mock_client(lambda request: httpx.Response(status, request=request))
    link = filelink.classify("https://cdn.example.com/talk.mp4")
    assert link is not None

    with pytest.raises(IngestError, match=message):
        filelink.download(link, tmp_path, max_bytes=10**9, client=client)


def test_redirect_to_a_loopback_address_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        filelink,
        "resolve_host",
        lambda host: ["127.0.0.1"] if host == "internal.test" else ["93.184.216.34"],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": "http://internal.test/secret.mp4"}, request=request
        )

    link = filelink.classify("https://cdn.example.com/talk.mp4")
    assert link is not None
    with pytest.raises(IngestError, match="local or special address"):
        filelink.download(link, tmp_path, max_bytes=10**9, client=mock_client(handler))


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "::1", "0.0.0.0"])
def test_special_addresses_are_refused(monkeypatch: pytest.MonkeyPatch, address: str) -> None:
    monkeypatch.setattr(filelink, "resolve_host", lambda host: [address])

    with pytest.raises(IngestError):
        filelink.assert_public_url("https://example.com/v.mp4")


def test_lan_addresses_are_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(filelink, "resolve_host", lambda host: ["192.168.1.20"])

    filelink.assert_public_url("http://nas.local/v.mp4")


@pytest.mark.parametrize(
    ("disposition", "expected"),
    [
        ('attachment; filename="a b.mp4"', "a b.mp4"),
        ("attachment; filename*=UTF-8''caf%C3%A9.mov", "café.mov"),
        ('attachment; filename="../../evil.mp4"', "evil.mp4"),
        ("", "talk.webm"),
    ],
)
def test_filename_from_headers(disposition: str, expected: str) -> None:
    response = httpx.Response(
        200,
        headers={"content-disposition": disposition} if disposition else {},
        request=httpx.Request("GET", "https://cdn.example.com/x/talk.webm"),
    )

    assert filelink.filename_from(response, "video.mp4") == expected


# --------------------------------------------------------------------------
# ingest_link: download, then the upload path
# --------------------------------------------------------------------------


@pytest.fixture
def fake_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def probe(path: Path) -> ffmpeg.MediaInfo:
        return ffmpeg.MediaInfo(
            path=path, duration_s=42.0, has_video=True, has_audio=True, width=1920, height=1080
        )

    monkeypatch.setattr(ingest, "_probe_and_validate", probe)


def test_ingest_link_hands_the_file_to_the_upload_flow(
    autoclip_home: Path, monkeypatch: pytest.MonkeyPatch, fake_probe: None
) -> None:
    client = mock_client(
        lambda request: video_response(
            request, **{"content-disposition": 'attachment; filename="Keynote.mp4"'}
        )
    )
    monkeypatch.setattr(filelink, "new_client", lambda: client)
    link = "https://www.dropbox.com/s/abc/Keynote.mp4?dl=0"

    source = ingest.ingest_link(link)

    assert source.type == "upload"
    assert source.url == link
    assert source.title == "Keynote"
    assert source.filename == "Keynote.mp4"
    assert Path(source.path).read_bytes() == VIDEO
    assert Path(source.path).parent.parent == autoclip_home / "media"
    # The staging folder is gone.
    assert not any((autoclip_home / "media" / ".incoming").iterdir())


def test_ingest_link_rejects_unknown_links() -> None:
    with pytest.raises(IngestError, match="not a supported"):
        ingest.ingest_link("https://example.com/page")


# --------------------------------------------------------------------------
# Drive folders
# --------------------------------------------------------------------------

SUB_ID = "1SubFolderIdxxxxxxxx"
FOLDER_PAGE = f"""<!DOCTYPE html><html><head><title>Stream VODs - Google Drive</title></head>
<body><div class="flip-entries">
<div class="flip-entry" id="entry-1VidOneIdxxxxxxxxx" tabindex="0" role="link">
<div class="flip-entry-info"><a href="https://drive.google.com/file/d/1VidOneIdxxxxxxxxx/view?usp=drive_web" target="_blank">
<div class="flip-entry-list-icon"><img src="https://drive-thirdparty.googleusercontent.com/16/type/video/mp4" alt=""></div>
<div class="flip-entry-title">Episode 1 &amp; 2.mp4</div></a></div></div>
<div class="flip-entry" id="entry-1DocIdxxxxxxxxxxxx" tabindex="0" role="link">
<div class="flip-entry-info"><a href="https://docs.google.com/document/d/1DocIdxxxxxxxxxxxx/edit?usp=drive_web" target="_blank">
<div class="flip-entry-list-icon"><img src="https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.google-apps.document" alt=""></div>
<div class="flip-entry-title">Notes</div></a></div></div>
<div class="flip-entry" id="entry-{SUB_ID}" tabindex="0" role="link">
<div class="flip-entry-info"><a href="https://drive.google.com/drive/folders/{SUB_ID}" target="_blank">
<div class="flip-entry-list-icon"><img src="https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.google-apps.folder" alt=""></div>
<div class="flip-entry-title">Raw</div></a></div></div>
</div></body></html>"""

SUB_PAGE = """<html><head><title>Raw - Google Drive</title></head><body><div class="flip-entries">
<div class="flip-entry" id="entry-1VidTwoIdxxxxxxxxx" tabindex="0" role="link">
<div class="flip-entry-info"><a href="https://drive.google.com/file/d/1VidTwoIdxxxxxxxxx/view">
<div class="flip-entry-list-icon"><img src="https://drive-thirdparty.googleusercontent.com/16/type/video/quicktime" alt=""></div>
<div class="flip-entry-title">cam.mov</div></a></div></div>
</div></body></html>"""

FOLDER_ID = "1FolderIdxxxxxxxxxxxx"


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        (f"https://drive.google.com/drive/folders/{FOLDER_ID}?usp=sharing", FOLDER_ID),
        (f"https://drive.google.com/drive/u/0/folders/{FOLDER_ID}", FOLDER_ID),
        (f"https://drive.google.com/embeddedfolderview?id={FOLDER_ID}", FOLDER_ID),
        (f"https://drive.google.com/file/d/{FOLDER_ID}/view", None),
        ("https://example.com/folders/abcdefghijkl", None),
    ],
)
def test_folder_ids(link: str, expected: str | None) -> None:
    assert drivefolder.folder_id(link) == expected


def test_parse_embedded_view() -> None:
    title, entries = drivefolder.parse_embedded_view(FOLDER_PAGE)

    assert title == "Stream VODs"
    assert [(e.id, e.name, e.mime_type, e.is_folder) for e in entries] == [
        ("1VidOneIdxxxxxxxxx", "Episode 1 & 2.mp4", "video/mp4", False),
        ("1DocIdxxxxxxxxxxxx", "Notes", "application/vnd.google-apps.document", False),
        (SUB_ID, "Raw", drivefolder.FOLDER_MIME, True),
    ]


def test_list_folder_keeps_media_and_walks_subfolders() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = SUB_PAGE if request.url.params["id"] == SUB_ID else FOLDER_PAGE
        return httpx.Response(200, headers={"content-type": "text/html"}, text=page)

    listing = drivefolder.list_folder(FOLDER_ID, client=mock_client(handler))

    assert listing.name == "Stream VODs"
    assert [(f.name, f.folder) for f in listing.files] == [
        ("Episode 1 & 2.mp4", ""),
        ("cam.mov", "Raw"),
    ]
    assert listing.files[0].url == "https://drive.google.com/file/d/1VidOneIdxxxxxxxxx/view"
    assert not listing.truncated


def test_private_folder_is_a_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "drive.google.com":
            return httpx.Response(
                302, headers={"location": "https://accounts.google.com/ServiceLogin?x=1"}
            )
        return httpx.Response(200, headers={"content-type": "text/html"}, text=SIGN_IN)

    with pytest.raises(IngestError, match="private"):
        drivefolder.list_folder(FOLDER_ID, client=mock_client(handler))


def test_unparseable_folder_page_is_a_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/html"}, text="<html><title>New Drive</title>x"
        )

    with pytest.raises(IngestError, match="Could not read that Google Drive folder"):
        drivefolder.list_folder(FOLDER_ID, client=mock_client(handler))


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@pytest.fixture
def client(autoclip_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


def _fake_source(url: str) -> Source:
    return Source(id=new_id(), type="upload", path="C:/media/x.mp4", title="Linked", url=url)


def test_youtube_endpoint_accepts_file_links(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_ingest_link(url, settings=None, *, on_progress=None):
        calls.append(url)
        on_progress(10, 100)
        return _fake_source(url)

    monkeypatch.setattr(ingest, "ingest_link", fake_ingest_link)
    link = "https://www.dropbox.com/s/abc/talk.mp4?dl=0"

    response = client.post("/api/sources/youtube", json={"url": link})

    assert response.status_code == 201, response.text
    assert response.json()["url"] == link
    assert calls == [link]
    assert store.get_source(response.json()["id"]) is not None


def test_link_endpoint_reports_progress(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_ingest_link(url, settings=None, *, on_progress=None):
        on_progress(25, 100)
        return _fake_source(url)

    monkeypatch.setattr(ingest, "ingest_link", fake_ingest_link)

    response = client.post(
        "/api/sources/link",
        json={"url": "https://cdn.example.com/a.mp4", "progress_id": "abc123"},
    )
    progress = client.get("/api/sources/link/progress/abc123").json()

    assert response.status_code == 201
    assert progress == {"done_bytes": 25, "total_bytes": 100, "state": "done"}


def test_link_endpoint_explains_failures(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(url, settings=None, *, on_progress=None):
        raise IngestError("This Google Drive file is private.", hint="Share it.")

    monkeypatch.setattr(ingest, "ingest_link", fail)

    response = client.post(
        "/api/sources/link", json={"url": f"https://drive.google.com/file/d/{DRIVE_ID}/view"}
    )

    assert response.status_code == 422
    assert response.json()["detail"]["hint"] == "Share it."


def test_drive_folder_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    listing = drivefolder.DriveFolder(
        id=FOLDER_ID,
        name="VODs",
        files=[drivefolder.DriveFile("1VidOneIdxxxxxxxxx", "a.mp4", "video/mp4", "")],
    )
    monkeypatch.setattr(drivefolder, "list_folder", lambda folder: listing)

    body = client.post(
        "/api/sources/drive-folder",
        json={"url": f"https://drive.google.com/drive/folders/{FOLDER_ID}"},
    ).json()

    assert body["name"] == "VODs"
    assert body["files"][0]["url"] == "https://drive.google.com/file/d/1VidOneIdxxxxxxxxx/view"
    folder_as_file = client.post(
        "/api/sources/link", json={"url": f"https://drive.google.com/drive/folders/{FOLDER_ID}"}
    )
    assert folder_as_file.status_code == 400

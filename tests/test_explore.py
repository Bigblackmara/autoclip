"""Explore: YouTube / Twitch search, channel tabs and Peek.

yt-dlp is never run: ``explore.ytdlp.extract`` is replaced by a fake that
returns canned info dicts shaped like yt-dlp's flat / full output.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from autoclip import app as app_module
from autoclip import explore
from autoclip.app import create_app
from autoclip.explore import cache as cache_module
from autoclip.explore import service, twitch, youtube, ytdlp
from autoclip.explore.models import ExploreItem, SearchResult
from fastapi.testclient import TestClient


class FakeYtDlp:
    """Records every extract() call and answers from a URL -> info table."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses: dict[str, Any] = responses or {}
        self.calls: list[tuple[str, bool, int | None]] = []

    def __call__(self, url: str, *, flat: bool, limit: int | None = None) -> dict[str, Any]:
        self.calls.append((url, flat, limit))
        for key, value in self.responses.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected yt-dlp call: {url}")


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    explore.clear_cache()
    twitch.reset_token()
    yield
    explore.clear_cache()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeYtDlp:
    f = FakeYtDlp()
    monkeypatch.setattr(ytdlp, "extract", f)
    return f


@pytest.fixture
def no_twitch_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(twitch, "credentials", lambda: None)


# --------------------------------------------------------------------------
# Canned yt-dlp output
# --------------------------------------------------------------------------


def _video(vid: str, **extra: Any) -> dict[str, Any]:
    return {
        "_type": "url",
        "ie_key": "Youtube",
        "id": vid,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "title": f"Video {vid}",
        "channel": "Some Podcast",
        "channel_id": "UCabcdefghijklmnopqrstuv",
        "channel_url": "https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv",
        "duration": 3600.0,
        "view_count": 120000,
        "thumbnails": [
            {"url": f"https://i.ytimg.com/vi/{vid}/small.jpg", "width": 168, "height": 94},
            {"url": f"https://i.ytimg.com/vi/{vid}/big.jpg", "width": 336, "height": 188},
        ],
        "live_status": None,
        **extra,
    }


SEARCH_RESULTS = {
    "_type": "playlist",
    "id": "podcast",
    "entries": [
        _video("aaaaaaaaaaa"),
        {
            "_type": "url",
            "ie_key": "YoutubeTab",
            "id": "UCzzzzzzzzzzzzzzzzzzzzzz",
            "url": "https://www.youtube.com/channel/UCzzzzzzzzzzzzzzzzzzzzzz",
            "title": "A Channel",
            "channel_follower_count": 5000,
            "thumbnails": [{"url": "//yt3.ggpht.com/avatar.jpg", "width": 88, "height": 88}],
        },
        {
            "_type": "url",
            "ie_key": "YoutubeTab",
            "id": "PLxyz",
            "url": "https://www.youtube.com/playlist?list=PLxyz",
            "title": "A playlist",
        },
        _video("bbbbbbbbbbb", live_status="is_live", duration=None),
        _video("ccccccccccc", live_status="is_upcoming"),
        _video(
            "ddddddddddd",
            url="https://www.youtube.com/shorts/ddddddddddd",
            duration=42.0,
            upload_date="20261101",
        ),
    ],
}


# --------------------------------------------------------------------------
# YouTube
# --------------------------------------------------------------------------


class TestSearchParam:
    def test_plain_type_filters(self) -> None:
        assert youtube.search_param(type="videos") == "EgIQAQ=="
        assert youtube.search_param(type="channels") == "EgIQAg=="
        assert youtube.search_param(type="shorts") == "EgIQCQ=="

    def test_sort_and_filters(self) -> None:
        # Sort by view count, videos only — YouTube's own "CAMSAhAB".
        assert youtube.search_param(type="videos", sort="views") == "CAMSAhAB"
        # Newest, uploaded this week, over 20 minutes.
        assert (
            youtube.search_param(type="videos", sort="date", uploaded="week", duration="long")
            == "CAISBggDEAEYAg=="
        )

    def test_live_sets_live_feature(self) -> None:
        assert youtube.search_param(type="live") == "EgQQAUAB"

    def test_channels_ignore_sort_and_date(self) -> None:
        assert youtube.search_param(type="channels", sort="views", uploaded="week") == "EgIQAg=="

    def test_duration_only_applies_to_videos(self) -> None:
        assert youtube.search_param(type="shorts", duration="long") == "EgIQCQ=="

    def test_search_url_is_encoded(self) -> None:
        url = youtube.search_url("a b&c", "EgIQAQ==")
        assert url == "https://www.youtube.com/results?search_query=a+b%26c&sp=EgIQAQ%3D%3D"


class TestYouTubeSearch:
    def test_videos_parse_and_drop_channels_playlists_upcoming(self, fake: FakeYtDlp) -> None:
        fake.responses["ytsearch"] = SEARCH_RESULTS
        items = youtube.search("podcast", type="videos", limit=10)
        ids = [i.id for i in items]
        assert ids == ["aaaaaaaaaaa", "bbbbbbbbbbb", "ddddddddddd"]
        first = items[0]
        assert first.kind == "video"
        assert first.title == "Video aaaaaaaaaaa"
        assert first.channel == "Some Podcast"
        assert first.duration_s == 3600.0
        assert first.views == 120000
        assert first.thumbnail.endswith("/big.jpg")  # widest thumbnail wins
        assert first.url == "https://www.youtube.com/watch?v=aaaaaaaaaaa"
        assert items[1].is_live and items[1].kind == "live"
        assert items[2].kind == "short"
        assert items[2].upload_date == "2026-11-01"
        # A plain search uses yt-dlp's search prefix, flat, capped.
        assert fake.calls == [("ytsearch10:podcast", True, 10)]

    def test_filtered_search_uses_results_page_with_sp(self, fake: FakeYtDlp) -> None:
        fake.responses["youtube.com/results"] = SEARCH_RESULTS
        youtube.search("podcast", type="videos", sort="views", duration="long", limit=5)
        url, flat, limit = fake.calls[0]
        assert url.startswith("https://www.youtube.com/results?search_query=podcast&sp=")
        assert flat and limit == 5

    def test_channels_tab_keeps_only_channels(self, fake: FakeYtDlp) -> None:
        fake.responses["youtube.com/results"] = SEARCH_RESULTS
        items = youtube.search("podcast", type="channels")
        assert [i.kind for i in items] == ["channel"]
        assert items[0].subscribers == 5000
        assert items[0].thumbnail == "https://yt3.ggpht.com/avatar.jpg"

    def test_live_tab_marks_everything_live(self, fake: FakeYtDlp) -> None:
        fake.responses["youtube.com/results"] = SEARCH_RESULTS
        items = youtube.search("podcast", type="live")
        assert items and all(i.kind == "live" and i.is_live for i in items)

    def test_shorts_fall_back_to_hashtag_search(self, fake: FakeYtDlp) -> None:
        empty = {"_type": "playlist", "entries": [_video("aaaaaaaaaaa")]}  # no shorts
        loose = {
            "_type": "playlist",
            "entries": [
                _video("eeeeeeeeeee", duration=55.0),
                _video("fffffffffff", duration=600.0),
            ],
        }
        fake.responses["%23shorts"] = loose
        fake.responses["youtube.com/results"] = empty
        items = youtube.search("cats", type="shorts", limit=4)
        assert [i.id for i in items] == ["eeeeeeeeeee"]
        assert items[0].kind == "short"
        assert items[0].url == "https://www.youtube.com/shorts/eeeeeeeeeee"

    def test_empty_query_returns_nothing(self, fake: FakeYtDlp) -> None:
        assert youtube.search("   ") == []
        assert fake.calls == []


class TestYouTubeChannel:
    CHANNEL = {
        "_type": "playlist",
        "id": "UCabcdefghijklmnopqrstuv",
        "title": "Some Podcast - Videos",
        "channel": "Some Podcast",
        "channel_id": "UCabcdefghijklmnopqrstuv",
        "channel_url": "https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv",
        "channel_follower_count": 1_200_000,
        "description": "Long talks.",
        "thumbnails": [
            {
                "id": "banner_uncropped",
                "url": "https://yt3/banner.jpg",
                "width": 2560,
                "height": 424,
            },
            {
                "id": "avatar_uncropped",
                "url": "https://yt3/avatar.jpg",
                "width": 900,
                "height": 900,
            },
        ],
        "entries": [
            {k: v for k, v in _video("aaaaaaaaaaa").items() if not k.startswith("channel")},
            _video("bbbbbbbbbbb", live_status="was_live"),
        ],
    }

    def test_channel_base_normalises_inputs(self) -> None:
        assert youtube.channel_base("https://www.youtube.com/@mkbhd/videos?x=1") == (
            "https://www.youtube.com/@mkbhd"
        )
        assert youtube.channel_base("@mkbhd") == "https://www.youtube.com/@mkbhd"
        assert youtube.channel_base("UCabcdefghijklmnopqrstuv") == (
            "https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv"
        )
        assert youtube.channel_base("Lex Fridman") == "https://www.youtube.com/@LexFridman"

    def test_videos_tab(self, fake: FakeYtDlp) -> None:
        fake.responses["/videos"] = self.CHANNEL
        page = youtube.channel("https://www.youtube.com/@somepod", tab="videos", limit=20)
        assert fake.calls == [("https://www.youtube.com/@somepod/videos", True, 20)]
        assert page.channel.title == "Some Podcast"
        assert page.channel.avatar == "https://yt3/avatar.jpg"
        assert page.channel.banner == "https://yt3/banner.jpg"
        assert page.channel.subscribers == 1_200_000
        assert [i.id for i in page.items] == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
        # Entries without channel fields inherit them from the tab.
        assert page.items[0].channel == "Some Podcast"
        assert page.items[0].channel_url.endswith("UCabcdefghijklmnopqrstuv")
        assert not page.items[1].is_live  # a finished stream is clippable

    def test_shorts_and_streams_tabs(self, fake: FakeYtDlp) -> None:
        fake.responses["/shorts"] = self.CHANNEL
        fake.responses["/streams"] = self.CHANNEL
        shorts = youtube.channel("@somepod", tab="shorts")
        assert all(i.kind == "short" for i in shorts.items)
        streams = youtube.channel("@somepod", tab="streams")
        assert streams.tab == "streams"
        assert fake.calls[-1][0] == "https://www.youtube.com/@somepod/streams"


# --------------------------------------------------------------------------
# Twitch
# --------------------------------------------------------------------------


TWITCH_VODS = {
    "_type": "playlist",
    "id": "somestreamer",
    "entries": [
        {
            "_type": "url_transparent",
            "ie_key": "TwitchVod",
            "id": "v222",
            "url": "https://www.twitch.tv/videos/222",
            "title": "Live right now",
            "thumbnail": "https://vod-secure.twitch.tv/_404/404_processing_320x180.png",
            "duration": 5400.0,
            "view_count": 10,
        },
        {
            "_type": "url_transparent",
            "ie_key": "TwitchVod",
            "id": "v111",
            "url": "https://www.twitch.tv/videos/111",
            "title": "Yesterday's stream",
            "thumbnail": "https://static-cdn.jtvnw.net/cf_vods/x/thumb0-320x180.jpg",
            "duration": 14400.0,
            "view_count": 9000,
        },
        {
            "_type": "url_transparent",
            "id": "v000",
            "url": "https://www.twitch.tv/videos/000",
            "title": "Flagged live",
            "live_status": "is_live",
        },
    ],
}

TWITCH_CLIPS = {
    "_type": "playlist",
    "entries": [
        {
            "_type": "url_transparent",
            "ie_key": "TwitchClips",
            "id": "FunnyClipSlug",
            "url": "https://clips.twitch.tv/FunnyClipSlug",
            "title": "chat lost it",
            "thumbnail": "https://clips-media/thumb.jpg",
            "duration": 28.0,
            "timestamp": 1_763_000_000,
            "view_count": 50000,
        }
    ],
}


class TestTwitch:
    def test_login_parsing(self) -> None:
        assert twitch.login_of("https://www.twitch.tv/Shroud/videos") == "shroud"
        assert twitch.login_of("@xQc") == "xqc"
        assert twitch.is_valid_login("caseoh_")
        assert not twitch.is_valid_login("two words")

    def test_vods_hide_live_broadcasts(self, fake: FakeYtDlp) -> None:
        fake.responses["/videos?filter=archives"] = TWITCH_VODS
        page = twitch.channel("somestreamer", tab="vods", limit=10)
        assert fake.calls == [
            ("https://www.twitch.tv/somestreamer/videos?filter=archives&sort=time", True, 10)
        ]
        assert [i.id for i in page.items] == ["v111"]
        vod = page.items[0]
        assert vod.kind == "vod" and vod.source == "twitch"
        assert vod.duration_s == 14400.0 and vod.views == 9000
        assert vod.channel_url == "https://www.twitch.tv/somestreamer"

    def test_top_clips(self, fake: FakeYtDlp) -> None:
        fake.responses["/clips?filter=clips&range=7d"] = TWITCH_CLIPS
        page = twitch.channel("https://www.twitch.tv/somestreamer", tab="clips")
        clip = page.items[0]
        assert clip.kind == "clip"
        assert clip.url == "https://clips.twitch.tv/FunnyClipSlug"
        assert clip.upload_date is not None and clip.upload_date.startswith("2025-")

    def test_bad_channel_name_is_a_clear_error(self, fake: FakeYtDlp) -> None:
        with pytest.raises(ytdlp.ExploreError, match="not a Twitch channel"):
            twitch.channel("not a name!")
        assert fake.calls == []

    def test_search_without_credentials_reads_a_channel_name(
        self, fake: FakeYtDlp, no_twitch_creds: None
    ) -> None:
        result = service.search("Shroud", source="twitch")
        assert result.twitch_search is False
        assert [(i.kind, i.id) for i in result.items] == [("channel", "shroud")]
        assert "credentials" in (result.note or "")
        assert fake.calls == []

    def test_search_without_credentials_and_no_valid_name(self, no_twitch_creds: None) -> None:
        result = service.search("gta rp funny", source="twitch")
        assert result.items == [] and result.note

    def test_search_with_credentials_uses_helix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(twitch, "credentials", lambda: ("cid", "secret"))
        posts: list[str] = []
        gets: list[tuple[str, dict, dict]] = []

        def fake_post(url: str, params: dict[str, str]) -> dict:
            posts.append(url)
            return {"access_token": "tok", "expires_in": 3600}

        def fake_get(url: str, params: dict[str, str], headers: dict[str, str]) -> dict:
            gets.append((url, params, headers))
            return {
                "data": [
                    {
                        "broadcaster_login": "summit1g",
                        "display_name": "summit1g",
                        "id": "7",
                        "title": "GTA RP",
                        "thumbnail_url": "https://x/p.png",
                        "is_live": True,
                    },
                    {"display_name": "no login"},
                ]
            }

        items = twitch.search_channels(
            "summit", creds=("cid", "secret"), http_get=fake_get, http_post=fake_post
        )
        assert [i.id for i in items] == ["summit1g"]
        assert gets[0][2]["Authorization"] == "Bearer tok"
        assert gets[0][1]["query"] == "summit"
        # The app token is reused for the next search.
        twitch.search_channels("x", creds=("cid", "secret"), http_get=fake_get, http_post=fake_post)
        assert len(posts) == 1

    def test_helix_failure_is_explained(self) -> None:
        def boom(*_a: Any, **_k: Any) -> dict:
            raise RuntimeError("401 Unauthorized")

        with pytest.raises(ytdlp.ExploreError, match="Twitch channel search failed"):
            twitch.search_channels("x", creds=("a", "b"), http_get=boom, http_post=boom)


# --------------------------------------------------------------------------
# Peek
# --------------------------------------------------------------------------


def _heatmap(duration: float, hot: list[tuple[float, float]], *, bins: int = 100) -> list[dict]:
    step = duration / bins
    out = []
    for i in range(bins):
        start = i * step
        value = 0.15
        for a, b in hot:
            if a <= start < b:
                value = 1.0 if (a + b) / 2 - step <= start <= (a + b) / 2 + step else 0.8
        out.append({"start_time": start, "end_time": start + step, "value": value})
    return out


def _full_info(**extra: Any) -> dict[str, Any]:
    duration = 3000.0
    return {
        "id": "aaaaaaaaaaa",
        "title": "Big Interview",
        "channel": "Some Podcast",
        "channel_url": "https://www.youtube.com/channel/UCabc",
        "webpage_url": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
        "duration": duration,
        "view_count": 1_000_000,
        "like_count": 40_000,
        "upload_date": "20261020",
        "description": "x" * 5000,
        "thumbnail": "https://i.ytimg.com/vi/aaaaaaaaaaa/maxres.jpg",
        "live_status": "not_live",
        "chapters": [
            {"start_time": 0, "end_time": 600, "title": "Intro"},
            {"start_time": 600, "end_time": 3000, "title": "The story"},
        ],
        "heatmap": _heatmap(
            duration,
            [(300, 360), (900, 960), (1500, 1560), (2100, 2160), (2400, 2460), (2700, 2760)],
        ),
        **extra,
    }


class TestPeek:
    def test_heatmap_peaks_and_chapters(self) -> None:
        result = service.peek_from_info(_full_info(), "https://youtu.be/aaaaaaaaaaa")
        assert result.platform == "youtube"
        assert result.url == "https://www.youtube.com/watch?v=aaaaaaaaaaa"
        assert result.duration_s == 3000.0
        assert result.views == 1_000_000 and result.likes == 40_000
        assert result.upload_date == "2026-10-20"
        assert len(result.description) == service.DESCRIPTION_CHARS
        assert len(result.heatmap) == 100
        assert [c.title for c in result.chapters] == ["Intro", "The story"]
        # Six hot stretches in the graph, only the top five are kept,
        # strongest first, each inside one of the hot stretches.
        assert len(result.peaks) == 5
        values = [p.value for p in result.peaks]
        assert values == sorted(values, reverse=True)
        hot = [(300, 360), (900, 960), (1500, 1560), (2100, 2160), (2400, 2460), (2700, 2760)]
        for p in result.peaks:
            assert any(a - 30 <= p.peak_s <= b + 30 for a, b in hot), p
            assert p.start_s <= p.peak_s <= p.end_s
            assert p.reasons

    def test_no_heatmap_means_no_peaks(self) -> None:
        info = _full_info(heatmap=None)
        result = service.peek_from_info(info, "https://youtu.be/aaaaaaaaaaa")
        assert result.heatmap == [] and result.peaks == []
        assert result.chapters

    def test_live_and_twitch(self) -> None:
        info = _full_info(
            heatmap=None,
            live_status="is_live",
            webpage_url="https://www.twitch.tv/videos/111",
        )
        result = service.peek_from_info(info, "https://www.twitch.tv/videos/111")
        assert result.is_live and result.platform == "twitch"

    def test_playlist_is_rejected(self) -> None:
        with pytest.raises(ytdlp.ExploreError, match="channel or playlist"):
            service.peek_from_info({"_type": "playlist", "entries": []}, "https://x")

    def test_peek_runs_full_extraction_and_caches(self, fake: FakeYtDlp) -> None:
        fake.responses["watch?v=aaaaaaaaaaa"] = _full_info()
        url = "https://www.youtube.com/watch?v=aaaaaaaaaaa"
        first = service.peek(url)
        second = service.peek(url)
        assert first == second
        assert fake.calls == [(url, False, None)]

    def test_non_link_rejected(self, fake: FakeYtDlp) -> None:
        with pytest.raises(ytdlp.ExploreError):
            service.peek("just words")


# --------------------------------------------------------------------------
# Cache + errors
# --------------------------------------------------------------------------


class TestCache:
    def test_ttl_expiry(self) -> None:
        now = [0.0]
        c = cache_module.TTLCache(clock=lambda: now[0])
        calls: list[int] = []
        assert c.cached("k", lambda: calls.append(1) or "v", 10) == "v"
        assert c.cached("k", lambda: calls.append(1) or "w", 10) == "v"
        now[0] = 11
        assert c.cached("k", lambda: calls.append(1) or "w", 10) == "w"
        assert len(calls) == 2

    def test_bounded(self) -> None:
        c = cache_module.TTLCache(max_entries=2)
        for k in "abc":
            c.put(k, k, 60)
        assert len(c) == 2 and c.get("a") is None and c.get("c") == "c"

    def test_failures_are_not_cached(self, fake: FakeYtDlp) -> None:
        fake.responses["ytsearch"] = ytdlp.ExploreError("blocked")
        with pytest.raises(ytdlp.ExploreError):
            service.search("podcast")
        fake.responses["ytsearch"] = SEARCH_RESULTS
        assert service.search("podcast").items
        assert len(fake.calls) == 2

    def test_search_results_are_cached(self, fake: FakeYtDlp) -> None:
        fake.responses["ytsearch"] = SEARCH_RESULTS
        service.search("podcast")
        service.search("  podcast ")
        assert len(fake.calls) == 1


class TestErrorTranslation:
    def test_bot_check(self) -> None:
        err = ytdlp.translate_error(RuntimeError("Sign in to confirm you're not a bot"))
        assert "bot" in str(err) and "cookies" in err.hint

    def test_rate_limited(self) -> None:
        err = ytdlp.translate_error(RuntimeError("HTTP Error 429: Too Many Requests"))
        assert "refused" in str(err)

    def test_generic(self) -> None:
        err = ytdlp.translate_error(RuntimeError("weird extractor failure"))
        assert "update-ytdlp" in err.hint and "weird extractor failure" in err.hint


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


def _item(i: int) -> ExploreItem:
    return ExploreItem(
        kind="video",
        source="youtube",
        id=f"id{i}",
        url=f"https://www.youtube.com/watch?v=id{i}",
        title=f"T{i}",
        duration_s=100.0,
    )


class TestApi:
    def test_search_passes_filters(self, client: TestClient, monkeypatch) -> None:
        seen: dict[str, Any] = {}

        def fake_search(q: str, **kw: Any) -> SearchResult:
            seen.update(q=q, **kw)
            return SearchResult(items=[_item(1)], twitch_search=False)

        monkeypatch.setattr(explore, "search", fake_search)
        r = client.get(
            "/api/explore/search",
            params={
                "q": "podcast",
                "type": "shorts",
                "duration": "long",
                "sort": "views",
                "uploaded": "week",
                "limit": 12,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["items"][0]["id"] == "id1"
        assert body["items"][0]["duration_s"] == 100.0
        assert seen == {
            "q": "podcast",
            "source": "youtube",
            "type": "shorts",
            "duration": "long",
            "sort": "views",
            "uploaded": "week",
            "limit": 12,
        }

    def test_search_validates(self, client: TestClient) -> None:
        assert (
            client.get("/api/explore/search", params={"q": "x", "type": "nope"}).status_code == 422
        )
        assert client.get("/api/explore/search", params={"q": "   "}).status_code == 400
        assert client.get("/api/explore/search").status_code == 422

    def test_errors_carry_message_and_hint(self, client: TestClient, monkeypatch) -> None:
        def boom(*_a: Any, **_k: Any) -> Any:
            raise explore.ExploreError("YouTube asked for proof you're not a bot.", hint="cookies")

        monkeypatch.setattr(explore, "search", boom)
        r = client.get("/api/explore/search", params={"q": "x"})
        assert r.status_code == 502
        assert r.json()["detail"] == {
            "message": "YouTube asked for proof you're not a bot.",
            "hint": "cookies",
        }

    def test_channel_route(self, client: TestClient, fake: FakeYtDlp) -> None:
        fake.responses["/videos?filter=archives"] = TWITCH_VODS
        r = client.get(
            "/api/explore/channel",
            params={"url": "https://www.twitch.tv/somestreamer", "tab": "vods"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["channel"]["source"] == "twitch"
        assert [i["id"] for i in body["items"]] == ["v111"]

    def test_peek_route(self, client: TestClient, fake: FakeYtDlp) -> None:
        fake.responses["watch?v=aaaaaaaaaaa"] = _full_info()
        r = client.get(
            "/api/explore/peek", params={"url": "https://www.youtube.com/watch?v=aaaaaaaaaaa"}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["peaks"]) == 5
        assert body["heatmap"][0].keys() == {"start_s", "end_s", "value"}
        assert client.get("/api/explore/peek", params={"url": "nope"}).status_code == 400

    def test_status(self, client: TestClient, no_twitch_creds: None) -> None:
        r = client.get("/api/explore/status")
        assert r.status_code == 200 and r.json() == {"twitch_search": False}


class TestRound2:
    def test_shorts_use_the_real_filter_when_hashtag_search_is_empty(self, fake: FakeYtDlp) -> None:
        fake.responses["%23shorts"] = {"_type": "playlist", "entries": []}
        fake.responses["youtube.com/results"] = {
            "_type": "playlist",
            "entries": [
                _video(
                    "ggggggggggg", url="https://www.youtube.com/shorts/ggggggggggg", duration=None
                )
            ],
        }
        items = youtube.search("cats", type="shorts")
        assert [(i.id, i.kind) for i in items] == [("ggggggggggg", "short")]
        assert "sp=EgIQCQ" in fake.calls[-1][0]

    def test_missing_channel_tab_is_an_empty_tab(self, fake: FakeYtDlp) -> None:
        fake.responses["/streams"] = ytdlp.translate_error(
            RuntimeError("ERROR: [youtube:tab] @x: This channel does not have a streams tab")
        )
        fake.responses["/videos"] = TestYouTubeChannel.CHANNEL
        page = youtube.channel("@x", tab="streams")
        assert page.items == [] and page.tab == "streams"
        assert page.channel.title == "Some Podcast"

    def test_one_dominant_moment_still_gets_five_spaced_peaks(self) -> None:
        import numpy as np
        from autoclip.pipeline.signals import build_curve

        duration = 1000.0
        heat = []
        for i in range(100):
            start = i * 10.0
            # One huge hill and four small bumps.
            value = 1.0 if 500 <= start < 520 else 0.45 if i in (10, 30, 70, 90) else 0.1
            heat.append((start, start + 10.0, value))
        curve = build_curve(duration, heatmap=heat)
        assert len(curve.hotspots) < 5  # the strict detector alone finds fewer
        peaks = service.top_peaks(curve.curve, curve.hotspots)
        assert len(peaks) == 5
        assert peaks[0].value == max(p.value for p in peaks)
        centres = sorted(p.peak_s for p in peaks)
        assert all(b - a >= 10 for a, b in zip(centres, centres[1:], strict=False))
        assert np.isclose(peaks[0].peak_s, 510, atol=15)

    def test_flat_curve_has_no_extra_peaks(self) -> None:
        assert service.top_peaks([0.0] * 100, []) == []

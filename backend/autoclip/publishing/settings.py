"""Settings for posting normal-mode clips (a JSON file, not config.json).

The YouTube / Zernio connections themselves are shared with GTA Mode (OS
keyring, set up in GTA Mode's Posting tab); these are only the defaults for
how a normal clip is posted.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from ..gta.settings import YOUTUBE_CATEGORIES
from .jsonstore import JsonFile

_FILE = JsonFile("publishing", "settings.json", dict)


class PublishSettings(BaseModel):
    #: YouTube category for normal clips ("24" = Entertainment).
    category_id: str = "24"
    #: Append #Shorts to the description when the clip's hashtags lack it.
    shorts_tag: bool = True
    #: A scheduled post is sent to the platform this many minutes before its
    #: time (until then it can be cancelled with no trace anywhere).
    lead_minutes: int = 120
    #: Instagram username to post as when several are connected at Zernio
    #: ("" = GTA Mode's setting, else the only connected account).
    instagram_account: str = ""

    model_config = {"extra": "ignore"}

    @field_validator("category_id")
    @classmethod
    def _category(cls, value: str) -> str:
        if value not in YOUTUBE_CATEGORIES:
            raise ValueError(f"category_id must be one of {', '.join(YOUTUBE_CATEGORIES)}")
        return value

    @field_validator("lead_minutes")
    @classmethod
    def _lead(cls, value: int) -> int:
        if not 15 <= value <= 1440:
            raise ValueError("lead_minutes must be between 15 and 1440")
        return value

    @field_validator("instagram_account")
    @classmethod
    def _account(cls, value: str) -> str:
        text = str(value).strip().lstrip("@").strip()
        if len(text) > 64 or any(ch.isspace() for ch in text):
            raise ValueError("Instagram account must be a username like clipgod (no spaces)")
        return text


def load() -> PublishSettings:
    try:
        return PublishSettings.model_validate(_FILE.read())
    except ValueError:
        return PublishSettings()


def save(settings: PublishSettings) -> PublishSettings:
    with _FILE.edit() as data:
        data.clear()
        data.update(settings.model_dump(mode="json"))
    return settings


def instagram_account(settings: PublishSettings | None = None) -> str:
    """The account hint for Zernio: ours, else GTA Mode's."""
    settings = settings or load()
    if settings.instagram_account:
        return settings.instagram_account
    try:
        from ..gta.settings import load as load_gta

        return load_gta().posting.instagram_account
    except Exception:
        return ""

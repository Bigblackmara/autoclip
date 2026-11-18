"""Autopilot rules (what to watch, which videos qualify, when to post)."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_TIME = re.compile(r"^(\d{1,2}):(\d{1,2})$")

#: A country's main language, used when a topic source names only a country.
COUNTRY_LANGUAGE = {
    "US": "en",
    "GB": "en",
    "CA": "en",
    "AU": "en",
    "IN": "hi",
    "DE": "de",
    "AT": "de",
    "FR": "fr",
    "ES": "es",
    "MX": "es",
    "AR": "es",
    "BR": "pt",
    "PT": "pt",
    "IT": "it",
    "NL": "nl",
    "PL": "pl",
    "TR": "tr",
    "RU": "ru",
    "JP": "ja",
    "KR": "ko",
    "ID": "id",
    "PH": "en",
    "SA": "ar",
    "EG": "ar",
}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class AutopilotSource(BaseModel):
    """A YouTube channel (link / @handle / UC… id) or a topic search."""

    kind: Literal["channel", "topic"]
    value: str = Field(min_length=1, max_length=300)
    #: Topic only (optional): ISO country code, e.g. "US".
    country: str = ""
    #: Optional: only videos in this language (ISO 639-1, e.g. "en").
    language: str = ""

    @field_validator("value")
    @classmethod
    def _value(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Give a channel link/handle or a topic")
        return text

    @field_validator("country")
    @classmethod
    def _country(cls, value: str) -> str:
        text = value.strip().upper()
        if text and not re.fullmatch(r"[A-Z]{2}", text):
            raise ValueError("Country must be a 2-letter code like US")
        return text

    @field_validator("language")
    @classmethod
    def _language(cls, value: str) -> str:
        text = value.strip().lower()
        if text and not re.fullmatch(r"[a-z]{2,3}", text):
            raise ValueError("Language must be a code like en")
        return text

    @property
    def wanted_language(self) -> str:
        return self.language or COUNTRY_LANGUAGE.get(self.country, "")

    @property
    def label(self) -> str:
        if self.kind == "channel":
            return self.value
        extra = " ".join(x for x in (self.country, self.language) if x)
        return f"“{self.value}”" + (f" ({extra})" if extra else "")


class AutopilotFilters(BaseModel):
    min_views: int = Field(default=10_000, ge=0, le=10_000_000_000)
    min_minutes: float = Field(default=5, ge=0, le=600)
    max_minutes: float = Field(default=180, ge=1, le=900)
    max_age_days: int = Field(default=14, ge=1, le=365)

    @model_validator(mode="after")
    def _range(self) -> AutopilotFilters:
        if self.min_minutes >= self.max_minutes:
            raise ValueError("Minimum length must be below the maximum")
        return self


class AutopilotIn(BaseModel):
    """What the editor sends (create and update)."""

    name: str = Field(min_length=1, max_length=80)
    sources: list[AutopilotSource] = Field(min_length=1, max_length=20)
    filters: AutopilotFilters = Field(default_factory=AutopilotFilters)
    clips_per_video: int = Field(default=2, ge=1, le=5)
    #: Daily posting times, "HH:MM" (24 h) in ``timezone``.
    slots: list[str] = Field(default_factory=lambda: ["12:00", "18:00"])
    #: IANA zone like "Europe/Berlin"; "" = this PC's local time.
    timezone: str = ""
    platforms: list[Literal["youtube", "instagram"]] = Field(default_factory=lambda: ["youtube"])
    #: Clips wait in "Review" until approved, instead of being booked at once.
    approve_before_posting: bool = True
    #: How long a new video may wait for YouTube's "Most replayed" graph;
    #: after that it is clipped anyway (the AI uses audio + comments).
    heatmap_wait_hours: float = Field(default=24, ge=0, le=240)
    #: The user confirms they have the rights to clip and repost these videos.
    rights_confirmed: bool = False

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Give the autopilot a name")
        return text

    @field_validator("slots")
    @classmethod
    def _slots(cls, value: list[str]) -> list[str]:
        cleaned: set[str] = set()
        for raw in value:
            match = _TIME.match(str(raw).strip())
            if not match:
                raise ValueError(f"Posting time {raw!r} is not a time like 17:00")
            hour, minute = int(match.group(1)), int(match.group(2))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError(f"Posting time {raw!r} is not a valid time of day")
            cleaned.add(f"{hour:02d}:{minute:02d}")
        if not 1 <= len(cleaned) <= 12:
            raise ValueError("Give between 1 and 12 posting times")
        return sorted(cleaned)

    @field_validator("timezone")
    @classmethod
    def _zone(cls, value: str) -> str:
        text = value.strip()
        if text:
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

            try:
                ZoneInfo(text)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError(f"Unknown time zone {text!r}") from exc
        return text

    @field_validator("platforms")
    @classmethod
    def _platforms(cls, value: list[str]) -> list[str]:
        out = list(dict.fromkeys(value))
        if not out:
            raise ValueError("Pick YouTube, Instagram or both")
        return out


class Autopilot(AutopilotIn):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    #: False = paused (no scanning, clipping or booking; booked posts still go out).
    enabled: bool = True
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)

    model_config = {"extra": "ignore"}

    def tzinfo(self):
        """The zone slots are read in (None = this PC's local time)."""
        if not self.timezone:
            return None
        from zoneinfo import ZoneInfo

        return ZoneInfo(self.timezone)

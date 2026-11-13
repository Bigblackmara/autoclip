"""Pydantic wire models for the Live Watch REST API (A10).

These mirror ``frontend/src/live/api.ts`` field-for-field — the frontend
types are the frozen contract, so every field name and nullability here is
chosen to match them exactly. Row dataclasses (``live.models``) stay the
storage shape; these are the API boundary, the same split as
``api/schemas.py`` vs ``db/models.py``.

Deviations from api.ts: none. The only additive fields are ``capturing`` /
``detail`` on LiveStreamOut (F1's poll backfills via its own list call) —
both optional with safe defaults so the frontend shape still validates.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..live import models
from ..live.settings import LiveSettings

LivePlatform = Literal["youtube", "instagram"]


# --------------------------------------------------------------------------
# Streams
# --------------------------------------------------------------------------


class LiveStreamOut(BaseModel):
    """One watched stream, exactly the ``LiveStream`` interface in api.ts."""

    id: str
    url: str
    label: str
    enabled: bool
    state: str
    state_detail: str
    last_error: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def of(cls, row: models.LiveStreamRow) -> LiveStreamOut:
        return cls(
            id=row.id,
            url=row.url,
            label=row.label,
            enabled=row.enabled,
            state=row.state if isinstance(row.state, str) else str(row.state),
            state_detail=row.state_detail or "",
            last_error=row.last_error,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @classmethod
    def of_snapshot(cls, snapshot: dict) -> LiveStreamOut:
        """Build from ``manager.stream_snapshot()`` — the API-facing view."""
        return cls(
            id=snapshot["id"],
            url=snapshot["url"],
            label=snapshot.get("label") or "",
            enabled=bool(snapshot.get("enabled", True)),
            state=snapshot.get("state") or "waiting_for_live",
            state_detail=snapshot.get("detail") or "",
            last_error=snapshot.get("error"),
            created_at=snapshot.get("created_at") or "",
            updated_at=snapshot.get("updated_at") or "",
        )


class LiveStreamIn(BaseModel):
    """POST /api/live/streams body."""

    url: str
    label: str = ""


# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------


class LiveClipOut(BaseModel):
    """One live clip, exactly the ``LiveClip`` interface in api.ts."""

    id: str
    stream_id: str
    status: str
    title: str
    score: int
    reason: str
    hook: str
    trigger_start_s: float
    trigger_end_s: float
    clip_start_s: float
    clip_end_s: float
    caption_style: str
    ratio: str
    error: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def of(cls, row: models.LiveClipRow) -> LiveClipOut:
        return cls(
            id=row.id,
            stream_id=row.stream_id,
            status=row.status,
            title=row.title or "",
            score=int(row.score or 0),
            reason=row.reason or "",
            hook=row.hook or "",
            trigger_start_s=float(row.trigger_start_s or 0.0),
            trigger_end_s=float(row.trigger_end_s or 0.0),
            clip_start_s=float(row.clip_start_s or 0.0),
            clip_end_s=float(row.clip_end_s or 0.0),
            caption_style=row.caption_style or "bold_pop",
            ratio=row.ratio or "9:16",
            error=row.error,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


# --------------------------------------------------------------------------
# Uploads
# --------------------------------------------------------------------------


class LiveUploadOut(BaseModel):
    """One upload attempt, exactly the ``LiveUpload`` interface in api.ts."""

    id: str
    clip_id: str
    platform: LivePlatform
    status: str
    upload_url: str
    platform_media_id: str
    error: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def of(cls, row: models.LiveUploadRow) -> LiveUploadOut:
        return cls(
            id=row.id,
            clip_id=row.clip_id,
            platform=row.platform,  # type: ignore[arg-type]
            status=row.status,
            upload_url=row.upload_url or "",
            platform_media_id=row.platform_media_id or "",
            error=row.error,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class LiveUploadIn(BaseModel):
    """POST /api/live/clips/{id}/upload body."""

    platform: LivePlatform


class LiveCredentialsIn(BaseModel):
    """POST /api/live/uploads/{platform}/credentials body.

    The secret is handed to ``save_client_credentials`` and NEVER echoed
    back — the response is only ``{"saved": true}``.
    """

    client_id: str
    client_secret: str


class LiveAttestationIn(BaseModel):
    """POST /api/live/uploads/attestation body."""

    accepted: bool


# --------------------------------------------------------------------------
# Status + settings
# --------------------------------------------------------------------------


class LiveCapsOut(BaseModel):
    caps: dict[str, float | int] = Field(default_factory=dict)


class LiveStatusOut(BaseModel):
    """``LiveStatus`` in api.ts: enabled, watching, clipping, caps."""

    enabled: bool
    watching: int
    clipping: int
    caps: dict[str, float | int]


# --------------------------------------------------------------------------
# SSE payload models — the data payloads, not the SSE envelope
# --------------------------------------------------------------------------


class LiveEventOut(BaseModel):
    """``LiveEvent`` in api.ts: the SSE frame the frontend parses."""

    type: str
    stream_id: str | None = None
    clip_id: str | None = None
    data: dict | None = None


# --------------------------------------------------------------------------
# Settings — mirror LiveSettingsData field-for-field
# --------------------------------------------------------------------------


class LiveSettingsOut(BaseModel):
    """Wire shape = LiveSettings.model_dump(); api.ts ``LiveSettingsData``.

    The underlying pydantic model already matches the frontend interface
    section-for-section, so the dump IS the wire shape — no re-mapping. The
    wrapper class exists so response_model validation and schema docs have
    a concrete model, and so a future field rename in the settings tree
    shows up here at the API boundary instead of shipping silently.
    """

    model_config = {"extra": "allow"}

    enabled: bool
    rolling_buffer: dict
    rate_limit: dict
    detection: dict
    uploads: dict
    workers: dict

    @classmethod
    def of(cls, settings: LiveSettings) -> LiveSettingsOut:
        return cls(**settings.model_dump(mode="json"))


class LiveSettingsIn(BaseModel):
    """PUT /api/live/settings body: full LiveSettings tree.

    The frontend sends the merged whole object (F3 comment), and the
    backend replaces the stored section wholesale — validated through
    ``LiveSettings.model_validate`` so an invalid section is a 422 at the
    boundary, not a corrupt config.json.
    """

    model_config = {"extra": "ignore"}

    @classmethod
    def validated(cls, raw: dict) -> LiveSettings:
        return LiveSettings.model_validate(raw)

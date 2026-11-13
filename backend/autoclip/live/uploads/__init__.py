"""Upload subsystem for Live Watch (A7 owns bodies).

youtube.py    — YouTube Data API v3 resumable uploads (official API, httpx only)
instagram.py  — Instagram Graph API resumable Reels uploads (official API)
tokens.py     — OS-keyring OAuth token/client store, code exchange, refresh

Hard rules (from the task spec): official APIs only, no browser automation,
tokens and client credentials in the OS keyring (service ``autoclip``), never
in config files and never in logs; a rights-attestation flag
(``live_rights_attestation``, written by the UI's acceptance endpoint) gates
every upload call.

Public surface (manager + REST code against these):

    await upload_youtube(path, meta)     -> UploadResult
    await upload_instagram(path, meta)   -> UploadResult
    await youtube_status()               -> {"connected": bool, "account": str | None}
    await instagram_status()             -> {"connected": bool, "account": str | None}

plus the OAuth helpers in :mod:`.tokens` the REST layer wires:
``save_client_credentials`` / ``authorization_url`` / ``exchange_code`` /
``accept_rights`` and friends. Re-exported here for one-line imports by the
manager; the modules above own the bodies.
"""

from __future__ import annotations

from .instagram import instagram_status, upload_instagram
from .tokens import (
    ATTESTATION_KEY,
    RightsNotAccepted,
    UploadError,
    UploadMeta,
    UploadResult,
    accept_rights,
    authorization_url,
    clear_rights,
    clear_tokens,
    exchange_code,
    has_rights_attestation,
    load_client_credentials,
    load_tokens,
    refresh_tokens,
    require_rights_attestation,
    rights_attestation,
    save_client_credentials,
    save_tokens,
)
from .youtube import upload_youtube, youtube_status

__all__ = [
    "ATTESTATION_KEY",
    "RightsNotAccepted",
    "UploadError",
    "UploadMeta",
    "UploadResult",
    "accept_rights",
    "authorization_url",
    "clear_rights",
    "clear_tokens",
    "exchange_code",
    "has_rights_attestation",
    "instagram_status",
    "load_client_credentials",
    "load_tokens",
    "refresh_tokens",
    "require_rights_attestation",
    "rights_attestation",
    "save_client_credentials",
    "save_tokens",
    "upload_instagram",
    "upload_youtube",
    "youtube_status",
]

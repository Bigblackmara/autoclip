"""Run one YouTube download in a fresh interpreter.

Used by :func:`autoclip.pipeline.ingest.download_in_subprocess` after yt-dlp
was upgraded on disk while AutoClip was running: this process imports yt_dlp
anew, so the upgraded version is the one that downloads.

Protocol: a JSON object ``{url, target_dir, settings}`` on stdin. Progress is
printed as ``AUTOCLIP-PROGRESS <0..1>`` lines and the outcome as one
``AUTOCLIP-RESULT <json>`` line, ``{"ok": true, "metadata": {...}}`` or
``{"ok": false, "message": ..., "download_error": bool}``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    from autoclip.config import IngestSettings
    from autoclip.pipeline import ingest

    payload = json.loads(sys.stdin.read())
    settings = IngestSettings.model_validate(payload["settings"])

    def progress(fraction: float) -> None:
        print(f"{ingest.WORKER_PROGRESS}{fraction:.4f}", flush=True)

    try:
        metadata = ingest.download_in_process(
            payload["url"], settings, Path(payload["target_dir"]), progress
        )
    except ingest.YtdlpFailure as exc:
        result = {"ok": False, "message": str(exc), "download_error": exc.download_error}
        print(ingest.WORKER_RESULT + json.dumps(result), flush=True)
        return 1
    print(ingest.WORKER_RESULT + json.dumps({"ok": True, "metadata": metadata}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Skeletons for the Live Watch REST API routers (A10 owns real bodies).

Will be mounted under /api/live with these endpoints (contract frozen for
the frontend agents; bodies land with A10):

    GET  /api/live/streams                 list watched streams + live state
    POST /api/live/streams                 {url, label?} → add
    DELETE /api/live/streams/{id}          remove (stops capture, frees buffer)
    POST /api/live/streams/{id}/pause      pause
    POST /api/live/streams/{id}/resume     resume
    GET  /api/live/streams/{id}/events     SSE: state + clip events for stream
    GET  /api/live/clips                   list live clips (all streams)
    GET  /api/live/clips/{id}              one clip
    GET  /api/live/clips/{id}/media        the rendered mp4 (FileResponse)
    GET  /api/live/events                  SSE: global live feed (states+clips)
    GET  /api/live/uploads                 list upload records
    POST /api/live/clips/{id}/upload       {platform} manual upload trigger
    POST /api/live/clips/{id}/discard      drop a clip without rendering
    GET  /api/live/status                  {enabled, watching, clipping, caps}
    GET  /api/live/settings                LiveSettings (read)
    PUT  /api/live/settings                LiveSettings (write, validated)
    POST /api/live/uploads/youtube/start   OAuth start (returns redirect URL)
    POST /api/live/uploads/instagram/start OAuth start (returns redirect URL)
    GET  /api/live/uploads/youtube/status   token present? account ok?
    GET  /api/live/uploads/instagram/status token present? account ok?
"""

from __future__ import annotations


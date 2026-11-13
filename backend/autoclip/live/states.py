"""Per-stream state machine for Live Watch.

One module of truth for watcher states, so the manager, API, CLI, and UI all
report the same vocabulary. States and transitions::

    waiting_for_live → connecting → watching → clipping
                                    ↓
                              reconnecting → watching (or ended)
    any state → ended (user removed / channel ended permanently)
    any state → error (terminal failure after exhausting retries)

``clipping`` means "a candidate from this stream is being turned into a clip";
the capture loop keeps running underneath it — a live moment shouldn't pause the
buffer just because we're cutting a clip out of it.
"""

from __future__ import annotations

from enum import StrEnum


class StreamState(StrEnum):
    OFF = "off"  # paused by the user; no task running
    WAITING_FOR_LIVE = "waiting_for_live"
    CONNECTING = "connecting"
    WATCHING = "watching"
    CLIPPING = "clipping"
    RECONNECTING = "reconnecting"
    ENDED = "ended"
    ERROR = "error"


#: Non-terminal states: a watcher task is alive or paused-but-resumable.
ACTIVE_STATES: tuple[StreamState, ...] = (
    StreamState.WAITING_FOR_LIVE,
    StreamState.CONNECTING,
    StreamState.WATCHING,
    StreamState.CLIPPING,
    StreamState.RECONNECTING,
)

#: Terminal states: the watcher task has fully stopped.
TERMINAL_STATES: tuple[StreamState, ...] = (StreamState.ENDED, StreamState.ERROR)


def is_terminal(state: StreamState) -> bool:
    """Whether ``state`` means the watcher task has fully stopped."""
    return state in TERMINAL_STATES


def can_transition(old: StreamState, new: StreamState) -> bool:
    """Whether a watcher may move directly from ``old`` to ``new``.

    Enforced by the watcher's ``set_state``: an illegal jump (e.g. a capture
    callback firing after ``ended``) is logged and ignored rather than
    corrupting the state the UI shows.
    """
    if old == new:
        return True
    if new == StreamState.ERROR:
        return True
    if old == StreamState.OFF or old in TERMINAL_STATES:
        # Only user actions (resume) can leave off/ended/error — and resume is
        # modelled as a fresh watcher start, not an in-place transition.
        return False
    if new == StreamState.OFF:
        return True  # pause
    if new == StreamState.ENDED:
        return True  # user removed, or the stream ended permanently
    # Everything else must follow the forward path.
    order = [
        StreamState.WAITING_FOR_LIVE,
        StreamState.CONNECTING,
        StreamState.WATCHING,
    ]
    if old == StreamState.CLIPPING:
        return new in (StreamState.WATCHING, StreamState.RECONNECTING)
    if old == StreamState.RECONNECTING:
        return new in (
            StreamState.CONNECTING,
            StreamState.WATCHING,
            StreamState.WAITING_FOR_LIVE,
        )
    if new == StreamState.CLIPPING:
        return old == StreamState.WATCHING
    if new == StreamState.RECONNECTING:
        return True  # from connecting or watching
    # Forward edges: waiting → connecting → watching.
    return new in order

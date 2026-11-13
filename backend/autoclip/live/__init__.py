"""Live Watch — one module per concern, wired together by the manager.

Layout (each file owned by exactly one worker; see docs/LIVE_WATCH.md)::

    watcher.py      per-stream supervisor task: state machine + reconnect loop
    manager.py      LiveManager: the single LiveWatch entry point
    registry.py     persistence for watched streams (config.json, live section)
    capture.py      streamlink/yt-dlp → ffmpeg → rolling .ts buffer + signals
    buffer.py       RingBuffer: segment list, range extraction, pruning
    detector.py      SignalAggregator + Trigger decision logic
    audio_signals.py audio energy/RMS spike detection
    chat_signals.py  chat readers (Twitch IRC only, v1)
    judge.py         LLM gate — score a candidate window with one provider call
    clip_factory.py  buffer range → finished clip, reusing pipeline stages
    circuit.py       per-stream cooldown, enforced before any candidate run
    amp.py           adaptive amplification (optional post-processing)
    uploads/         YouTube + Instagram Graph API uploaders, token store
"""

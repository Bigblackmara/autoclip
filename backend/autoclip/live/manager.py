"""LiveManager — the single Live Watch entry point (A8 owns bodies).

Owns: watcher task registry, add/remove/pause/resume, the bounded clip queue,
detector loop wiring, judge rate limits (via circuit.py), event publishing.
"""

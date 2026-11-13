"""Audio energy spike detector (skeleton — A2 owns bodies).

Runs on the rolling buffer's audio, computes per-second RMS levels, flags
windows whose energy z-score (vs a rolling baseline) exceeds live settings.
"""

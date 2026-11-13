"""Trigger decision logic for viral-moment detection (skeleton — A4 owns bodies).

Cheap signals run continuously; the aggregator rolls them up and decides when a
window has enough combined evidence to spend an LLM call on.
"""

from __future__ import annotations

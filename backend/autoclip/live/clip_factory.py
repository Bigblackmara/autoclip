"""Clip production: buffer range → finished clip (skeleton — A8b owns bodies).

This is the Live Watch equivalent of pipeline/runner.py but for live windows:
it reuses prepare/transcribe/reframe/captions/export stages and runs them
against a buffer extraction, NOT a job row.
"""

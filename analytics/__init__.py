"""Persistence and, from week 4, the gap report.

The harness never writes to Postgres itself. It emits events and finished
turns; this package persists them. That separation is what lets harness unit
tests run in milliseconds with no database.
"""

"""PhysioQC - Headless CLI for real-time Signal Quality + auto-event logging on physiological signals."""

__version__ = "0.1.0"
__author__ = "PhysioQC Team"

from . import cli, sqi, simulate, ingest, annotate

__all__ = ["cli", "sqi", "simulate", "ingest", "annotate"]

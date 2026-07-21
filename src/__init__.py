"""Shared package for the F-DATA/PM100 HPC prediction thesis.

Reusable logic lives here, not in notebooks (see Track B6 of the
conceptualization plan) — notebooks import from this package and stay
thin orchestration/reporting layers so F-DATA and PM100 pipelines can't
silently drift out of sync from copy-pasted code.
"""

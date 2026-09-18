"""Revision stamp for result sidecars.

The working copy that runs the experiments is not always a git checkout (the
release lives in a separate git checkout, and the server copy is a plain
directory), so ``git rev-parse`` is tried first and a ``REVISION`` file at the
repository root second. That file holds the short hash of the checkout commit
the copy was taken from and is written by ``scripts/write_revision.sh`` after
each commit; it is ignored by git so it never goes stale inside the checkout
itself. Its value is suffixed with ``+file`` so a sidecar shows which route
produced it. Every script that writes a ``.meta.json`` sidecar records this
value under ``git_revision``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def git_revision(root: Path | None = None) -> str:
    """Short git hash of ``root`` (default: the repository), or ``<hash>+file``
    from its ``REVISION`` file, or ``"unknown"``."""
    root = Path(root) if root is not None else _REPO_ROOT
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    rev = root / "REVISION"
    if rev.exists():
        token = rev.read_text().split()
        if token:
            return token[0] + "+file"
    return "unknown"

"""Shared artifact utilities for BasketTube pipeline stages.

Provides:
- config_key: deterministic short hash from pipeline params
- artifact_path: canonical output path for a pipeline stage result
- status_path_for: sidecar status file path alongside an artifact
- atomic_write_json: write JSON atomically (tempfile + os.replace)
- write_status: write an artifact status sidecar ("active", "complete", "error")
- read_status: read sidecar, returning {"status": "pending"} if missing/corrupt
- check_stale: crash recovery — reset stale "active" sidecars to "pending"
"""

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path


def config_key(params: dict) -> str:
    """Return a deterministic short hash for *params*.

    JSON-serialises the dict with sorted keys, SHA-256 hashes the result, and
    returns ``"c-"`` followed by the first 7 hex characters of the digest.
    """
    serialised = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialised.encode()).hexdigest()
    return "c-" + digest[:7]


def artifact_path(data_dir: Path, stage: str, cfg_key: str, stem: str) -> Path:
    """Return the canonical path for a pipeline artifact.

    Layout: ``data_dir / "analysis" / stage / cfg_key / "{stem}.{ext}"``

    The extension is ``.mp4`` for the ``"renders"`` stage; ``.json`` otherwise.
    """
    ext = "mp4" if stage == "renders" else "json"
    return data_dir / "analysis" / stage / cfg_key / f"{stem}.{ext}"


def status_path_for(artifact: Path) -> Path:
    """Return the sidecar status path for *artifact*.

    E.g. ``video1.json`` → ``video1.status.json``.
    """
    return artifact.with_suffix(".status.json")


def atomic_write_json(path: Path, data: dict) -> None:
    """Write *data* as JSON to *path* atomically.

    Creates all parent directories. Writes to a sibling temp file, then calls
    ``os.replace`` so the operation is atomic on POSIX systems. Cleans up the
    temp file on failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_status(
    sidecar: Path,
    status: str,
    *,
    config_key: str | None = None,
    error: str | None = None,
) -> None:
    """Write a status sidecar atomically.

    Recognised *status* values:

    - ``"active"``   — sets ``started_at`` to the current epoch timestamp.
    - ``"complete"`` — preserves ``started_at`` from the existing sidecar (if
      any), sets ``completed_at``, and computes ``duration_ms``.
    - ``"error"``    — records the ``error`` message.

    Optional *config_key* is stored verbatim when provided.
    """
    now = time.time()
    payload: dict = {"status": status}

    if config_key is not None:
        payload["config_key"] = config_key

    if status == "active":
        payload["started_at"] = now
    elif status == "complete":
        # Try to read existing sidecar to preserve started_at for duration calc.
        existing = read_status(sidecar)
        started_at = existing.get("started_at")
        if started_at is not None:
            payload["started_at"] = started_at
            payload["duration_ms"] = (now - started_at) * 1000.0
        else:
            payload["duration_ms"] = None
        payload["completed_at"] = now
        # Preserve config_key from existing sidecar if not explicitly provided.
        if config_key is None and "config_key" in existing:
            payload["config_key"] = existing["config_key"]
    elif status == "error":
        if error is not None:
            payload["error"] = error

    atomic_write_json(sidecar, payload)


def read_status(sidecar: Path) -> dict:
    """Read and return the sidecar JSON.

    Returns ``{"status": "pending"}`` if the file is missing or its contents
    cannot be parsed as JSON.
    """
    try:
        return json.loads(sidecar.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "pending"}


def write_resolved_config(
    output_dir: Path, stage: str, config_key: str, params: dict, upstream: dict,
) -> None:
    """Write a frozen config snapshot alongside an artifact directory."""
    from datetime import datetime, timezone
    data = {
        "config_key": config_key, "stage": stage, "params": params,
        "upstream": upstream, "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_dir / "config.resolved.json", data)


def check_stale(sidecar: Path, timeout_s: float = 600.0) -> dict:
    """Crash-recovery helper.

    Reads the sidecar status.  If the status is ``"active"`` and ``started_at``
    is older than *timeout_s* seconds ago, the sidecar is deleted and
    ``{"status": "pending"}`` is returned.  In all other cases the status dict
    is returned unchanged.
    """
    current = read_status(sidecar)
    if current.get("status") != "active":
        return current

    started_at = current.get("started_at")
    if started_at is not None and (time.time() - started_at) > timeout_s:
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass
        return {"status": "pending"}

    return current

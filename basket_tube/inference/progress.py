"""Atomic progress file writer for GPU inference stages.

The API orchestrator polls ``_progress.json`` inside each stage's output
directory and streams the result to the browser via SSE.  Writes are atomic
(tmp + os.replace) so the reader never sees a partial file.
"""

import json
import os
import tempfile
import time
from pathlib import Path


def write_progress(output_dir: Path, frame: int, total_frames: int) -> None:
    """Write progress atomically (tmp + rename) for API orchestrator polling."""
    progress_path = output_dir / "_progress.json"
    data = {"frame": frame, "total_frames": total_frames, "updated_at": time.time()}
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, str(progress_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

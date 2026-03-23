import json
from pathlib import Path
from api.src.artifacts import artifact_path, delete_artifact, write_status, status_path_for


def test_delete_artifact_removes_all_files(tmp_path):
    data_dir = tmp_path
    stage, cfg_key, stem = "detections", "c-abc1234", "test_video"

    art = artifact_path(data_dir, stage, cfg_key, stem)
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps({"n_frames": 10}))

    sidecar = status_path_for(art)
    write_status(sidecar, "complete", config_key=cfg_key)

    progress = art.parent / "_progress.json"
    progress.write_text(json.dumps({"frame": 5}))

    assert art.exists()
    assert sidecar.exists()
    assert progress.exists()

    delete_artifact(data_dir, stage, cfg_key, stem)

    assert not art.exists()
    assert not sidecar.exists()
    assert not progress.exists()


def test_delete_artifact_noop_when_missing(tmp_path):
    delete_artifact(tmp_path, "detections", "c-missing", "no_video")

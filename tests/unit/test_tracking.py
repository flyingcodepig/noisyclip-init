"""Unit tests for run tracking and artifact paths."""

from __future__ import annotations

import json

import pytest

from noisyclip.tracking.artifacts import ArtifactStore, create_run_dir
from noisyclip.tracking.manifest import RunManifest
from noisyclip.utils.paths import PathSafetyError


def test_artifact_store_refuses_path_escape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ArtifactStore keeps paths inside the run directory."""

    store = ArtifactStore(tmp_path / "run")
    assert store.checkpoint().parent.name == "checkpoints"
    with pytest.raises(PathSafetyError):
        store.path("../escape.pt")


def test_run_manifest_writes_done_and_failed_markers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """RunManifest persists status markers with diagnostics."""

    run_dir = create_run_dir(tmp_path, "abc")
    manifest = RunManifest(run_dir, {"config_digest": "c"})
    manifest.mark_failed("boom", stage="unit")
    failed = json.loads((run_dir / "FAILED").read_text(encoding="utf-8"))
    assert failed == {"reason": "boom", "stage": "unit"}

    done_dir = create_run_dir(tmp_path, "done")
    done_manifest = RunManifest(done_dir, {})
    done_manifest.mark_done()
    assert (done_dir / "DONE").is_file()


def test_create_run_dir_refuses_existing_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Existing run directories fail when fail_if_run_exists is true."""

    create_run_dir(tmp_path, "same")
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "same")

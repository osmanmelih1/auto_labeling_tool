"""Where the YOLO base checkpoint ends up.

Ultralytics downloads a checkpoint it cannot find into the current working
directory rather than to the path it was given. Left alone that scatters .pt
files across the project root, which is how two of them ended up there. These
tests pin the four cases the resolution has to handle.

Skipped when ultralytics is not installed, so the rest of the suite still runs
on a machine without the training dependencies.
"""

from pathlib import Path

import pytest

pytest.importorskip("ultralytics", reason="training dependencies are not installed")

from src.core.step6_train import BASE_MODEL, YoloTrainer  # noqa: E402


def assert_is_stored_checkpoint(resolved: str) -> None:
    """Assert a resolved path points at the stored checkpoint.

    Compared as paths rather than as strings: the constant is written with
    forward slashes and Windows hands back backslashes, which is correct on both
    and unequal as text.

    Args:
        resolved: Path returned by the trainer.
    """
    assert Path(resolved) == Path(BASE_MODEL)


def test_the_default_checkpoint_lives_with_the_other_weights():
    """SAM and DINOv3 are in data/models; the detector base belongs there too."""
    assert BASE_MODEL.startswith("data/models/")


def test_the_first_run_defers_to_ultralytics(project_sandbox):
    """With nothing on disk only Ultralytics can supply the file.

    Args:
        project_sandbox: The sandboxed project root.
    """
    assert YoloTrainer().resolve_base_model() == "yolov8n.pt"


def test_a_downloaded_checkpoint_is_moved_out_of_the_root(project_sandbox):
    """Tidying after training is what keeps the root clean.

    Args:
        project_sandbox: The sandboxed project root.
    """
    (project_sandbox / "yolov8n.pt").write_bytes(b"weights")

    YoloTrainer().tidy_downloaded_checkpoints()

    assert not (project_sandbox / "yolov8n.pt").exists()
    assert (project_sandbox / "data" / "models" / "yolov8n.pt").read_bytes() == b"weights"


def test_the_amp_check_download_is_filed_away_too(project_sandbox):
    """Ultralytics downloads a second, unrelated model to verify mixed precision.

    Only the base model was handled at first, which is why a stray yolo26n.pt
    kept reappearing at the project root after every training run.

    Args:
        project_sandbox: The sandboxed project root.
    """
    (project_sandbox / "yolov8n.pt").write_bytes(b"base")
    (project_sandbox / "yolo26n.pt").write_bytes(b"amp check")

    YoloTrainer().tidy_downloaded_checkpoints()

    assert list(project_sandbox.glob("*.pt")) == []
    assert (project_sandbox / "data" / "models" / "yolo26n.pt").read_bytes() == b"amp check"


def test_a_second_run_reuses_the_local_copy(project_sandbox):
    """No network, no download, no stray file.

    Args:
        project_sandbox: The sandboxed project root.
    """
    models = project_sandbox / "data" / "models"
    models.mkdir(parents=True)
    (models / "yolov8n.pt").write_bytes(b"weights")

    assert_is_stored_checkpoint(YoloTrainer().resolve_base_model())


def test_a_stray_checkpoint_is_adopted_rather_than_redownloaded(project_sandbox):
    """Someone upgrading from an older checkout already has the file.

    Args:
        project_sandbox: The sandboxed project root.
    """
    (project_sandbox / "yolov8n.pt").write_bytes(b"weights")

    assert_is_stored_checkpoint(YoloTrainer().resolve_base_model())
    assert not (project_sandbox / "yolov8n.pt").exists()


def test_tidying_never_overwrites_the_stored_checkpoint(project_sandbox):
    """A leftover at the root must not replace the copy already in use.

    Args:
        project_sandbox: The sandboxed project root.
    """
    models = project_sandbox / "data" / "models"
    models.mkdir(parents=True)
    (models / "yolov8n.pt").write_bytes(b"stored")
    (project_sandbox / "yolov8n.pt").write_bytes(b"different")

    YoloTrainer().tidy_downloaded_checkpoints()

    assert (models / "yolov8n.pt").read_bytes() == b"stored"
    assert not (project_sandbox / "yolov8n.pt").exists()

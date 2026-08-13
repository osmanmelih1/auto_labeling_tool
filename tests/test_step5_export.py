"""What reaches the training set, and what is deliberately kept out.

Three different things look identical on disk and mean opposite things: an image
with no label file, an image whose label file is empty, and an image whose label
file is a draft nobody has approved. Treating them the same is how a detector
ends up trained on frames nobody looked at.
"""

import json

import pytest
from PyQt6.QtGui import QImage

from src.core.step5_export import BACKGROUND_SHARE, DatasetExporter
from src.core.yolo_format import write_yolo_boxes


@pytest.fixture
def dataset(project_sandbox):
    """Build a small dataset covering every case the exporter has to separate.

    Args:
        project_sandbox: The sandboxed project root.

    Returns:
        pathlib.Path: The project root.
    """
    images = project_sandbox / "data" / "deduplicated"
    images.mkdir(parents=True)
    labels = project_sandbox / "data" / "labels"
    labels.mkdir(parents=True)

    for index in range(20):
        stem = f"object_{index:02d}"
        QImage(320, 240, QImage.Format.Format_RGB32).save(str(images / f"{stem}.png"))
        write_yolo_boxes(str(labels / f"{stem}.txt"), [(index % 3, 0.5, 0.5, 0.2, 0.2)])

    for index in range(10):
        stem = f"confirmed_empty_{index:02d}"
        QImage(320, 240, QImage.Format.Format_RGB32).save(str(images / f"{stem}.png"))
        write_yolo_boxes(str(labels / f"{stem}.txt"), [])

    QImage(320, 240, QImage.Format.Format_RGB32).save(str(images / "never_labelled.png"))

    QImage(320, 240, QImage.Format.Format_RGB32).save(str(images / "still_queued.png"))
    write_yolo_boxes(str(labels / "still_queued.txt"), [(0, 0.5, 0.5, 0.2, 0.2)])
    (project_sandbox / "data" / "review_queue.json").write_text(
        json.dumps({"pending": {"still_queued": {"image_key": "still_queued"}}, "rejected": {}}),
        encoding="utf-8",
    )

    return project_sandbox


def test_an_image_with_no_label_is_not_exported(dataset):
    """To YOLO an unlabelled image asserts the frame is empty, which nobody checked.

    Args:
        dataset: The prepared project.
    """
    stems = {image.stem for image, _, _ in DatasetExporter().find_pairs()}

    assert "never_labelled" not in stems


def test_a_frame_still_in_the_review_queue_is_not_exported(dataset):
    """A label file existing is not the same as a label file being agreed with.

    Propagation and prediction write their proposals before anyone looks at them.
    Exporting one that is still queued trains the model on a draft.

    Args:
        dataset: The prepared project.
    """
    stems = {image.stem for image, _, _ in DatasetExporter().find_pairs()}

    assert "still_queued" not in stems


def test_confirmed_empty_frames_are_exported_as_backgrounds(dataset):
    """A human confirming a frame is empty is exactly what teaches "no object here".

    Args:
        dataset: The prepared project.
    """
    stems = {image.stem for image, _, _ in DatasetExporter().find_pairs()}

    assert any(stem.startswith("confirmed_empty") for stem in stems)


def test_backgrounds_are_capped_at_their_share(dataset):
    """A training set that is mostly empty frames teaches mostly emptiness.

    Args:
        dataset: The prepared project.
    """
    pairs = DatasetExporter().find_pairs()
    backgrounds = [image for image, _, class_ids in pairs if not class_ids]

    assert len(backgrounds) == pytest.approx(len(pairs) * BACKGROUND_SHARE, abs=1)


def test_the_choice_of_backgrounds_is_deterministic(dataset):
    """Re-exporting the same labels must reproduce the same dataset.

    Args:
        dataset: The prepared project.
    """
    first = [image.stem for image, _, class_ids in DatasetExporter().find_pairs() if not class_ids]
    second = [image.stem for image, _, class_ids in DatasetExporter().find_pairs() if not class_ids]

    assert first == second


def test_no_backgrounds_at_all_is_not_an_error(project_sandbox):
    """Most projects will have none until someone has rejected a frame.

    Args:
        project_sandbox: The sandboxed project root.
    """
    images = project_sandbox / "data" / "deduplicated"
    images.mkdir(parents=True)
    labels = project_sandbox / "data" / "labels"
    labels.mkdir(parents=True)
    QImage(320, 240, QImage.Format.Format_RGB32).save(str(images / "only.png"))
    write_yolo_boxes(str(labels / "only.txt"), [(0, 0.5, 0.5, 0.2, 0.2)])

    pairs = DatasetExporter().find_pairs()

    assert [image.stem for image, _, _ in pairs] == ["only"]

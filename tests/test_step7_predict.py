"""Pre-labelling with the trained detector.

The detector itself is Ultralytics' problem. What is tested here is everything
around it: which images get proposed at all, how a frame is tiered, and that the
queue entry it writes is the same shape the review screen already reads. A
mismatch there would not raise — the review screen would simply show a card with
nothing in it.
"""

from pathlib import Path

import pytest
from PyQt6.QtGui import QImage

pytest.importorskip("ultralytics", reason="training dependencies are not installed")

from src.core.review_queue import load_queue  # noqa: E402
from src.core.step7_predict import DetectorPreLabeller, find_latest_weights  # noqa: E402
from src.core.yolo_format import read_yolo_boxes, write_yolo_boxes  # noqa: E402


class FakeBox:
    """One detection shaped like an Ultralytics box."""

    def __init__(self, class_id: int, confidence: float, box: tuple):
        """Store the detection.

        Args:
            class_id: Predicted class.
            confidence: Predicted confidence.
            box: Normalised ``(x_center, y_center, width, height)``.
        """
        self.cls = _Scalar(class_id)
        self.conf = _Scalar(confidence)
        self.xywhn = [_Vector(box)]


class _Vector:
    """A row that answers ``.tolist()``, as a torch tensor row does."""

    def __init__(self, values):
        """Store the values.

        Args:
            values: The row to wrap.
        """
        self._values = list(values)

    def tolist(self):
        """Return the wrapped values.

        Returns:
            list: The row as plain floats.
        """
        return list(self._values)


class _Scalar:
    """A value that answers ``.item()``, as torch tensors do."""

    def __init__(self, value):
        """Store the value.

        Args:
            value: The scalar to wrap.
        """
        self._value = value

    def item(self):
        """Return the wrapped value.

        Returns:
            The scalar.
        """
        return self._value


class FakeResult:
    """One image's worth of detections."""

    def __init__(self, boxes):
        """Store the boxes.

        Args:
            boxes: The detections for this image.
        """
        self.boxes = boxes


class FakeModel:
    """Stands in for a trained detector, returning a scripted answer per image."""

    def __init__(self, by_stem: dict):
        """Store the scripted detections.

        Args:
            by_stem: Detections keyed by image stem.
        """
        self.by_stem = by_stem
        self.conf_asked = None

    def predict(self, paths, conf, verbose):
        """Return the scripted detections for each path.

        Args:
            paths: Image paths in this batch.
            conf: Confidence floor the caller asked for.
            verbose: Ignored.

        Returns:
            list: One FakeResult per path.
        """
        self.conf_asked = conf
        return [FakeResult(self.by_stem.get(Path(p).stem, [])) for p in paths]


@pytest.fixture
def prelabeller(project_sandbox, monkeypatch):
    """Build a pre-labeller over four images with a scripted detector.

    Args:
        project_sandbox: The sandboxed project root.
        monkeypatch: Used to replace the model with a scripted one.

    Returns:
        DetectorPreLabeller: Ready to run.
    """
    images = project_sandbox / "data" / "deduplicated"
    images.mkdir(parents=True)
    for stem in ("confident", "unsure", "nothing", "already_done"):
        QImage(320, 240, QImage.Format.Format_RGB32).save(str(images / f"{stem}.png"))

    labels = project_sandbox / "data" / "labels"
    labels.mkdir(parents=True)
    write_yolo_boxes(str(labels / "already_done.txt"), [(1, 0.5, 0.5, 0.2, 0.2)])

    weights = project_sandbox / "runs" / "train" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not really a checkpoint")

    monkeypatch.setattr("src.core.step7_predict.YOLO", lambda *a, **k: None)
    # The scarce-class gate is exercised on its own below; these tests are about
    # the confidence tiers, so it is switched off here rather than silently
    # deciding their outcome.
    prelabeller = DetectorPreLabeller(weights=str(weights), min_examples=0)
    prelabeller.model = FakeModel(
        {
            "confident": [FakeBox(2, 0.93, (0.3, 0.5, 0.2, 0.3)), FakeBox(2, 0.88, (0.7, 0.5, 0.2, 0.3))],
            "unsure": [FakeBox(2, 0.91, (0.3, 0.5, 0.2, 0.3)), FakeBox(3, 0.41, (0.7, 0.5, 0.1, 0.2))],
            "nothing": [],
        }
    )
    return prelabeller


def test_no_trained_detector_is_a_clear_error(project_sandbox):
    """Running this before training is a mistake worth naming.

    Args:
        project_sandbox: The sandboxed project root.
    """
    assert find_latest_weights() is None
    with pytest.raises(FileNotFoundError, match="Train YOLO"):
        DetectorPreLabeller()


def test_the_most_recent_run_is_chosen(project_sandbox):
    """Retraining should change which detector pre-labels the next batch.

    Args:
        project_sandbox: The sandboxed project root.
    """
    for name, mtime in (("train", 1_000_000), ("train-2", 2_000_000)):
        weights = project_sandbox / "runs" / name / "weights" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"x")
        import os

        os.utime(weights, (mtime, mtime))

    assert "train-2" in find_latest_weights()


def test_images_that_already_have_a_label_are_left_alone(prelabeller):
    """A label means a human decided, or was deemed not to need to. Do not overwrite.

    Args:
        prelabeller: The pre-labeller under test.
    """
    stems = {p.stem for p in prelabeller.unlabelled_images()}

    assert stems == {"confident", "unsure", "nothing"}


def test_a_confident_frame_is_written_without_a_human(prelabeller, project_sandbox):
    """Every box above the threshold means nobody needs to look.

    Args:
        prelabeller: The pre-labeller under test.
        project_sandbox: The sandboxed project root.
    """
    prelabeller.run()

    boxes = read_yolo_boxes(str(project_sandbox / "data" / "labels" / "confident.txt"))
    assert [class_id for class_id, *_ in boxes] == [2, 2]
    assert "confident" not in load_queue(prelabeller.review_queue_path)["pending"]


def test_one_weak_box_sends_the_whole_frame_to_review(prelabeller):
    """A frame is only accepted outright when every box in it would be.

    Args:
        prelabeller: The pre-labeller under test.
    """
    prelabeller.run()
    pending = load_queue(prelabeller.review_queue_path)["pending"]

    assert "unsure" in pending
    assert pending["unsure"]["weakest_score"] == pytest.approx(0.41)
    assert pending["unsure"]["score"] == pytest.approx(0.91)


def test_the_queue_entry_carries_a_score_per_box(prelabeller):
    """The review editor draws uncertain boxes differently and needs them in order.

    Args:
        prelabeller: The pre-labeller under test.
    """
    prelabeller.run()
    entry = load_queue(prelabeller.review_queue_path)["pending"]["unsure"]

    assert entry["box_scores"] == [0.91, 0.41]
    assert entry["object_count"] == 2
    assert entry["method"] == "detector_prediction"


def test_a_frame_with_no_detections_is_queued_rather_than_dropped(prelabeller, project_sandbox):
    """Ignoring these was the one option with no value in it.

    A frame the detector passed over is either a confirmed empty dock, which is
    worth training on, or an object it missed, which is worth catching. Spot
    checking twelve of them on the real dataset turned up one missed pallet.

    Args:
        prelabeller: The pre-labeller under test.
        project_sandbox: The sandboxed project root.
    """
    prelabeller.run()

    entry = load_queue(prelabeller.review_queue_path)["pending"]["nothing"]
    assert entry["object_count"] == 0
    assert entry["method"] == "detector_found_nothing"
    assert read_yolo_boxes(str(project_sandbox / "data" / "labels" / "nothing.txt")) == []


def test_an_unreviewed_empty_frame_is_not_yet_a_confirmed_background(prelabeller, project_sandbox):
    """The empty label only becomes a claim once a human accepts it.

    Args:
        prelabeller: The pre-labeller under test.
        project_sandbox: The sandboxed project root.
    """
    prelabeller.run()

    assert "nothing" in load_queue(prelabeller.review_queue_path)["pending"]


def test_the_detector_is_asked_for_everything_above_the_review_floor(prelabeller):
    """Filtering at the auto threshold would silently discard the review band.

    Args:
        prelabeller: The pre-labeller under test.
    """
    prelabeller.run()

    assert prelabeller.model.conf_asked == prelabeller.review_threshold


def test_a_class_with_almost_no_examples_is_never_auto_accepted(prelabeller):
    """Confidence is calibrated by training data, so a class seen twice has none.

    Observed on the real dataset: a detector with two examples of one class
    reported 0.95 and 0.98 on new frames. Those labels would have entered the
    next training set unread, and the error would compound every round.

    Args:
        prelabeller: The pre-labeller under test.
    """
    prelabeller.min_examples = 25
    prelabeller.model = FakeModel({"confident": [FakeBox(2, 0.99, (0.3, 0.5, 0.2, 0.3))]})

    prelabeller.run()

    assert "confident" in load_queue(prelabeller.review_queue_path)["pending"]


def test_a_well_evidenced_class_is_still_auto_accepted(prelabeller, project_sandbox):
    """The gate must not turn every frame into review work.

    Args:
        prelabeller: The pre-labeller under test.
        project_sandbox: The sandboxed project root.
    """
    labels = project_sandbox / "data" / "labels"
    for index in range(30):
        write_yolo_boxes(str(labels / f"seed_{index}.txt"), [(2, 0.5, 0.5, 0.2, 0.2)])

    prelabeller.min_examples = 25
    prelabeller.model = FakeModel({"confident": [FakeBox(2, 0.99, (0.3, 0.5, 0.2, 0.3))]})

    prelabeller.run()

    assert "confident" not in load_queue(prelabeller.review_queue_path)["pending"]


def test_one_scarce_class_holds_back_the_whole_frame(prelabeller, project_sandbox):
    """A frame is accepted outright only when every box in it can be.

    Args:
        prelabeller: The pre-labeller under test.
        project_sandbox: The sandboxed project root.
    """
    labels = project_sandbox / "data" / "labels"
    for index in range(30):
        write_yolo_boxes(str(labels / f"seed_{index}.txt"), [(2, 0.5, 0.5, 0.2, 0.2)])

    prelabeller.min_examples = 25
    prelabeller.model = FakeModel(
        {"confident": [FakeBox(2, 0.99, (0.3, 0.5, 0.2, 0.3)), FakeBox(4, 0.97, (0.7, 0.5, 0.2, 0.3))]}
    )

    prelabeller.run()

    assert "confident" in load_queue(prelabeller.review_queue_path)["pending"]


def test_the_boxes_are_still_written_when_a_frame_is_held_back(prelabeller, project_sandbox):
    """Held back means "a human decides", not "the work is thrown away".

    Args:
        prelabeller: The pre-labeller under test.
        project_sandbox: The sandboxed project root.
    """
    prelabeller.min_examples = 25
    prelabeller.model = FakeModel({"confident": [FakeBox(2, 0.99, (0.3, 0.5, 0.2, 0.3))]})

    prelabeller.run()

    assert read_yolo_boxes(str(project_sandbox / "data" / "labels" / "confident.txt")) == [
        (2, 0.3, 0.5, 0.2, 0.3)
    ]


def test_a_previously_rejected_frame_is_not_proposed_again(prelabeller):
    """Rejection memory is shared with propagation and must apply here too.

    Args:
        prelabeller: The pre-labeller under test.
    """
    from src.core.review_queue import save_queue

    save_queue(
        {"pending": {}, "rejected": {"unsure": {"score": 0.91}}},
        prelabeller.review_queue_path,
    )

    prelabeller.run()

    assert "unsure" not in load_queue(prelabeller.review_queue_path)["pending"]

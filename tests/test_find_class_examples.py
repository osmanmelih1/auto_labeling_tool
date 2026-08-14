"""Hunting for a class the dataset barely contains.

The risk in a discovery tool is that it quietly becomes a labelling tool. These
tests pin the boundary: it ranks, it copies previews, and it never writes to
data/labels. What it finds is a suggestion, and a suggestion that wrote itself
into the dataset would be worse than no tool at all.
"""

from pathlib import Path

import pytest
from PyQt6.QtGui import QImage

pytest.importorskip("ultralytics", reason="training dependencies are not installed")

from src.core.yolo_format import read_yolo_boxes, write_yolo_boxes  # noqa: E402
from src.tools.find_class_examples import ClassExampleFinder  # noqa: E402


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


class FakeBox:
    """One detection shaped like an Ultralytics box."""

    def __init__(self, class_id, confidence, box=(0.5, 0.5, 0.2, 0.2)):
        """Store the detection.

        Args:
            class_id: Predicted class.
            confidence: Predicted confidence.
            box: Normalised ``(x_center, y_center, width, height)``.
        """
        self.cls = _Scalar(class_id)
        self.conf = _Scalar(confidence)
        self.xywhn = [_Vector(box)]


class FakeResult:
    """One image's worth of detections."""

    def __init__(self, boxes):
        """Store the boxes.

        Args:
            boxes: The detections for this image.
        """
        self.boxes = boxes


class FakeModel:
    """Returns a scripted answer per image stem."""

    def __init__(self, by_stem):
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
def finder(project_sandbox, monkeypatch):
    """Search four frames for the rarest class in the fixture scheme.

    Args:
        project_sandbox: The sandboxed project root.
        monkeypatch: Used to replace the model with a scripted one.

    Returns:
        ClassExampleFinder: Ready to run.
    """
    images = project_sandbox / "data" / "deduplicated"
    images.mkdir(parents=True)
    for stem in ("strong", "faint", "labelled_already", "irrelevant"):
        QImage(320, 240, QImage.Format.Format_RGB32).save(str(images / f"{stem}.png"))

    labels = project_sandbox / "data" / "labels"
    labels.mkdir(parents=True)
    write_yolo_boxes(str(labels / "labelled_already.txt"), [(0, 0.5, 0.5, 0.2, 0.2)])
    write_yolo_boxes(str(labels / "irrelevant.txt"), [(2, 0.5, 0.5, 0.2, 0.2)])

    weights = project_sandbox / "runs" / "train" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not really a checkpoint")

    monkeypatch.setattr("ultralytics.YOLO", lambda *a, **k: None)
    finder = ClassExampleFinder("pallet_1", weights=str(weights))
    finder.model = FakeModel(
        {
            "strong": [FakeBox(0, 0.42), FakeBox(2, 0.99)],
            "faint": [FakeBox(0, 0.08)],
            "labelled_already": [FakeBox(0, 0.31)],
            "irrelevant": [FakeBox(2, 0.95)],
        }
    )
    return finder


def test_an_unknown_class_is_refused(project_sandbox):
    """Searching for a typo would return nothing and look like an answer.

    Args:
        project_sandbox: The sandboxed project root.
    """
    with pytest.raises(ValueError, match="No class named"):
        ClassExampleFinder("pallet_9")


def test_only_the_target_class_is_collected(finder):
    """A frame full of the common class is not evidence of the rare one.

    Args:
        finder: The finder under test.
    """
    found = finder.search(finder.images())

    assert [path.stem for path, *_ in found] == ["strong", "labelled_already", "faint"]


def test_results_are_ranked_by_confidence(finder):
    """The list is read from the top, so the order is the whole product.

    Args:
        finder: The finder under test.
    """
    found = finder.search(finder.images())

    assert [round(score, 2) for _, score, _ in found] == [0.42, 0.31, 0.08]


def test_labelled_frames_are_searched_too(finder):
    """A frame labelled as one class may hold a missed instance of another.

    That miss is exactly what is being hunted, so excluding labelled frames
    would exclude the most interesting result.

    Args:
        finder: The finder under test.
    """
    found = finder.search(finder.images())

    assert "labelled_already" in [path.stem for path, *_ in found]
    assert finder.already_labelled("labelled_already") is True
    assert finder.already_labelled("irrelevant") is False


def test_the_detector_is_asked_at_the_search_floor(finder):
    """Asking at a threshold anyone would trust defeats the point.

    Args:
        finder: The finder under test.
    """
    finder.search(finder.images())

    assert finder.model.conf_asked == finder.confidence


def test_previews_are_written_and_nothing_is_labelled(finder, project_sandbox):
    """The boundary this tool must not cross.

    Args:
        finder: The finder under test.
        project_sandbox: The sandboxed project root.
    """
    before = {p.name: read_yolo_boxes(str(p)) for p in (project_sandbox / "data" / "labels").glob("*.txt")}

    finder.run()

    previews = sorted(p.name for p in (project_sandbox / "data" / "candidates" / "pallet_1").glob("*.jpg"))
    assert len(previews) == 3
    assert previews[0].startswith("000_0.42_strong")

    after = {p.name: read_yolo_boxes(str(p)) for p in (project_sandbox / "data" / "labels").glob("*.txt")}
    assert after == before
    assert not (project_sandbox / "data" / "labels" / "strong.txt").exists()


def test_the_limit_bounds_what_is_written(finder, project_sandbox):
    """A shortlist of everything is not a shortlist.

    Args:
        finder: The finder under test.
        project_sandbox: The sandboxed project root.
    """
    finder.run(limit=2)

    previews = list((project_sandbox / "data" / "candidates" / "pallet_1").glob("*.jpg"))
    assert len(previews) == 2


def test_a_rerun_replaces_the_previous_shortlist(finder, project_sandbox):
    """Stale candidates from an older model would be read as current ones.

    Args:
        finder: The finder under test.
        project_sandbox: The sandboxed project root.
    """
    finder.run()
    finder.run(limit=1)

    previews = list((project_sandbox / "data" / "candidates" / "pallet_1").glob("*.jpg"))
    assert len(previews) == 1


def test_a_class_the_detector_never_suspects_says_so(finder, project_sandbox, capsys):
    """Silence and "nothing found" look identical unless one of them speaks.

    Args:
        finder: The finder under test.
        project_sandbox: The sandboxed project root.
        capsys: Captured output.
    """
    finder.model = FakeModel({})

    finder.run()

    out = capsys.readouterr().out
    assert "never suspects" in out
    assert "Seed some by hand" in out
    assert not (project_sandbox / "data" / "candidates").exists()


def test_an_unlabelled_frame_is_answered_quietly(project_sandbox, capsys):
    """A frame with no label file is what this search exists to find.

    Asked through read_yolo_boxes, the missing file is reported as an error, and
    a search across three thousand images buries its own results under [-] lines
    about frames that are perfectly fine.

    Args:
        project_sandbox: The sandboxed project root.
        capsys: Captured output.
    """
    labels = project_sandbox / "data" / "labels"
    labels.mkdir(parents=True)
    finder = ClassExampleFinder.__new__(ClassExampleFinder)
    finder.label_dir = labels
    finder.class_id = 0

    assert finder.already_labelled("never_seen") is False
    assert capsys.readouterr().out == ""

"""Reporting where the dataset stands between steps.

Every number here drives a decision the operator makes next — whether a class is
thin enough to hunt for, whether it has earned the detector's trust, how much of
the set is background. A wrong count sends the next hour in the wrong direction,
so the arithmetic is tested away from Qt.
"""

import json

import pytest

from src.core.dataset_summary import summarise
from src.core.yolo_format import write_yolo_boxes


@pytest.fixture
def labelled(project_sandbox):
    """Write three classes and a small, deliberately lopsided set of labels.

    Args:
        project_sandbox: The sandboxed project root.

    Returns:
        pathlib.Path: The label directory.
    """
    (project_sandbox / "data" / "classes.json").write_text(
        json.dumps({"classes": ["common", "scarce", "unused"]}), encoding="utf-8"
    )

    labels = project_sandbox / "data" / "labels"
    labels.mkdir(parents=True)
    for index in range(4):
        write_yolo_boxes(str(labels / f"busy_{index}.txt"), [(0, 0.5, 0.5, 0.2, 0.2)] * 2)
    write_yolo_boxes(str(labels / "rare.txt"), [(1, 0.5, 0.5, 0.2, 0.2)])
    write_yolo_boxes(str(labels / "nothing_here.txt"), [])
    return labels


def test_the_counts_describe_what_is_on_disk(labelled):
    """The headline numbers: frames decided, frames empty, boxes drawn.

    Args:
        labelled: The label directory.
    """
    summary = summarise(trust_at=5)

    assert summary.frames == 6
    assert summary.empty == 1
    assert summary.boxes == 9


def test_a_class_below_the_threshold_is_marked_untrusted(labelled):
    """This is the flag that decides whether pre-labelling may auto-accept it.

    Args:
        labelled: The label directory.
    """
    summary = summarise(trust_at=5)
    rows = {row.name: row for row in summary.classes}

    assert rows["common"].boxes == 8
    assert rows["common"].trusted is True
    assert rows["scarce"].boxes == 1
    assert rows["scarce"].trusted is False


def test_a_class_nobody_has_drawn_is_still_reported(labelled):
    """A missing row hides the very thing worth seeing.

    A class defined but never used cannot be learned, and the operator needs to
    notice that before training rather than after.

    Args:
        labelled: The label directory.
    """
    summary = summarise(trust_at=5)
    rows = {row.name: row for row in summary.classes}

    assert "unused" in rows
    assert rows["unused"].boxes == 0
    assert rows["unused"].trusted is False


def test_the_share_shows_how_lopsided_the_set_is(labelled):
    """Imbalance is what starved palet_1li, and it is invisible in raw counts.

    Args:
        labelled: The label directory.
    """
    rows = {row.name: row for row in summarise(trust_at=5).classes}

    assert rows["common"].share == pytest.approx(8 / 9)
    assert rows["scarce"].share == pytest.approx(1 / 9)
    assert rows["unused"].share == 0.0


def test_an_empty_project_reports_zeroes_rather_than_dividing_by_them(project_sandbox):
    """The panel is drawn before anything is labelled, on every fresh project.

    Args:
        project_sandbox: The sandboxed project root.
    """
    (project_sandbox / "data" / "classes.json").write_text(
        json.dumps({"classes": ["thing"]}), encoding="utf-8"
    )

    summary = summarise()

    assert summary.frames == 0
    assert summary.boxes == 0
    assert summary.classes[0].share == 0.0


def test_a_missing_label_directory_is_not_an_error(project_sandbox):
    """Step 1 has not run yet on a fresh checkout, and the GUI still opens.

    Args:
        project_sandbox: The sandboxed project root.
    """
    (project_sandbox / "data" / "classes.json").write_text(
        json.dumps({"classes": ["thing"]}), encoding="utf-8"
    )

    summary = summarise(label_dir="data/labels_that_do_not_exist")

    assert summary.frames == 0
    assert summary.boxes == 0

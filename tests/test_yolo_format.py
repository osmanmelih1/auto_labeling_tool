"""The label format every step reads and writes.

This module is the narrowest waist of the whole pipeline: seeding writes through
it, propagation writes through it, the review editor rewrites through it and the
exporter reads through it. A silent change in rounding or parsing here would
show up as mislabelled data rather than as an error.
"""

import pytest

from src.core.yolo_format import (
    box_area,
    iou,
    read_yolo_boxes,
    write_yolo_boxes,
    yolo_box_to_pixels,
)


def test_round_trip_is_exact_to_six_places(tmp_path):
    """Writing and reading back must not move a box."""
    path = str(tmp_path / "labels.txt")
    boxes = [(0, 0.125, 0.25, 0.5, 0.75), (4, 0.9, 0.1, 0.05, 0.02)]

    write_yolo_boxes(path, boxes)

    assert read_yolo_boxes(path) == boxes


def test_malformed_lines_are_skipped_not_raised(tmp_path):
    """One bad line must not abandon the other labels in the file.

    A run spans thousands of images and the caller has nothing better to do with
    the exception than ignore that line, so the parser drops it quietly.
    """
    path = tmp_path / "labels.txt"
    path.write_text(
        "0 0.1 0.2 0.3 0.4\n"
        "\n"
        "1 0.1 0.2 0.3\n"  # too few fields
        "2 0.1 0.2 0.3 0.4 0.5\n"  # too many fields
        "x 0.1 0.2 0.3 0.4\n"  # class id is not an integer
        "3 a b c d\n"  # coordinates are not numbers
        "  4   0.5   0.5   0.2   0.2  \n",  # ragged whitespace is still valid
        encoding="utf-8",
    )

    assert read_yolo_boxes(str(path)) == [(0, 0.1, 0.2, 0.3, 0.4), (4, 0.5, 0.5, 0.2, 0.2)]


def test_missing_file_reads_as_empty(tmp_path):
    """An image with no label file has no boxes, which is not an error."""
    assert read_yolo_boxes(str(tmp_path / "nothing.txt")) == []


def test_yolo_box_to_pixels_centres_the_box():
    """A centred half-size box must cover the middle half of the frame."""
    assert yolo_box_to_pixels((0.5, 0.5, 0.5, 0.5), 400, 200) == (100, 50, 300, 150)


def test_box_area_is_normalised():
    """Area is a fraction of the frame, not a pixel count."""
    assert box_area((0.5, 0.5, 0.5, 0.5)) == 0.25


def test_iou_of_a_box_with_itself_is_one():
    """Identical boxes overlap completely."""
    box = (0.5, 0.5, 0.4, 0.4)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    """Boxes that do not touch must not be merged as duplicates."""
    assert iou((0.2, 0.2, 0.2, 0.2), (0.8, 0.8, 0.2, 0.2)) == 0.0


def test_iou_of_half_overlapping_boxes():
    """Two equal boxes sharing half their area score one third.

    Intersection over union, not intersection over area: half the area shared
    means 0.5 over 1.5. Propagation merges above 0.55, so this pair stays two
    separate detections, which is the behaviour that keeps adjacent pallets from
    collapsing into one box.
    """
    a = (0.25, 0.5, 0.5, 0.5)
    b = (0.50, 0.5, 0.5, 0.5)
    assert abs(iou(a, b) - 1 / 3) < 1e-9


def test_a_frame_with_no_label_file_reads_as_no_boxes_and_says_nothing(tmp_path, capsys):
    """An unlabelled frame is the ordinary state here, not a failure.

    Every frame starts without a label file, and most callers are asking exactly
    whether one exists yet. Reported as an error, it made the GUI and the class
    search print a [-] line per healthy frame and bury their real output.

    Args:
        tmp_path: Pytest's temporary directory.
        capsys: Captured output.
    """
    assert read_yolo_boxes(str(tmp_path / "never_labelled.txt")) == []
    assert capsys.readouterr().out == ""


def test_a_file_that_cannot_be_read_is_still_reported(tmp_path, capsys, monkeypatch):
    """Silence is only right for absence; a real read failure must be visible.

    Args:
        tmp_path: Pytest's temporary directory.
        capsys: Captured output.
        monkeypatch: Used to make opening the file fail.
    """
    path = tmp_path / "unreadable.txt"
    path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def refuse(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", refuse)

    assert read_yolo_boxes(str(path)) == []
    assert "[-] Could not read" in capsys.readouterr().out

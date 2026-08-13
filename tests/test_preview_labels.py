"""Rendering the labels as they are now.

The pipeline's debug images are snapshots of what a machine proposed when it
proposed it, and stay that way after a human corrects the frame. Reading them
back as the dataset is how a corrected frame gets corrected twice. These render
the label files instead, so the thing on screen is the thing on disk.
"""

import pytest
from PyQt6.QtGui import QImage

from src.core.yolo_format import read_yolo_boxes, write_yolo_boxes
from src.tools.preview_labels import main, summarise


@pytest.fixture
def frames(project_sandbox):
    """Write four frames covering the cases the filters have to separate.

    Args:
        project_sandbox: The sandboxed project root.

    Returns:
        pathlib.Path: The project root.
    """
    images = project_sandbox / "data" / "deduplicated"
    images.mkdir(parents=True)
    labels = project_sandbox / "data" / "labels"
    labels.mkdir(parents=True)

    for stem, boxes in (
        ("two_pallets", [(2, 0.3, 0.5, 0.2, 0.3), (2, 0.7, 0.5, 0.2, 0.3)]),
        ("one_carton", [(3, 0.5, 0.5, 0.2, 0.2)]),
        ("confirmed_empty", []),
        ("mixed", [(2, 0.3, 0.5, 0.2, 0.3), (3, 0.7, 0.5, 0.2, 0.3)]),
    ):
        QImage(320, 240, QImage.Format.Format_RGB32).save(str(images / f"{stem}.png"))
        write_yolo_boxes(str(labels / f"{stem}.txt"), boxes)

    return project_sandbox


def previews(project_sandbox) -> list[str]:
    """List the preview filenames written so far.

    Args:
        project_sandbox: The sandboxed project root.

    Returns:
        list: Filenames, sorted.
    """
    return sorted(p.name for p in (project_sandbox / "data" / "previews").glob("*.jpg"))


def test_every_labelled_frame_is_rendered_by_default(frames):
    """The default is a look at the whole dataset.

    Args:
        frames: The prepared project.
    """
    assert main([]) == 0

    assert len(previews(frames)) == 4


def test_the_filename_says_what_is_in_the_frame(frames):
    """A folder of previews is scanned by filename before it is opened.

    Args:
        frames: The prepared project.
    """
    main([])

    assert any(name.startswith("2-pallet_3_") for name in previews(frames))
    assert any(name.startswith("empty_") for name in previews(frames))


def test_only_the_requested_keys_are_rendered(frames):
    """Checking a handful of frames should not render four hundred.

    Args:
        frames: The prepared project.
    """
    main(["--keys", "one_carton,mixed"])

    assert len(previews(frames)) == 2


def test_keys_can_come_from_a_file(frames):
    """Thirty keys do not belong on a command line.

    Args:
        frames: The prepared project.
    """
    keys_file = frames / "keys.txt"
    keys_file.write_text("one_carton\nmixed\n\n", encoding="utf-8")

    main(["--keys-file", str(keys_file)])

    assert len(previews(frames)) == 2


def test_filtering_by_class_finds_every_frame_holding_it(frames):
    """Auditing one class is the common case: it is the one someone doubts.

    Args:
        frames: The prepared project.
    """
    main(["--class", "carton"])

    assert sorted(name.rsplit("carton_", 1)[1] for name in previews(frames)) == [
        "mixed.jpg",
        "one_carton.jpg",
    ]


def test_an_unknown_class_is_refused_rather_than_returning_nothing(frames):
    """Silently rendering zero frames for a typo looks like an answer.

    Args:
        frames: The prepared project.
    """
    assert main(["--class", "pallet_9"]) == 1


def test_confirmed_empty_frames_can_be_reviewed_on_their_own(frames):
    """These become background images, so they are worth a look before training.

    Args:
        frames: The prepared project.
    """
    main(["--empty"])

    assert previews(frames) == ["empty_confirmed_empty.jpg"]


def test_a_rerun_replaces_the_previous_previews(frames):
    """A stale preview read as current is the bug this tool exists to avoid.

    Args:
        frames: The prepared project.
    """
    main([])
    main(["--keys", "mixed"])

    assert len(previews(frames)) == 1


def test_the_labels_are_never_touched(frames):
    """This looks at the dataset; it does not have opinions about it.

    Args:
        frames: The prepared project.
    """
    labels = frames / "data" / "labels"
    before = {p.name: read_yolo_boxes(str(p)) for p in labels.glob("*.txt")}

    main([])

    assert {p.name: read_yolo_boxes(str(p)) for p in labels.glob("*.txt")} == before


def test_summarise_counts_each_class():
    """The filename has to distinguish one pallet from three at a glance."""
    names = ["a", "b", "c"]

    assert summarise(names, [(0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (2, 0, 0, 0, 0)]) == "2-a_1-c"
    assert summarise(names, []) == "empty"

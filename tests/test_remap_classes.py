"""Reorganising the class scheme without corrupting the labels.

A class's position in the list is its YOLO class id, so removing one from the
middle changes what every label above it means. That is why the class editor
refused to do it, and it is why this tool has to renumber every label file in
the same breath. A bug here does not raise: it silently relabels the dataset.
"""

import json

import pytest

from src.core.yolo_format import count_boxes_per_class, read_yolo_boxes, write_yolo_boxes
from src.tools.remap_classes import build_remap, main


@pytest.fixture
def project(project_sandbox):
    """Write four classes and a handful of labels using all of them.

    Args:
        project_sandbox: The sandboxed project root.

    Returns:
        pathlib.Path: The label directory.
    """
    (project_sandbox / "data" / "classes.json").write_text(
        json.dumps(
            {
                "classes": [
                    {"name": "pallet_1", "description": "one row"},
                    {"name": "pallet_2", "description": "two rows"},
                    {"name": "irregular", "description": "everything else"},
                    {"name": "carton", "description": "cardboard"},
                ]
            }
        ),
        encoding="utf-8",
    )

    labels = project_sandbox / "data" / "labels"
    labels.mkdir(parents=True)
    write_yolo_boxes(str(labels / "a.txt"), [(0, 0.1, 0.1, 0.1, 0.1), (3, 0.2, 0.2, 0.1, 0.1)])
    write_yolo_boxes(str(labels / "b.txt"), [(2, 0.3, 0.3, 0.1, 0.1), (3, 0.4, 0.4, 0.1, 0.1)])
    write_yolo_boxes(str(labels / "c.txt"), [(1, 0.5, 0.5, 0.1, 0.1)])
    write_yolo_boxes(str(labels / "empty.txt"), [])
    return labels


def test_counting_reports_every_class_in_use(project):
    """The report is the reason to run this at all: which classes carry the dataset.

    Args:
        project: The label directory.
    """
    assert count_boxes_per_class(str(project)) == {0: 1, 1: 1, 2: 1, 3: 2}


def test_classes_above_the_removed_one_shift_down():
    """This is the renumbering the class editor could not do and so forbade."""
    records = [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]

    remap = build_remap(records, removed=1, reassign_to=None)

    assert remap == {0: 0, 1: None, 2: 1, 3: 2}


def test_a_merge_target_above_the_removed_class_shifts_too():
    """The subtle case: the destination itself moves down when it sat above.

    Merging class 1 into class 3 cannot simply write 3, because after the removal
    the class formerly known as 3 is class 2.
    """
    records = [{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}]

    remap = build_remap(records, removed=1, reassign_to=3)

    assert remap == {0: 0, 1: 2, 2: 1, 3: 2}


def test_a_merge_target_below_the_removed_class_stays_put():
    """Nothing below the removal moves."""
    records = [{"name": "a"}, {"name": "b"}, {"name": "c"}]

    remap = build_remap(records, removed=2, reassign_to=0)

    assert remap == {0: 0, 1: 1, 2: 0}


def test_nothing_is_written_without_apply(project, project_sandbox):
    """A dry run is the safety net; the labels are the only copy of the work.

    Args:
        project: The label directory.
        project_sandbox: The sandboxed project root.
    """
    before = {p.name: read_yolo_boxes(str(p)) for p in project.glob("*.txt")}

    assert main(["--merge", "irregular", "carton"]) == 0

    assert {p.name: read_yolo_boxes(str(p)) for p in project.glob("*.txt")} == before
    assert len(json.loads((project_sandbox / "data" / "classes.json").read_text())["classes"]) == 4


def test_deleting_a_class_that_still_holds_boxes_is_refused(project):
    """Throwing away labelled work needs a more deliberate instruction than this.

    Args:
        project: The label directory.
    """
    assert main(["--delete", "irregular", "--apply"]) == 1
    assert count_boxes_per_class(str(project)) == {0: 1, 1: 1, 2: 1, 3: 2}


def test_deleting_an_unused_class_renumbers_the_rest(project, project_sandbox):
    """The whole point: class 3 becomes class 2 in every file that used it.

    Args:
        project: The label directory.
        project_sandbox: The sandboxed project root.
    """
    write_yolo_boxes(str(project / "b.txt"), [(3, 0.4, 0.4, 0.1, 0.1)])

    assert main(["--delete", "irregular", "--apply"]) == 0

    names = [
        c["name"] for c in json.loads((project_sandbox / "data" / "classes.json").read_text())["classes"]
    ]
    assert names == ["pallet_1", "pallet_2", "carton"]
    assert read_yolo_boxes(str(project / "a.txt")) == [
        (0, 0.1, 0.1, 0.1, 0.1),
        (2, 0.2, 0.2, 0.1, 0.1),
    ]
    assert read_yolo_boxes(str(project / "b.txt")) == [(2, 0.4, 0.4, 0.1, 0.1)]


def test_merging_moves_the_boxes_and_renumbers(project, project_sandbox):
    """Every irregular box becomes a carton, and carton's own id moves down.

    Args:
        project: The label directory.
        project_sandbox: The sandboxed project root.
    """
    assert main(["--merge", "irregular", "carton", "--apply"]) == 0

    names = [
        c["name"] for c in json.loads((project_sandbox / "data" / "classes.json").read_text())["classes"]
    ]
    assert names == ["pallet_1", "pallet_2", "carton"]
    assert count_boxes_per_class(str(project)) == {0: 1, 1: 1, 2: 3}


def test_descriptions_survive_the_renumbering(project, project_sandbox):
    """The rule that decides what belongs in a class is the expensive part of it.

    Args:
        project: The label directory.
        project_sandbox: The sandboxed project root.
    """
    main(["--merge", "irregular", "carton", "--apply"])

    records = json.loads((project_sandbox / "data" / "classes.json").read_text())["classes"]
    assert records[2] == {"name": "carton", "description": "cardboard"}


def test_an_unknown_class_name_is_reported_not_guessed(project):
    """Acting on a typo would relabel the wrong class silently.

    Args:
        project: The label directory.
    """
    with pytest.raises(SystemExit):
        main(["--delete", "palet_3lu", "--apply"])


def test_a_class_cannot_be_merged_into_itself(project):
    """A no-op that removes a class and keeps its boxes would corrupt them.

    Args:
        project: The label directory.
    """
    assert main(["--merge", "carton", "carton", "--apply"]) == 1


def test_the_report_runs_with_no_arguments(project, capsys):
    """Reading the class distribution is worth doing on its own.

    Args:
        project: The label directory.
        capsys: Captured output.
    """
    assert main([]) == 0

    out = capsys.readouterr().out
    assert "irregular" in out
    assert "5 box(es)" in out

"""The sidebar panel that keeps the dataset's standing on screen.

The arithmetic is tested in test_dataset_summary. What is worth testing here is
the wiring: that the panel reads the disk when told to, survives a project with
no classes at all, and does not accumulate stale rows across refreshes — it
rebuilds them, and a leak would show as a class listed twice.
"""

import json

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QScrollArea, QSizePolicy  # noqa: E402

from src.core.yolo_format import write_yolo_boxes  # noqa: E402
from src.gui.app import AutoLabelingApp, DatasetPanel  # noqa: E402


def write_classes(root, names):
    """Define the project's classes.

    Args:
        root: The sandboxed project root.
        names: Class names in id order.
    """
    (root / "data" / "classes.json").write_text(json.dumps({"classes": names}), encoding="utf-8")


def test_the_panel_reports_the_labels_on_disk(project_sandbox, qapp):
    """The whole point is that the number on screen is the number in data/labels.

    Args:
        project_sandbox: The sandboxed project root.
        qapp: The shared QApplication.
    """
    write_classes(project_sandbox, ["pallet"])
    labels = project_sandbox / "data" / "labels"
    labels.mkdir(parents=True)
    write_yolo_boxes(str(labels / "a.txt"), [(0, 0.5, 0.5, 0.2, 0.2)] * 3)
    write_yolo_boxes(str(labels / "b.txt"), [])

    panel = DatasetPanel()

    assert "3 boxes in 1 frame" in panel.totals_label.text()
    assert "1 confirmed empty" in panel.totals_label.text()
    assert len(panel.row_widgets) == 1


def test_refreshing_replaces_the_rows_rather_than_stacking_them(project_sandbox, qapp):
    """The panel refreshes after every step, so a leak here grows all session.

    Args:
        project_sandbox: The sandboxed project root.
        qapp: The shared QApplication.
    """
    write_classes(project_sandbox, ["pallet", "box"])
    (project_sandbox / "data" / "labels").mkdir(parents=True)

    panel = DatasetPanel()
    for _ in range(3):
        panel.refresh()

    assert len(panel.row_widgets) == 2


def test_a_class_added_while_the_window_is_open_appears_on_refresh(project_sandbox, qapp):
    """Classes are edited from this same window, so the panel cannot cache them.

    Args:
        project_sandbox: The sandboxed project root.
        qapp: The shared QApplication.
    """
    write_classes(project_sandbox, ["pallet"])
    (project_sandbox / "data" / "labels").mkdir(parents=True)
    panel = DatasetPanel()

    write_classes(project_sandbox, ["pallet", "box"])
    panel.refresh()

    assert len(panel.row_widgets) == 2


def test_a_project_with_no_classes_shows_a_hint_instead_of_an_empty_box(project_sandbox, qapp):
    """This is what every new project looks like on first launch.

    Args:
        project_sandbox: The sandboxed project root.
        qapp: The shared QApplication.
    """
    write_classes(project_sandbox, [])

    panel = DatasetPanel()

    assert len(panel.row_widgets) == 1
    assert "No classes yet" in panel.row_widgets[0].text()


def ancestors(widget):
    """Walk a widget's parents up to the window.

    Args:
        widget: The widget to start from.

    Yields:
        QWidget: Each ancestor in turn.
    """
    parent = widget.parent()
    while parent is not None:
        yield parent
        parent = parent.parent()


def test_the_last_pipeline_button_can_be_scrolled_to(project_sandbox, qapp):
    """The last step button became unreachable once the sidebar held ten controls.

    Enlarging the window did not help: the sidebar is a fixed width and the
    shortfall is vertical, so Qt compressed the column until the lower buttons
    were clipped and then gone. What fixes it is that the controls sit inside a
    scroll area, and that is what is asserted — the button being on screen at
    one particular window height would not survive the eleventh control.

    Args:
        project_sandbox: The sandboxed project root.
        qapp: The shared QApplication.
    """
    write_classes(project_sandbox, ["pallet"])
    window = AutoLabelingApp()

    assert any(isinstance(parent, QScrollArea) for parent in ancestors(window.btn_predict))


def test_the_dataset_panel_is_not_inside_the_scrolling_part(project_sandbox, qapp):
    """A number that has to be scrolled to is a number nobody reads.

    The panel is pinned under the scroll area so it stays visible however long
    the pipeline list grows.

    Args:
        project_sandbox: The sandboxed project root.
        qapp: The shared QApplication.
    """
    write_classes(project_sandbox, ["pallet"])
    window = AutoLabelingApp()

    assert not any(isinstance(parent, QScrollArea) for parent in ancestors(window.dataset_panel))


def test_the_panel_takes_only_the_height_it_needs(project_sandbox, qapp):
    """Pinned and expanding, it swallowed the sidebar and hid the step buttons.

    Being pinned below the scroll area means any height it claims is height the
    pipeline does not get, so its vertical policy has to be Maximum rather than
    the default.

    Args:
        project_sandbox: The sandboxed project root.
        qapp: The shared QApplication.
    """
    write_classes(project_sandbox, ["pallet"])
    window = AutoLabelingApp()

    assert window.dataset_panel.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Maximum

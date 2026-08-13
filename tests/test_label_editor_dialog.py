"""Editing a labelled frame that is not in any queue.

The review screen could always do this, but only for frames a propagation or
prediction run had queued. Once accepted, a frame left the queue and there was
no way to correct it short of deleting its label and starting over. Splitting a
class after labelling has begun makes that gap immediate: the frames holding the
old class are exactly the ones nobody has queued.
"""

import pytest
from PyQt6.QtGui import QImage

from src.core.yolo_format import read_yolo_boxes, write_yolo_boxes
from src.gui.app import LabelEditorDialog


@pytest.fixture
def frame(project_sandbox):
    """Write an image and a two-box label file.

    Args:
        project_sandbox: The sandboxed project root.

    Returns:
        tuple: The image path and the label path, as strings.
    """
    image_path = project_sandbox / "frame.png"
    QImage(320, 240, QImage.Format.Format_RGB32).save(str(image_path))
    label_path = project_sandbox / "frame.txt"
    write_yolo_boxes(str(label_path), [(4, 0.3, 0.5, 0.2, 0.3), (4, 0.7, 0.5, 0.2, 0.3)])
    return str(image_path), str(label_path)


def test_the_dialog_opens_the_existing_boxes(qapp, frame):
    """Editing starts from what is on disk, not from a blank canvas.

    Args:
        qapp: The shared QApplication.
        frame: The image and label paths.
    """
    dialog = LabelEditorDialog(*frame)

    assert [box["class_id"] for box in dialog.editor.boxes] == [4, 4]


def test_reclassifying_rewrites_one_line_rather_than_adding_one(qapp, frame):
    """The worry this answers: a corrected box must not become two boxes.

    Changing a class rewrites the class id of an existing label line. The box
    count is unchanged, and there is no second label for the same object.
    """
    image_path, label_path = frame
    dialog = LabelEditorDialog(image_path, label_path)

    dialog.editor.select(0)
    dialog.class_combo.setCurrentIndex(1)

    boxes = read_yolo_boxes(label_path)
    assert len(boxes) == 2
    assert [class_id for class_id, *_ in boxes] == [1, 4]


def test_the_geometry_survives_a_class_change(qapp, frame):
    """Placing the box is the expensive part; correcting the class must not undo it.

    Args:
        qapp: The shared QApplication.
        frame: The image and label paths.
    """
    image_path, label_path = frame
    before = read_yolo_boxes(label_path)
    dialog = LabelEditorDialog(image_path, label_path)

    dialog.editor.select(0)
    dialog.class_combo.setCurrentIndex(2)

    assert [box[1:] for box in read_yolo_boxes(label_path)] == [box[1:] for box in before]


def test_an_edit_is_saved_without_a_save_button(qapp, frame):
    """There is no save step, so there is nothing a close can lose.

    Args:
        qapp: The shared QApplication.
        frame: The image and label paths.
    """
    image_path, label_path = frame
    dialog = LabelEditorDialog(image_path, label_path)

    dialog.editor.select(1)
    dialog.delete_btn.click()
    dialog.close()

    assert len(read_yolo_boxes(label_path)) == 1


def test_the_toolbar_follows_the_selection(qapp, frame):
    """A class dropdown pointing at nothing is a way to corrupt a label.

    Args:
        qapp: The shared QApplication.
        frame: The image and label paths.
    """
    dialog = LabelEditorDialog(*frame)
    assert dialog.class_combo.isEnabled() is False
    assert dialog.delete_btn.isEnabled() is False

    dialog.editor.select(1)

    assert dialog.class_combo.isEnabled() is True
    assert dialog.delete_btn.isEnabled() is True
    assert dialog.class_combo.currentIndex() == 4


def test_the_status_line_reports_what_was_saved(qapp, frame):
    """Confirmation that the file changed, without opening it.

    Args:
        qapp: The shared QApplication.
        frame: The image and label paths.
    """
    dialog = LabelEditorDialog(*frame)
    dialog.editor.select(0)
    dialog.class_combo.setCurrentIndex(1)

    assert "pallet_2" in dialog.status_label.text()
    assert "irregular" in dialog.status_label.text()


def test_a_missing_image_is_reported_rather_than_crashing(qapp, project_sandbox):
    """A label file can outlive its image.

    Args:
        qapp: The shared QApplication.
        project_sandbox: The sandboxed project root.
    """
    dialog = LabelEditorDialog(str(project_sandbox / "gone.png"), str(project_sandbox / "gone.txt"))

    assert "Could not open" in dialog.status_label.text()

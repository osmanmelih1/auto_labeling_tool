"""The review canvas, where a human corrects what propagation produced.

Every assertion here is about the label file on disk, not about what the widget
believes. The editor's whole contract is that an edit is saved the moment it is
made, so a test that only checked the widget's own state would pass while the
pipeline received nothing.
"""

import pytest
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent

from src.core.yolo_format import read_yolo_boxes
from src.gui.label_editor import MIN_BOX_PX, LabelEditorView, rect_to_yolo, yolo_to_rect

IMAGE_SIZE = (400, 200)

# Synthesised mouse events carry integer viewport coordinates, so a drag is only
# ever accurate to one screen pixel, which is a little over one image pixel at
# the fitted scale. Assertions on dragged geometry allow for that; assertions on
# what reaches the label file do not need to.
DRAG_TOLERANCE_PX = 1.5


def press(view, scene_point, button=Qt.MouseButton.LeftButton):
    """Send a mouse press at a point given in scene coordinates.

    Args:
        view: The editor under test.
        scene_point: Position in image pixels.
        button: Mouse button to press.
    """
    pos = QPointF(view.mapFromScene(scene_point))
    view.mousePressEvent(
        QMouseEvent(QEvent.Type.MouseButtonPress, pos, button, button, Qt.KeyboardModifier.NoModifier)
    )


def drag_to(view, scene_point):
    """Send a mouse move to a point given in scene coordinates.

    Args:
        view: The editor under test.
        scene_point: Position in image pixels.
    """
    pos = QPointF(view.mapFromScene(scene_point))
    view.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove,
            pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


def release(view, scene_point):
    """Send a mouse release at a point given in scene coordinates.

    Args:
        view: The editor under test.
        scene_point: Position in image pixels.
    """
    pos = QPointF(view.mapFromScene(scene_point))
    view.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


@pytest.fixture
def editor(qapp, make_frame):
    """Open an editor on a two-box frame.

    Args:
        qapp: The shared QApplication.
        make_frame: Factory writing an image and a label file.

    Returns:
        tuple: The view and the path of the label file it is editing.
    """
    image_path, label_path = make_frame(
        "frame",
        [(0, 0.25, 0.5, 0.2, 0.4), (2, 0.75, 0.5, 0.1, 0.2)],
        size=IMAGE_SIZE,
    )
    view = LabelEditorView()
    view.resize(400, 200)
    view.show()
    assert view.load(image_path, label_path, [0.91, 0.80]) is True
    return view, label_path


def test_geometry_helpers_are_inverses():
    """A box converted to pixels and back must be unchanged."""
    box = (0.3, 0.6, 0.25, 0.5)
    rect = yolo_to_rect(box, *IMAGE_SIZE)
    assert rect_to_yolo(rect, *IMAGE_SIZE) == box


def test_load_places_boxes_in_image_pixels(editor):
    """A quarter-width box on a 400px frame starts at x=60 and spans 80px."""
    view, _ = editor
    assert len(view.boxes) == 2

    rect = view.boxes[0]["item"].rect()
    assert rect.x() == pytest.approx(60.0)
    assert rect.width() == pytest.approx(80.0)


def test_loading_does_not_disturb_the_label(editor):
    """Opening a frame must be a read; only an edit may rewrite the file."""
    view, label_path = editor
    assert view.to_yolo() == read_yolo_boxes(label_path)


def test_a_missing_image_is_reported_rather_than_crashing(qapp, tmp_path):
    """A queue entry can outlive its image; the editor must say so calmly."""
    view = LabelEditorView()
    assert view.load(str(tmp_path / "gone.png"), str(tmp_path / "gone.txt")) is False
    assert view.boxes == []


def test_number_key_reclassifies_only_the_selected_box(editor):
    """Reclassifying is the most repeated action in review and must cost one key."""
    view, label_path = editor
    view.select(0)

    view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_3, Qt.KeyboardModifier.NoModifier))

    assert [class_id for class_id, *_ in read_yolo_boxes(label_path)] == [2, 2]


def test_reclassifying_drops_the_machine_confidence(editor):
    """The score described the class the machine proposed, not the one a human chose."""
    view, _ = editor
    view.select(0)
    assert view.boxes[0]["score"] == 0.91

    view.set_selected_class(3)

    assert view.boxes[0]["score"] is None


def test_a_number_key_with_no_selection_changes_nothing(editor):
    """A stray keystroke must not silently relabel the frame."""
    view, label_path = editor
    view.select(-1)
    before = read_yolo_boxes(label_path)

    view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_2, Qt.KeyboardModifier.NoModifier))

    assert read_yolo_boxes(label_path) == before


def test_a_number_key_beyond_the_class_list_is_ignored(editor):
    """Five classes are defined, so 9 must not write class id 8."""
    view, label_path = editor
    view.select(0)

    view.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_9, Qt.KeyboardModifier.NoModifier))

    assert read_yolo_boxes(label_path)[0][0] == 0


def test_delete_removes_the_box_from_the_file(editor):
    """Deleting a spurious box is why review is an editor and not a verdict."""
    view, label_path = editor
    view.select(1)

    view.delete_selected()

    assert len(view.boxes) == 1
    assert len(read_yolo_boxes(label_path)) == 1
    assert view.selected_index == -1


def test_handles_hit_test_where_they_are_drawn(editor):
    """A handle the user can see must be a handle the user can grab."""
    view, _ = editor
    view.select(0)
    rect = view.boxes[0]["item"].rect()

    assert view._handle_at(QPointF(rect.left(), rect.top())) == (-1, -1)
    assert view._handle_at(QPointF(rect.right(), rect.center().y())) == (1, 0)
    assert view._handle_at(rect.center()) is None


def test_the_smallest_box_wins_a_click(editor):
    """A box nested inside another has to stay reachable."""
    view, _ = editor
    big = view.boxes[0]["item"].rect()
    view._add_box(1, QRectF(big.center().x() - 5, big.center().y() - 5, 10, 10), None)

    assert view._box_at(big.center()) == 2


def test_clamp_keeps_a_box_inside_the_image(editor):
    """The scene is padded so the view can scroll; boxes may not use that room."""
    view, _ = editor

    clamped = view._clamp(QRectF(-50, -50, 800, 600))

    assert clamped.left() >= 0 and clamped.top() >= 0
    assert clamped.right() <= IMAGE_SIZE[0] and clamped.bottom() <= IMAGE_SIZE[1]


def test_clamp_enforces_a_minimum_size(editor):
    """A zero-area box would be a label the detector cannot learn from."""
    view, _ = editor

    clamped = view._clamp(QRectF(200, 100, 0, 0))

    assert clamped.width() >= MIN_BOX_PX
    assert clamped.height() >= MIN_BOX_PX


def test_dragging_a_handle_resizes_and_saves(editor):
    """The commonest correction: propagation put the box slightly short."""
    view, label_path = editor
    view.select(0)
    before = QRectF(view.boxes[0]["item"].rect())

    press(view, QPointF(before.right(), before.bottom()))
    assert view._mode == "resize"
    drag_to(view, QPointF(before.right() + 40, before.bottom() + 20))
    release(view, QPointF(before.right() + 40, before.bottom() + 20))

    after = view.boxes[0]["item"].rect()
    assert after.width() > before.width() + 30
    assert read_yolo_boxes(label_path)[0][3] == pytest.approx(after.width() / IMAGE_SIZE[0], abs=1e-5)


def test_dragging_inside_a_box_moves_it(editor):
    """Moving must not resize: a correctly sized box in the wrong place is common."""
    view, _ = editor
    view.select(0)
    before = QRectF(view.boxes[0]["item"].rect())

    press(view, before.center())
    assert view._mode == "move"
    drag_to(view, QPointF(before.center().x() + 20, before.center().y()))
    release(view, QPointF(before.center().x() + 20, before.center().y()))

    after = view.boxes[0]["item"].rect()
    assert after.width() == pytest.approx(before.width(), abs=DRAG_TOLERANCE_PX)
    assert after.left() == pytest.approx(before.left() + 20, abs=DRAG_TOLERANCE_PX)


def test_dragging_on_empty_background_adds_a_box(editor):
    """Propagation misses objects; adding one must not mean starting the frame over."""
    view, label_path = editor
    view.default_class_id = 1

    press(view, QPointF(300, 20))
    assert view._mode == "create"
    drag_to(view, QPointF(360, 60))
    release(view, QPointF(360, 60))

    assert len(view.boxes) == 3
    assert view.boxes[-1]["class_id"] == 1
    assert len(read_yolo_boxes(label_path)) == 3


def test_clicking_a_box_to_select_it_is_not_an_edit(editor):
    """Selection must not rewrite the label or discard the machine's confidence.

    It did, and the cost was invisible: every selection rewrote the file, wiped
    the score of a box nobody had touched, and filled the console with edits that
    never happened, which made the log useless as a record of the work.
    """
    view, label_path = editor
    before = read_yolo_boxes(label_path)
    score_before = view.boxes[0]["score"]
    centre = view.boxes[0]["item"].rect().center()

    press(view, centre)
    release(view, centre)

    assert view.selected_index == 0
    assert read_yolo_boxes(label_path) == before
    assert view.boxes[0]["score"] == score_before


def test_a_stray_click_creates_nothing(editor):
    """A click that is not a drag is a misclick, not a label."""
    view, label_path = editor
    before = len(read_yolo_boxes(label_path))

    press(view, QPointF(380, 180))
    release(view, QPointF(380, 180))

    assert len(view.boxes) == 2
    assert len(read_yolo_boxes(label_path)) == before


def test_emptying_a_frame_leaves_an_empty_label(editor):
    """Deleting every box is a legitimate answer: the frame holds no objects."""
    view, label_path = editor
    view.select(1)
    view.delete_selected()
    view.select(0)
    view.delete_selected()

    assert read_yolo_boxes(label_path) == []

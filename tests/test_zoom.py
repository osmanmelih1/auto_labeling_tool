"""Zooming on both canvases.

The point under the cursor must stay under the cursor. This looks like a detail
and is not: a reviewer zooms in to judge a box edge, and a view that jumps to
its own centre on every notch makes that impossible.

It broke twice for two separate reasons, which is why it is tested rather than
eyeballed. Qt's AnchorUnderMouse zooms about the position the base class
recorded from the last mouse event it handled, so a view that consumes its own
move events zooms about a stale point. And a QGraphicsView whose scene fits
inside the viewport has nothing to scroll, so Qt centres it and discards any
attempt to shift it — which is why the scene is padded.
"""

import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent

from src.gui.app import ZoomableGraphicsView
from src.gui.label_editor import MAX_ZOOM_FACTOR, LabelEditorView

# Deliberately wider than it is tall, and larger than the viewport, so the two
# axes behave differently and a bug in one is not hidden by the other.
IMAGE_SIZE = (1600, 900)
VIEWPORT = (640, 420)

CURSORS = [QPoint(40, 30), QPoint(600, 380), QPoint(120, 350), QPoint(500, 40)]


def wheel(view, point, up=True):
    """Send a wheel event at a viewport position.

    Args:
        view: The view under test.
        point: Cursor position in viewport pixels.
        up: True to zoom in, False to zoom out.
    """
    view.wheelEvent(
        QWheelEvent(
            QPointF(point),
            view.mapToGlobal(point).toPointF(),
            QPoint(0, 0),
            QPoint(0, 120 if up else -120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
    )


@pytest.fixture
def editor(qapp, make_frame):
    """Open the review editor on a frame larger than its viewport.

    Args:
        qapp: The shared QApplication.
        make_frame: Factory writing an image and a label file.

    Returns:
        LabelEditorView: The view under test.
    """
    image_path, label_path = make_frame("frame", [(0, 0.5, 0.5, 0.2, 0.2)], size=IMAGE_SIZE)
    view = LabelEditorView()
    view.resize(*VIEWPORT)
    view.show()
    view.load(image_path, label_path, [0.9])
    return view


@pytest.fixture
def canvas(qapp, make_frame):
    """Open the seeding canvas on a frame larger than its viewport.

    Args:
        qapp: The shared QApplication.
        make_frame: Factory writing an image and a label file.

    Returns:
        ZoomableGraphicsView: The view under test.
    """
    image_path, _ = make_frame("frame", [], size=IMAGE_SIZE)
    view = ZoomableGraphicsView()
    view.resize(*VIEWPORT)
    view.show()
    view.load_image(image_path)
    return view


@pytest.mark.parametrize("cursor", CURSORS)
def test_the_point_under_the_cursor_stays_there_in_the_editor(editor, cursor):
    """Zooming in six notches must not move the pixel being pointed at.

    Args:
        editor: The review editor.
        cursor: Viewport position to zoom about.
    """
    anchored = editor.mapToScene(cursor)

    for _ in range(6):
        wheel(editor, cursor, up=True)

    back = editor.mapFromScene(anchored)
    assert abs(back.x() - cursor.x()) <= 2
    assert abs(back.y() - cursor.y()) <= 2


@pytest.mark.parametrize("cursor", CURSORS[:2])
def test_the_point_under_the_cursor_stays_there_on_the_seeding_canvas(canvas, cursor):
    """The seeding canvas shares the bug and therefore shares the test.

    Args:
        canvas: The seeding canvas.
        cursor: Viewport position to zoom about.
    """
    anchored = canvas.mapToScene(cursor)

    for _ in range(6):
        wheel(canvas, cursor, up=True)

    back = canvas.mapFromScene(anchored)
    assert abs(back.x() - cursor.x()) <= 2
    assert abs(back.y() - cursor.y()) <= 2


def test_zooming_out_returns_along_the_same_path(editor):
    """In and back out must land where it started, not drift towards the centre."""
    cursor = QPoint(90, 300)
    anchored = editor.mapToScene(cursor)
    fit = editor._fit_scale

    for _ in range(5):
        wheel(editor, cursor, up=True)
    for _ in range(5):
        wheel(editor, cursor, up=False)

    back = editor.mapFromScene(anchored)
    assert abs(back.x() - cursor.x()) <= 2
    assert abs(back.y() - cursor.y()) <= 2
    assert editor.transform().m11() == pytest.approx(fit)


def test_zooming_out_stops_at_the_fitted_image(editor):
    """Past the fitted view the image shrinks into an empty viewport and recentres."""
    for _ in range(30):
        wheel(editor, QPoint(300, 200), up=False)

    assert editor.transform().m11() == pytest.approx(editor._fit_scale)


def test_zooming_in_stops_at_the_ceiling(editor):
    """Past this the boxes are being placed on interpolated detail."""
    for _ in range(200):
        wheel(editor, QPoint(300, 200), up=True)

    assert editor.transform().m11() == pytest.approx(editor._fit_scale * MAX_ZOOM_FACTOR)


def test_the_handle_grab_radius_tracks_the_zoom(editor):
    """Handles are grabbed in screen pixels, so their scene radius must shrink."""
    at_fit = editor._tolerance()

    for _ in range(8):
        wheel(editor, QPoint(320, 210), up=True)

    assert editor._tolerance() < at_fit


def test_a_resize_refits_a_fitted_view(editor):
    """A view the user has not zoomed should keep showing the whole frame."""
    editor.resize(500, 300)

    assert editor.transform().m11() == pytest.approx(editor._fit_scale)


def test_a_resize_leaves_a_chosen_zoom_alone(editor, qapp):
    """Dragging the window must not throw away where the reviewer was looking."""
    wheel(editor, QPoint(200, 150), up=True)
    chosen = editor.transform().m11()

    editor.resize(700, 500)
    qapp.processEvents()

    assert editor.transform().m11() == pytest.approx(chosen)


def test_a_seed_box_cannot_be_drawn_into_the_scene_padding(canvas):
    """The padding exists so the view can scroll, not so boxes can leave the image."""
    inside = QPointF(canvas.mapFromScene(QPointF(800.0, 450.0)))
    outside = QPointF(canvas.mapFromScene(QPointF(2200.0, 1300.0)))

    canvas.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            inside,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    canvas.mouseMoveEvent(
        QMouseEvent(
            QEvent.Type.MouseMove,
            outside,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    canvas.mouseReleaseEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            outside,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )

    rect = canvas.pending_rect_item.rect()
    assert rect.right() <= IMAGE_SIZE[0]
    assert rect.bottom() <= IMAGE_SIZE[1]

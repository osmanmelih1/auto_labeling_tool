"""Interactive label editor for reviewing propagated boxes.

The review screen used to show a flat picture with the boxes burned into it. A
human could only accept the whole thing or throw it away, which is the wrong
choice to be offered: propagation is usually almost right. A box that is ten
pixels short, a spurious box on the floor, or the right box under the wrong
class each forced a rejection, and rejecting deletes the label, so the accurate
work SAM already did was thrown away with the one mistake.

This widget makes the review destructive only where it needs to be. Boxes can be
moved, resized, deleted, reclassified and added, and the label file is rewritten
in place. Rejection is left for the case it was meant for: the frame contains
nothing worth labelling.

It owns no pipeline state. It is handed an image and a label file, and writes
back to that same label file in the standard YOLO format, so the propagation
step and the exporter neither know nor care that a human touched it.
"""

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView

from src.core.class_config import class_color, class_name, load_classes
from src.core.tiers import AUTO_ACCEPT_THRESHOLD
from src.core.yolo_format import read_yolo_boxes, write_yolo_boxes

# Grab radius for the resize handles, in screen pixels rather than image pixels,
# so a handle stays equally easy to hit at any zoom level.
HANDLE_PX = 9

# Anything smaller than this in image pixels is treated as a stray click rather
# than a box.
MIN_BOX_PX = 4

# The eight resize handles, as (horizontal, vertical) edge selectors where -1 is
# the leading edge, 1 the trailing edge and 0 the midpoint.
HANDLES = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))

# How far past the fitted view zooming in is allowed to go. Beyond this the
# image is all interpolation and the boxes are being placed on invented detail.
MAX_ZOOM_FACTOR = 40.0

ZOOM_STEP = 1.15

# The scene is padded by this fraction of the image on every side. A
# QGraphicsView scrolls, and a scene that fits inside the viewport has nothing
# to scroll, so Qt centres it and quietly ignores any attempt to shift it. That
# is what makes an unpadded view zoom about its centre instead of about the
# cursor. The padding guarantees there is always somewhere to scroll to.
SCENE_PADDING = 0.5

HANDLE_CURSORS = {
    (-1, -1): Qt.CursorShape.SizeFDiagCursor,
    (1, 1): Qt.CursorShape.SizeFDiagCursor,
    (1, -1): Qt.CursorShape.SizeBDiagCursor,
    (-1, 1): Qt.CursorShape.SizeBDiagCursor,
    (0, -1): Qt.CursorShape.SizeVerCursor,
    (0, 1): Qt.CursorShape.SizeVerCursor,
    (-1, 0): Qt.CursorShape.SizeHorCursor,
    (1, 0): Qt.CursorShape.SizeHorCursor,
}


def class_qcolor(class_id: int) -> QColor:
    """Return the drawing colour for a class id.

    Colours are generated from the id rather than read from a fixed table, so the
    tool supports any number of classes in any project without a code change.

    Args:
        class_id: The class id to colour.

    Returns:
        QColor: A colour distinct from those of neighbouring class ids.
    """
    return QColor(*class_color(class_id))


def padded_scene_rect(width: int, height: int) -> QRectF:
    """Return the scene rectangle to use for an image of a given size.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        QRectF: The image rectangle with ``SCENE_PADDING`` added on every side.
    """
    return QRectF(
        -width * SCENE_PADDING,
        -height * SCENE_PADDING,
        width * (1 + 2 * SCENE_PADDING),
        height * (1 + 2 * SCENE_PADDING),
    )


def zoom_at_cursor(view: QGraphicsView, event, fit_scale: float) -> None:
    """Zoom a graphics view about the point under the cursor.

    Qt's ``AnchorUnderMouse`` looks like the obvious way to do this and is not
    reliable here: it zooms about the position Qt recorded from the last mouse
    event the base class saw, so a view that consumes its own move events keeps
    zooming about one stale point no matter where the cursor is. Doing the
    arithmetic explicitly removes that dependency — map the cursor to the scene,
    scale, map it again, and shift the view by however far it drifted.

    Zooming out stops at the fitted view, because past that the image shrinks
    inside the viewport and every further step recentres it, which reads as the
    picture jumping away from where the user was looking.

    Args:
        view: The view to zoom. Its transformation anchor must be ``NoAnchor``.
        event: The wheel event that triggered the zoom.
        fit_scale: Scale at which the whole image fits the viewport.
    """
    current = view.transform().m11()
    if not current:
        return

    step = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
    target = min(max(current * step, fit_scale), fit_scale * MAX_ZOOM_FACTOR)
    step = target / current
    if abs(step - 1.0) < 1e-9:
        return

    cursor = event.position().toPoint()
    before = view.mapToScene(cursor)
    view.scale(step, step)
    after = view.mapToScene(cursor)

    drift = after - before
    view.translate(drift.x(), drift.y())
    view.viewport().update()


def yolo_to_rect(box: tuple[float, float, float, float], width: int, height: int) -> QRectF:
    """Convert a normalised YOLO box into a pixel rectangle.

    Args:
        box: ``(x_center, y_center, width, height)`` normalised to [0, 1].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        QRectF: The same box in image pixel coordinates.
    """
    x_center, y_center, box_w, box_h = box
    return QRectF(
        (x_center - box_w / 2.0) * width,
        (y_center - box_h / 2.0) * height,
        box_w * width,
        box_h * height,
    )


def rect_to_yolo(rect: QRectF, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert a pixel rectangle into a normalised YOLO box.

    Args:
        rect: Rectangle in image pixel coordinates.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        tuple: ``(x_center, y_center, width, height)`` normalised to [0, 1].
    """
    return (
        rect.center().x() / width,
        rect.center().y() / height,
        rect.width() / width,
        rect.height() / height,
    )


class LabelEditorView(QGraphicsView):
    """Canvas that edits the YOLO boxes of one image in place.

    Boxes carry the confidence propagation gave them, and are drawn solid once
    that confidence would have been accepted outright and dashed while it would
    not. The reviewer's attention then goes to the dashed boxes, which is the
    only part of the frame a machine was unsure about.

    Attributes:
        boxes_changed: Emitted after any edit that alters the label content.
        selection_changed: Emitted with the selected box index, or -1 for none.
        status_message: Emitted with short text for the surrounding console.
    """

    boxes_changed = pyqtSignal()
    selection_changed = pyqtSignal(int)
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        """Configure the scene, render hints and interaction state.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # Zooming is anchored by hand in zoom_at_cursor, which needs Qt to leave
        # the transform alone.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setBackgroundBrush(QColor("#121212"))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.boxes: list[dict] = []
        self.selected_index = -1
        self.image_width = 0
        self.image_height = 0
        self.label_path: str | None = None
        self.default_class_id = 0
        self._fit_scale = 1.0

        self._mode: str | None = None
        self._handle: tuple[int, int] | None = None
        self._anchor = QPointF()
        self._origin_rect = QRectF()
        self._draft_item = None
        self._pan_start = None

    # ------------------------------------------------------------------ load

    def clear_image(self) -> None:
        """Empty the canvas and forget which label file was being edited."""
        self.scene.clear()
        self.boxes = []
        self.selected_index = -1
        self.label_path = None
        self.image_width = 0
        self.image_height = 0
        self._draft_item = None
        self.selection_changed.emit(-1)

    def load(self, image_path: str, label_path: str, scores: list[float] | None = None) -> bool:
        """Show an image together with the boxes of its label file.

        Args:
            image_path: Image to display.
            label_path: YOLO label file to edit; may not exist yet.
            scores: Per-box confidences in label-file order, when known.

        Returns:
            bool: True when the image could be loaded.
        """
        self.clear_image()

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            return False

        self.scene.addPixmap(pixmap)
        self.image_width = pixmap.width()
        self.image_height = pixmap.height()
        self.setSceneRect(padded_scene_rect(self.image_width, self.image_height))
        self._fit()

        self.label_path = label_path
        scores = scores or []
        for index, (class_id, *box) in enumerate(read_yolo_boxes(label_path)):
            score = scores[index] if index < len(scores) else None
            self._add_box(class_id, yolo_to_rect(tuple(box), self.image_width, self.image_height), score)

        return True

    def _fit(self) -> None:
        """Scale the whole image into the viewport and record that scale.

        The recorded scale is the floor for zooming out, so the image can never
        end up adrift in the middle of an empty viewport.
        """
        if not self.image_width:
            return
        self.resetTransform()
        # Fit the image, not the padded scene: the padding exists to make
        # scrolling possible, not to be looked at.
        self.fitInView(QRectF(0, 0, self.image_width, self.image_height), Qt.AspectRatioMode.KeepAspectRatio)
        self._fit_scale = self.transform().m11() or 1.0
        self.centerOn(self.image_width / 2, self.image_height / 2)

    def resizeEvent(self, event):
        """Refit the image when the pane changes size while fully zoomed out.

        A view left at the fitted scale should stay fitted; one the user has
        zoomed into should keep the zoom they chose.

        Args:
            event: Qt resize event.
        """
        was_fitted = abs(self.transform().m11() - self._fit_scale) < 1e-6
        super().resizeEvent(event)
        if self.image_width and was_fitted:
            self._fit()

    # ------------------------------------------------------------------ boxes

    def _add_box(self, class_id: int, rect: QRectF, score: float | None) -> int:
        """Create the graphics items for one box and register it.

        Args:
            class_id: Class the box belongs to.
            rect: Box in image pixel coordinates.
            score: Propagation confidence, or None for a hand-drawn box.

        Returns:
            int: Index of the new box.
        """
        item = self.scene.addRect(rect)
        tag = QGraphicsSimpleTextItem("")
        tag.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
        tag.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.scene.addItem(tag)

        self.boxes.append({"class_id": class_id, "score": score, "item": item, "tag": tag})
        index = len(self.boxes) - 1
        self._restyle(index)
        return index

    def _restyle(self, index: int) -> None:
        """Repaint one box's outline and caption from its current state.

        Args:
            index: Box to restyle.
        """
        box = self.boxes[index]
        rect = box["item"].rect()
        colour = class_qcolor(box["class_id"])
        uncertain = box["score"] is not None and box["score"] < AUTO_ACCEPT_THRESHOLD

        pen = QPen(QColor("#ffffff") if index == self.selected_index else colour)
        pen.setWidth(3 if index == self.selected_index else 2)
        pen.setCosmetic(True)
        if uncertain:
            pen.setStyle(Qt.PenStyle.DashLine)
        box["item"].setPen(pen)

        names = load_classes()
        caption = class_name(names, box["class_id"])
        if box["score"] is not None:
            caption = f"{caption} {box['score']:.2f}"
        if uncertain:
            caption = f"? {caption}"

        box["tag"].setText(caption)
        box["tag"].setBrush(colour)
        box["tag"].setPos(rect.left(), rect.top())

    def select(self, index: int) -> None:
        """Make one box the target of the class control and the delete key.

        Args:
            index: Box to select, or -1 to clear the selection.
        """
        previous = self.selected_index
        self.selected_index = index if 0 <= index < len(self.boxes) else -1

        for i in {previous, self.selected_index} - {-1}:
            if 0 <= i < len(self.boxes):
                self._restyle(i)

        self.viewport().update()
        self.selection_changed.emit(self.selected_index)

    def selected_class_id(self) -> int | None:
        """Return the class of the selected box.

        Returns:
            int | None: The class id, or None when nothing is selected.
        """
        if self.selected_index < 0:
            return None
        return self.boxes[self.selected_index]["class_id"]

    def set_selected_class(self, class_id: int) -> None:
        """Reclassify the selected box without touching its coordinates.

        Args:
            class_id: Class id to write instead.
        """
        if self.selected_index < 0:
            return
        box = self.boxes[self.selected_index]
        if box["class_id"] == class_id:
            return

        box["class_id"] = class_id
        # The machine's confidence was in the class it proposed, so it no longer
        # describes this box once a human has overruled it.
        box["score"] = None
        self._restyle(self.selected_index)
        self._commit(f"[*] Box {self.selected_index + 1} set to {class_name(load_classes(), class_id)}.")

    def delete_selected(self) -> None:
        """Remove the selected box from the image and from the label file."""
        if self.selected_index < 0:
            return

        box = self.boxes.pop(self.selected_index)
        self.scene.removeItem(box["item"])
        self.scene.removeItem(box["tag"])
        self.selected_index = -1
        self.selection_changed.emit(-1)

        for index in range(len(self.boxes)):
            self._restyle(index)
        self._commit("[-] Box deleted.")

    def to_yolo(self) -> list[tuple[int, float, float, float, float]]:
        """Return every box in the format written to a label file.

        Returns:
            list: ``(class_id, x_center, y_center, width, height)`` per box.
        """
        return [
            (box["class_id"], *rect_to_yolo(box["item"].rect(), self.image_width, self.image_height))
            for box in self.boxes
        ]

    def scores(self) -> list[float | None]:
        """Return the per-box confidences in label-file order.

        Returns:
            list: One score per box; None where a human drew or reclassified it.
        """
        return [box["score"] for box in self.boxes]

    def _commit(self, message: str) -> None:
        """Write the current boxes to the label file and announce the change.

        The label file is the pipeline's only record of this image, so an edit is
        saved the moment it is made rather than waiting for a save button. There
        is no state a crash or a mis-click could lose.

        Args:
            message: Console line describing what changed.
        """
        if self.label_path:
            write_yolo_boxes(self.label_path, self.to_yolo())
        self.status_message.emit(message)
        self.boxes_changed.emit()

    # --------------------------------------------------------------- geometry

    def _tolerance(self) -> float:
        """Return the handle grab radius converted to image pixels.

        Returns:
            float: Radius in scene units at the current zoom.
        """
        scale = self.transform().m11()
        return HANDLE_PX / scale if scale else HANDLE_PX

    @staticmethod
    def _handle_point(rect: QRectF, handle: tuple[int, int]) -> QPointF:
        """Return the scene position of one resize handle.

        Args:
            rect: Box the handle belongs to.
            handle: Edge selector pair.

        Returns:
            QPointF: Centre of the handle.
        """
        horizontal, vertical = handle
        x = rect.left() if horizontal < 0 else rect.right() if horizontal > 0 else rect.center().x()
        y = rect.top() if vertical < 0 else rect.bottom() if vertical > 0 else rect.center().y()
        return QPointF(x, y)

    def _handle_at(self, pos: QPointF) -> tuple[int, int] | None:
        """Find the handle of the selected box under a scene position.

        Args:
            pos: Position in scene coordinates.

        Returns:
            tuple | None: The handle, or None when the position misses them all.
        """
        if self.selected_index < 0:
            return None

        rect = self.boxes[self.selected_index]["item"].rect()
        tolerance = self._tolerance()
        for handle in HANDLES:
            point = self._handle_point(rect, handle)
            if abs(pos.x() - point.x()) <= tolerance and abs(pos.y() - point.y()) <= tolerance:
                return handle
        return None

    def _box_at(self, pos: QPointF) -> int:
        """Find the smallest box containing a scene position.

        The smallest is chosen so a box nested inside another stays reachable.

        Args:
            pos: Position in scene coordinates.

        Returns:
            int: Index of the box, or -1 when the position is on the background.
        """
        hits = [i for i, box in enumerate(self.boxes) if box["item"].rect().contains(pos)]
        if not hits:
            return -1

        def area(index: int) -> float:
            rect = self.boxes[index]["item"].rect()
            return rect.width() * rect.height()

        return min(hits, key=area)

    def _clamp(self, rect: QRectF) -> QRectF:
        """Keep a rectangle inside the image and above the minimum size.

        Args:
            rect: Rectangle to constrain.

        Returns:
            QRectF: The constrained rectangle.
        """
        left = max(0.0, min(rect.left(), self.image_width - MIN_BOX_PX))
        top = max(0.0, min(rect.top(), self.image_height - MIN_BOX_PX))
        right = min(float(self.image_width), max(rect.right(), left + MIN_BOX_PX))
        bottom = min(float(self.image_height), max(rect.bottom(), top + MIN_BOX_PX))
        return QRectF(left, top, right - left, bottom - top)

    # ----------------------------------------------------------------- events

    def drawForeground(self, painter, rect):
        """Draw the resize handles of the selected box.

        Args:
            painter: Painter supplied by Qt for the foreground layer.
            rect: Exposed scene rectangle to paint over.
        """
        super().drawForeground(painter, rect)
        if self.selected_index < 0:
            return

        box_rect = self.boxes[self.selected_index]["item"].rect()
        size = self._tolerance()
        painter.setPen(QPen(QColor("#101010"), 0))
        painter.setBrush(QColor("#ffffff"))
        for handle in HANDLES:
            point = self._handle_point(box_rect, handle)
            painter.drawRect(QRectF(point.x() - size / 2, point.y() - size / 2, size, size))

    def wheelEvent(self, event):
        """Zoom around the cursor in response to the scroll wheel.

        Args:
            event: Qt wheel event.
        """
        if not self.image_width:
            return
        zoom_at_cursor(self, event, self._fit_scale)

    def mousePressEvent(self, event):
        """Begin a pan, a resize, a move, or a new box.

        Args:
            event: Qt mouse event.
        """
        if not self.image_width:
            return

        if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._mode = "pan"
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        pos = self.mapToScene(event.pos())

        handle = self._handle_at(pos)
        if handle is not None:
            self._mode = "resize"
            self._handle = handle
            self._anchor = pos
            self._origin_rect = QRectF(self.boxes[self.selected_index]["item"].rect())
            return

        index = self._box_at(pos)
        if index >= 0:
            self.select(index)
            self._mode = "move"
            self._anchor = pos
            self._origin_rect = QRectF(self.boxes[index]["item"].rect())
            return

        self.select(-1)
        self._mode = "create"
        self._anchor = pos
        pen = QPen(class_qcolor(self.default_class_id), 2)
        pen.setCosmetic(True)
        self._draft_item = self.scene.addRect(QRectF(pos, pos), pen)

    def mouseMoveEvent(self, event):
        """Apply the pan, resize, move or draw in progress.

        Args:
            event: Qt mouse event.
        """
        if not self.image_width:
            return

        if self._mode == "pan" and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start = event.pos()
            return

        pos = self.mapToScene(event.pos())

        if self._mode is None:
            handle = self._handle_at(pos)
            if handle is not None:
                self.setCursor(HANDLE_CURSORS[handle])
            elif self._box_at(pos) >= 0:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return

        if self._mode == "resize":
            rect = QRectF(self._origin_rect)
            dx = pos.x() - self._anchor.x()
            dy = pos.y() - self._anchor.y()
            if self._handle[0] < 0:
                rect.setLeft(self._origin_rect.left() + dx)
            elif self._handle[0] > 0:
                rect.setRight(self._origin_rect.right() + dx)
            if self._handle[1] < 0:
                rect.setTop(self._origin_rect.top() + dy)
            elif self._handle[1] > 0:
                rect.setBottom(self._origin_rect.bottom() + dy)
            self._set_rect(self.selected_index, self._clamp(rect.normalized()))

        elif self._mode == "move":
            rect = QRectF(self._origin_rect)
            rect.translate(pos.x() - self._anchor.x(), pos.y() - self._anchor.y())
            rect.moveLeft(max(0.0, min(rect.left(), self.image_width - rect.width())))
            rect.moveTop(max(0.0, min(rect.top(), self.image_height - rect.height())))
            self._set_rect(self.selected_index, rect)

        elif self._mode == "create" and self._draft_item is not None:
            self._draft_item.setRect(self._clamp(QRectF(self._anchor, pos).normalized()))

    def _set_rect(self, index: int, rect: QRectF) -> None:
        """Move one box's outline and caption to a new rectangle.

        Args:
            index: Box to move.
            rect: New rectangle in image pixel coordinates.
        """
        if index < 0:
            return
        self.boxes[index]["item"].setRect(rect)
        self.boxes[index]["tag"].setPos(rect.left(), rect.top())
        self.viewport().update()

    def mouseReleaseEvent(self, event):
        """Finish the interaction in progress and save if the label changed.

        Args:
            event: Qt mouse event.
        """
        mode, self._mode = self._mode, None
        self._pan_start = None

        if mode == "pan":
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if mode in ("resize", "move"):
            # A hand-placed box is no longer described by the confidence the
            # machine had in the box it placed.
            if self.selected_index >= 0:
                self.boxes[self.selected_index]["score"] = None
                self._restyle(self.selected_index)
            self._commit("[*] Box adjusted.")
            return

        if mode == "create" and self._draft_item is not None:
            rect = self._draft_item.rect()
            self.scene.removeItem(self._draft_item)
            self._draft_item = None

            if rect.width() >= MIN_BOX_PX and rect.height() >= MIN_BOX_PX:
                index = self._add_box(self.default_class_id, rect, None)
                self.select(index)
                self._commit(f"[+] Box added as {class_name(load_classes(), self.default_class_id)}.")

    def keyPressEvent(self, event):
        """Delete the selected box, clear the selection, or set its class by number.

        Number keys are bound to classes because reclassifying is the single most
        repeated action in review: propagation places boxes well and cannot tell
        apart classes that differ only by how many of something is stacked.

        Args:
            event: Qt key event.
        """
        key = event.key()

        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            return

        if key == Qt.Key.Key_Escape:
            self.select(-1)
            return

        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            class_id = key - Qt.Key.Key_1
            if class_id < len(load_classes()):
                self.set_selected_class(class_id)
            return

        super().keyPressEvent(event)

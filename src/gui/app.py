"""Main desktop GUI for the Auto Labeling Tool.

Provides the two things the pipeline cannot do on its own: a canvas for drawing
seed boxes by hand, and a review screen where a human accepts or rejects the
borderline labels Step 4 produced.

The GUI never imports a step's internals. It launches each one as a separate
process and communicates through the same files the steps use between
themselves, so the decoupled architecture holds across the UI boundary too. The
only exception is the confidence thresholds, which are imported from Step 4 so
the numbers shown to the user cannot drift from the ones actually applied.
"""

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.class_config import class_color, class_name, load_classes, save_classes
from src.core.review_queue import accept, clear_rejections, load_queue, reject, save_queue

try:
    from src.core.step4_propagation import AUTO_ACCEPT_THRESHOLD, REVIEW_THRESHOLD
except Exception:
    AUTO_ACCEPT_THRESHOLD, REVIEW_THRESHOLD = 0.86, 0.78


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


def draw_yolo_boxes_on_pixmap(pixmap: QPixmap, label_path: str | None) -> QPixmap:
    """Draw every box in a YOLO label file onto a copy of a pixmap.

    Args:
        pixmap: Source image; it is copied rather than modified.
        label_path: Label file to read, or None to draw nothing.

    Returns:
        QPixmap: A copy of the pixmap with the boxes and class tags drawn on it.
    """
    result = QPixmap(pixmap)

    if not label_path or not os.path.exists(label_path):
        return result

    img_w, img_h = result.width(), result.height()
    if img_w == 0 or img_h == 0:
        return result

    try:
        with open(label_path) as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError:
        return result

    names = load_classes()

    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            xc, yc, w, h = (float(p) for p in parts[1:])
        except ValueError:
            continue

        box_w = w * img_w
        box_h = h * img_h
        x = (xc * img_w) - box_w / 2
        y = (yc * img_h) - box_h / 2

        color = class_qcolor(class_id)
        label = class_name(names, class_id)

        pen = QPen(color, 3)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(x, y, box_w, box_h))

        tag_width = max(70, 9 * len(label) + 16)
        tag_rect = QRectF(x, max(0, y - 22), tag_width, 20)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(tag_rect)

        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, label)

    painter.end()
    return result


class WorkerThread(QThread):
    """Runs a pipeline step in a background process and streams its output to the console."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, module_name: str):
        """Store the step to execute.

        Args:
            module_name: Dotted module path, e.g. ``src.core.step4_propagation``.
        """
        super().__init__()
        self.module_name = module_name

    def run(self):
        """Execute the step and emit each output line as it arrives."""
        try:
            # Steps are launched with -m rather than by file path. Running a file
            # puts only that file's directory on sys.path, so a step could not
            # import shared helpers such as src.core.sam_engine. The explicit cwd
            # also guarantees the relative data/... paths every step uses resolve
            # against the project root.
            process = subprocess.Popen(
                ["uv", "run", "python", "-m", self.module_name],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            for line in process.stdout:
                self.log_signal.emit(line)
            process.wait()
            self.finished_signal.emit(process.returncode)
        except Exception as e:
            self.log_signal.emit(f"\n[!] Critical Error: {str(e)}\n")
            self.finished_signal.emit(-1)


class ZoomableGraphicsView(QGraphicsView):
    """Annotation canvas with zoom, pan, crosshairs and two-stage box confirmation.

    A drawn box is not emitted immediately. It first becomes a pending box the
    user must confirm with Enter or discard with Escape, because a stray drag
    would otherwise write a seed label, and a bad seed poisons the prototype pool
    for the whole propagation run.

    Attributes:
        box_drawn_signal: Emitted with the confirmed box in scene coordinates.
        status_msg_signal: Emitted with short status text for the console panel.
    """

    box_drawn_signal = pyqtSignal(QRect)
    status_msg_signal = pyqtSignal(str)

    def __init__(self):
        """Configure the scene, render hints and interaction state."""
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#121212"))

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.setMouseTracking(True)

        self.image_item = None
        self.current_rect_item = None
        self.pending_rect_item = None
        self.start_pos = None
        self.mouse_pos = None

        self._is_panning = False
        self._pan_start = QPoint()

    def load_image(self, image_path):
        """Replace the canvas contents with an image, fitted to the viewport.

        Args:
            image_path: Path to the image to display.
        """
        self.scene.clear()
        self.current_rect_item = None
        self.pending_rect_item = None

        pixmap = QPixmap(image_path)
        self.image_item = self.scene.addPixmap(pixmap)

        self.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def drawForeground(self, painter, rect):
        """Draw the crosshair that follows the cursor.

        Args:
            painter: Painter supplied by Qt for the foreground layer.
            rect: Exposed scene rectangle to paint over.
        """
        super().drawForeground(painter, rect)

        if self.mouse_pos and not self._is_panning and self.image_item:
            pen = QPen(QColor(255, 255, 255, 150), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)

            x = self.mouse_pos.x()
            y = self.mouse_pos.y()

            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

    def wheelEvent(self, event):
        """Zoom around the cursor in response to the scroll wheel.

        Args:
            event: Qt wheel event.
        """
        if not self.image_item:
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor

        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        """Start panning on right/middle click, or start a new box on left click.

        Args:
            event: Qt mouse event.
        """
        if not self.image_item:
            return

        if event.button() in [Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton]:
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.pending_rect_item:
                self.scene.removeItem(self.pending_rect_item)
                self.pending_rect_item = None

            self.start_pos = self.mapToScene(event.pos())

            if self.current_rect_item:
                self.scene.removeItem(self.current_rect_item)

            pen = QPen(QColor(255, 0, 0), 2)
            pen.setCosmetic(True)
            self.current_rect_item = self.scene.addRect(QRectF(self.start_pos, self.start_pos), pen)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Track the cursor, pan the view, or resize the box being drawn.

        Args:
            event: Qt mouse event.
        """
        self.mouse_pos = self.mapToScene(event.pos())
        self.viewport().update()

        if self._is_panning:
            delta = event.pos() - self._pan_start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start = event.pos()
            event.accept()
            return

        if self.current_rect_item and self.start_pos and (event.buttons() & Qt.MouseButton.LeftButton):
            x = max(0, min(self.mouse_pos.x(), self.sceneRect().width()))
            y = max(0, min(self.mouse_pos.y(), self.sceneRect().height()))

            rect = QRectF(self.start_pos, QPointF(x, y)).normalized()
            self.current_rect_item.setRect(rect)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finish panning, or promote a large enough box to pending confirmation.

        Boxes smaller than a few pixels are discarded as accidental clicks.

        Args:
            event: Qt mouse event.
        """
        if event.button() in [Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton]:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton and self.current_rect_item:
            rect = self.current_rect_item.rect()

            if rect.width() > 5 and rect.height() > 5:
                self.pending_rect_item = self.current_rect_item
                pen = QPen(QColor(255, 255, 0), 2)
                pen.setCosmetic(True)
                self.pending_rect_item.setPen(pen)

                self.current_rect_item = None
                self.status_msg_signal.emit("[*] Box drawn. Press ENTER to confirm, or ESC to cancel.")
            else:
                self.scene.removeItem(self.current_rect_item)
                self.current_rect_item = None

            self.start_pos = None

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        """Confirm the pending box with Enter, or discard it with Escape.

        Args:
            event: Qt key event.
        """
        if self.pending_rect_item:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                pen = QPen(QColor(0, 255, 0), 2)
                pen.setCosmetic(True)
                self.pending_rect_item.setPen(pen)

                rect = self.pending_rect_item.rect()
                final_rect = QRect(int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

                self.box_drawn_signal.emit(final_rect)
                self.pending_rect_item = None

            elif event.key() == Qt.Key.Key_Escape:
                self.scene.removeItem(self.pending_rect_item)
                self.pending_rect_item = None
                self.viewport().update()
                self.status_msg_signal.emit("[-] Box canceled.")

        super().keyPressEvent(event)


class ReviewCardWidget(QFrame):
    """A single row in the Review Queue list: thumbnail + score + confidence bar + actions."""

    selected_signal = pyqtSignal(dict)
    accepted_signal = pyqtSignal(str)
    rejected_signal = pyqtSignal(str)

    THUMB_SIZE = 88

    def __init__(self, entry: dict, parent=None):
        """Build the card from one review queue entry.

        Args:
            entry: Queue record with the image key, score and source seed.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.entry = entry
        self.image_key = entry["image_key"]

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame { background-color: #2b2b2b; border-radius: 8px; border: 1px solid #3a3a3a; }
            QFrame:hover { border: 1px solid #0d6efd; }
        """)
        self._build_ui()

    def _build_ui(self):
        """Assemble the thumbnail, score readout, confidence bar and action buttons."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        thumb_label = QLabel()
        thumb_label.setFixedSize(self.THUMB_SIZE, self.THUMB_SIZE)
        thumb_label.setStyleSheet("background-color:#141414; border-radius:4px;")
        thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        img_path = self.entry.get("image_path")
        if img_path and os.path.exists(img_path):
            pixmap = QPixmap(img_path).scaled(
                self.THUMB_SIZE,
                self.THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            thumb_label.setPixmap(pixmap)
        else:
            thumb_label.setText("N/A")
            thumb_label.setStyleSheet(thumb_label.styleSheet() + "color:#666; font-size:10px;")
        layout.addWidget(thumb_label)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        name_label = QLabel(self.image_key)
        name_label.setStyleSheet("color:white; font-weight:bold; font-size:13px; border:none;")
        info_layout.addWidget(name_label)

        score = float(self.entry.get("score", 0.0))
        score_color = self._score_color(score)

        score_label = QLabel(f"Cosine Score: {score:.4f}")
        score_label.setStyleSheet(f"color:{score_color}; font-size:12px; font-weight:bold; border:none;")
        info_layout.addWidget(score_label)

        confidence_bar = QProgressBar()
        confidence_bar.setRange(0, 100)
        span = max(AUTO_ACCEPT_THRESHOLD - REVIEW_THRESHOLD, 1e-6)
        pct = int(max(0.0, min(1.0, (score - REVIEW_THRESHOLD) / span)) * 100)
        confidence_bar.setValue(pct)
        confidence_bar.setTextVisible(False)
        confidence_bar.setFixedHeight(6)
        confidence_bar.setStyleSheet(f"""
            QProgressBar {{ background-color:#3b3b3b; border-radius:3px; border: none; }}
            QProgressBar::chunk {{ background-color:{score_color}; border-radius:3px; }}
        """)
        info_layout.addWidget(confidence_bar)

        seed_label = QLabel(f"Source Seed: {self.entry.get('seed_source', '-')}")
        seed_label.setStyleSheet("color:#999999; font-size:11px; border:none;")
        info_layout.addWidget(seed_label)

        info_layout.addStretch()
        layout.addLayout(info_layout, stretch=1)

        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(6)

        accept_btn = QPushButton("✓ Accept")
        accept_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        accept_btn.setStyleSheet("""
            QPushButton {
                background-color:#198754; color:white; border:none;
                padding:6px 12px; border-radius:5px; font-size:12px; font-weight:bold;
            }
            QPushButton:hover { background-color:#157347; }
        """)
        accept_btn.clicked.connect(lambda: self.accepted_signal.emit(self.image_key))

        reject_btn = QPushButton("✗ Reject")
        reject_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reject_btn.setStyleSheet("""
            QPushButton {
                background-color:#dc3545; color:white; border:none;
                padding:6px 12px; border-radius:5px; font-size:12px; font-weight:bold;
            }
            QPushButton:hover { background-color:#bb2d3b; }
        """)
        reject_btn.clicked.connect(lambda: self.rejected_signal.emit(self.image_key))

        btn_layout.addWidget(accept_btn)
        btn_layout.addWidget(reject_btn)
        layout.addLayout(btn_layout)

    @staticmethod
    def _score_color(score: float) -> str:
        """Pick a colour showing how close a score is to the auto-accept threshold.

        Args:
            score: Patch similarity score of the queued match.

        Returns:
            str: Hex colour, green when nearly auto-accepted, orange when barely
            above the review threshold.
        """
        if score >= AUTO_ACCEPT_THRESHOLD - 0.02:
            return "#4caf50"
        elif score >= (REVIEW_THRESHOLD + AUTO_ACCEPT_THRESHOLD) / 2:
            return "#ffc107"
        else:
            return "#ff7043"

    def mousePressEvent(self, event):
        """Select this card so the preview pane shows its image.

        Args:
            event: Qt mouse event.
        """
        self.selected_signal.emit(self.entry)
        super().mousePressEvent(event)


class ReviewQueueDialog(QDialog):
    """Full Review Queue screen: scrollable card list and live preview pane."""

    def __init__(self, parent=None, review_queue_path: str = "data/review_queue.json"):
        """Load the queue from disk and build the review UI.

        Args:
            parent: Optional Qt parent, used to reach the console for logging.
            review_queue_path: JSON file written by the propagation step.
        """
        super().__init__(parent)
        self.setWindowTitle("Review Queue — Pending Auto-Labels")
        self.resize(1250, 780)

        self.review_queue_path = Path(review_queue_path)
        self.queue_data = self._load_queue()
        self.cards = {}
        self._selected_key = None

        self.session_accepted = 0
        self.session_rejected = 0

        self._build_ui()
        self._populate_cards()
        self._update_rejected_label()

    def _load_queue(self) -> dict:
        """Read the queue file through the shared store.

        Returns:
            dict: The queue, always containing ``pending`` and ``rejected``.
        """
        return load_queue(str(self.review_queue_path))

    def _save_queue(self) -> None:
        """Write the queue back to disk after an accept or reject."""
        save_queue(self.queue_data, str(self.review_queue_path))

    def _build_ui(self):
        """Assemble the card list, sort control, bulk actions and preview pane."""
        self.setStyleSheet("""
            QDialog { background-color:#1e1e1e; }
            QLabel { color:white; }
            QComboBox {
                background-color:#2b2b2b; color:white;
                border:1px solid #444; padding:5px; border-radius:4px;
            }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        header_layout = QHBoxLayout()
        self.count_label = QLabel()
        self.count_label.setStyleSheet("font-size:17px; font-weight:bold;")
        header_layout.addWidget(self.count_label)
        header_layout.addStretch()

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(
            [
                "Score: High → Low",
                "Score: Low → High",
                "Date: New → Old",
            ]
        )
        self.sort_combo.currentIndexChanged.connect(self._populate_cards)
        header_layout.addWidget(self.sort_combo)
        left_panel.addLayout(header_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border:none; background-color:#1e1e1e; }")
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background-color:#1e1e1e;")
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(8)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_content)
        left_panel.addWidget(self.scroll_area, stretch=1)

        bulk_layout = QHBoxLayout()
        bulk_accept_btn = QPushButton("Accept All")
        bulk_accept_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        bulk_accept_btn.setStyleSheet("""
            QPushButton {
                background-color:#198754; color:white;
                padding:9px; border-radius:5px; font-weight:bold;
            }
            QPushButton:hover { background-color:#157347; }
        """)
        bulk_accept_btn.clicked.connect(self._bulk_accept)

        bulk_reject_btn = QPushButton("Reject All")
        bulk_reject_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        bulk_reject_btn.setStyleSheet("""
            QPushButton {
                background-color:#dc3545; color:white;
                padding:9px; border-radius:5px; font-weight:bold;
            }
            QPushButton:hover { background-color:#bb2d3b; }
        """)
        bulk_reject_btn.clicked.connect(self._bulk_reject)

        bulk_layout.addWidget(bulk_accept_btn)
        bulk_layout.addWidget(bulk_reject_btn)
        left_panel.addLayout(bulk_layout)

        rejected_row = QHBoxLayout()
        self.rejected_label = QLabel("")
        self.rejected_label.setStyleSheet("color:#999999; font-size:11px;")
        rejected_row.addWidget(self.rejected_label, stretch=1)

        clear_btn = QPushButton("Clear Rejection History")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color:#3b3b3b; color:#cccccc; border:1px solid #555;
                padding:6px 10px; border-radius:5px; font-size:11px;
            }
            QPushButton:hover { background-color:#4a4a4a; }
        """)
        clear_btn.clicked.connect(self._clear_rejections)
        rejected_row.addWidget(clear_btn)
        left_panel.addLayout(rejected_row)

        main_layout.addLayout(left_panel, stretch=2)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        preview_title = QLabel("Preview (BBox automatically loaded from propagated source)")
        preview_title.setStyleSheet("font-size:15px; font-weight:bold;")
        right_panel.addWidget(preview_title)

        self.preview_label = QLabel("Select an item from the left list to preview.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("""
            background-color:#121212; border-radius:8px; color:#777; font-size:13px;
        """)
        self.preview_label.setMinimumSize(560, 480)
        right_panel.addWidget(self.preview_label, stretch=1)

        self.preview_info_label = QLabel("")
        self.preview_info_label.setStyleSheet("""
            color:#cccccc; font-size:13px; background-color:#252526;
            border-radius:6px; padding:10px;
        """)
        self.preview_info_label.setWordWrap(True)
        right_panel.addWidget(self.preview_info_label)

        main_layout.addLayout(right_panel, stretch=3)

    def _populate_cards(self):
        """Rebuild the card list from the queue in the currently selected order.

        Cards are recreated rather than reordered because the queue can change
        underneath the dialog while a propagation run is finishing.
        """
        while self.scroll_layout.count() > 1:
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.cards = {}

        entries = list(self.queue_data.get("pending", {}).values())

        sort_mode = self.sort_combo.currentIndex()
        if sort_mode == 0:
            entries.sort(key=lambda e: e.get("score", 0), reverse=True)
        elif sort_mode == 1:
            entries.sort(key=lambda e: e.get("score", 0))
        else:
            entries.sort(key=lambda e: e.get("flagged_at", ""), reverse=True)

        self.count_label.setText(f"{len(entries)} images pending review")

        if not entries:
            empty_label = QLabel("🎉 No pending images in queue. All clear!")
            empty_label.setStyleSheet("color:#4caf50; font-size:14px; padding:24px;")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.scroll_layout.insertWidget(0, empty_label)
            self.preview_label.clear()
            self.preview_label.setText("Select an item from the left list to preview.")
            self.preview_info_label.setText("")
            return

        for entry in entries:
            card = ReviewCardWidget(entry)
            card.selected_signal.connect(self._show_preview)
            card.accepted_signal.connect(self._accept_one)
            card.rejected_signal.connect(self._reject_one)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, card)
            self.cards[entry["image_key"]] = card

    def _show_preview(self, entry: dict):
        """Show the selected image with its propagated box drawn over it.

        Args:
            entry: The queue record backing the clicked card.
        """
        self._selected_key = entry.get("image_key")
        img_path = entry.get("image_path")
        label_path = entry.get("label_path")

        if not img_path or not os.path.exists(img_path):
            self.preview_label.clear()
            self.preview_label.setText("⚠ Image file not found.\n" + str(img_path))
            self.preview_info_label.setText("")
            return

        pixmap = QPixmap(img_path)
        pixmap = draw_yolo_boxes_on_pixmap(pixmap, label_path)
        scaled = pixmap.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

        score = float(entry.get("score", 0.0))
        self.preview_info_label.setText(
            f"<b>{entry.get('image_key')}</b><br>"
            f"Cosine Similarity: <b style='color:{ReviewCardWidget._score_color(score)}'>{score:.4f}</b> "
            f"&nbsp;|&nbsp; Thresholds: Review ≥ {REVIEW_THRESHOLD} · Auto ≥ {AUTO_ACCEPT_THRESHOLD}<br>"
            f"Source Seed: {entry.get('seed_source', '-')}<br>"
            f"Flagged At: {entry.get('flagged_at', '-')}"
        )

    def _accept_one(self, image_key: str):
        """Accept a queued label, leaving the file on disk and clearing the entry.

        Args:
            image_key: Key of the entry to accept.
        """
        entry = accept(self.queue_data, image_key)
        if entry is None:
            return
        self.session_accepted += 1
        self._save_queue()
        self._remove_card(image_key)
        self._log(f"[+] Accepted: {image_key} (score: {entry.get('score')})")

    def _reject_one(self, image_key: str):
        """Reject a queued label, delete its .txt file and remember the rejection.

        Deleting the label matters: one left on disk would be picked up as a seed
        prototype by the next propagation run. Recording the rejection matters
        just as much, otherwise the same image is proposed again on every run.

        Args:
            image_key: Key of the entry to reject.
        """
        entry = reject(self.queue_data, image_key)
        if entry is None:
            return
        label_path = entry.get("label_path")
        if label_path and os.path.exists(label_path):
            try:
                os.remove(label_path)
            except OSError as e:
                self._log(f"[!] Could not remove label file ({image_key}): {e}")
        self.session_rejected += 1
        self._save_queue()
        self._remove_card(image_key)
        self._log(f"[-] Rejected and label removed: {image_key} (score: {entry.get('score')})")

    def _remove_card(self, image_key: str):
        """Remove one card from the list and refresh the pending counter.

        Args:
            image_key: Key of the card to remove.
        """
        card = self.cards.pop(image_key, None)
        if card:
            card.deleteLater()

        remaining = len(self.queue_data.get("pending", {}))
        self.count_label.setText(f"{remaining} images pending review")

        if self._selected_key == image_key:
            self.preview_label.clear()
            self.preview_label.setText("Select an item from the left list to preview.")
            self.preview_info_label.setText("")
            self._selected_key = None

        if remaining == 0:
            self._populate_cards()

    def _bulk_accept(self):
        """Accept every pending entry after confirming with the user."""
        keys = list(self.queue_data.get("pending", {}).keys())
        if not keys:
            return
        confirm = QMessageBox.question(
            self,
            "Accept All",
            f"Are you sure you want to accept all {len(keys)} images?\n"
            f"(Labels are already copied; they will simply be cleared from the queue.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        for key in keys:
            accept(self.queue_data, key)
        self.session_accepted += len(keys)
        self._save_queue()
        self._populate_cards()
        self._log(f"[+] {len(keys)} images accepted in bulk.")

    def _bulk_reject(self):
        """Reject every pending entry after confirming, deleting their label files."""
        keys = list(self.queue_data.get("pending", {}).keys())
        if not keys:
            return
        confirm = QMessageBox.question(
            self,
            "Reject All",
            f"Are you sure you want to reject all {len(keys)} images?\n"
            "(Their .txt label files will be deleted and they will not be proposed "
            "again unless a later run scores them clearly higher.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        for key in keys:
            entry = reject(self.queue_data, key)
            if entry:
                label_path = entry.get("label_path")
                if label_path and os.path.exists(label_path):
                    with contextlib.suppress(OSError):
                        os.remove(label_path)
        self.session_rejected += len(keys)
        self._save_queue()
        self._populate_cards()
        self._log(f"[-] {len(keys)} images rejected in bulk and labels deleted.")

    def _clear_rejections(self):
        """Forget every past rejection so those images can be proposed again.

        Worth doing after the seed pool has grown substantially, when old
        rejections say more about the prototypes of the time than about the
        images themselves.
        """
        count = len(self.queue_data.get("rejected", {}))
        if not count:
            QMessageBox.information(self, "No Rejections", "Nothing has been rejected yet.")
            return

        confirm = QMessageBox.question(
            self,
            "Clear Rejection History",
            f"Forget {count} past rejection(s)?\n\n"
            "Those images will be proposed again by the next propagation run.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        cleared = clear_rejections(self.queue_data)
        self._save_queue()
        self._update_rejected_label()
        self._log(f"[*] Cleared {cleared} rejection(s); they can be proposed again.")

    def _update_rejected_label(self):
        """Show how many images are currently suppressed by past rejections."""
        count = len(self.queue_data.get("rejected", {}))
        self.rejected_label.setText(f"{count} image(s) suppressed by past rejections" if count else "")

    def _log(self, message: str):
        """Forward a message to the main window's console, if there is one.

        Args:
            message: Text to append, without a trailing newline.
        """
        parent = self.parent()
        if parent and hasattr(parent, "append_log"):
            parent.append_log(message + "\n")

    def closeEvent(self, event):
        """Report the session tally to the console before the dialog closes.

        Args:
            event: Qt close event.
        """
        if self.session_accepted or self.session_rejected:
            self._log(
                f"[*] Review Queue session closed. "
                f"Accepted: {self.session_accepted} | Rejected: {self.session_rejected}\n"
            )
        super().closeEvent(event)


class ClassManagerDialog(QDialog):
    """Editor for the project's object classes.

    Classes are data, not code: the same application has to label pallets in one
    project and something entirely different in the next. They are therefore
    edited here and stored in ``data/classes.json`` rather than being written
    into the source.

    A class's position in the list is its YOLO class id, so existing labels would
    silently change meaning if entries were reordered or removed from the middle.
    The dialog therefore only appends and renames, and refuses to delete anything
    other than the last entry.
    """

    def __init__(self, parent=None):
        """Load the current classes and build the editor.

        Args:
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Manage Classes")
        self.resize(460, 420)
        self.names = load_classes()
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        """Assemble the class list, the add field and the action buttons."""
        self.setStyleSheet("""
            QDialog { background-color:#1e1e1e; }
            QLabel { color:white; }
            QListWidget {
                background-color:#2b2b2b; color:white;
                border:1px solid #444; border-radius:4px; font-size:13px;
            }
            QLineEdit {
                background-color:#3b3b3b; color:white; border:1px solid #555;
                padding:6px; border-radius:3px; font-size:13px;
            }
            QPushButton {
                background-color:#0d6efd; color:white; border:none;
                padding:8px; border-radius:5px; font-size:13px;
            }
            QPushButton:hover { background-color:#0b5ed7; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header = QLabel("Object classes for this project")
        header.setStyleSheet("font-size:16px; font-weight:bold;")
        layout.addWidget(header)

        hint = QLabel(
            "A class's position is its YOLO class id, so existing labels depend on the "
            "order. Renaming is always safe; only the last class can be removed."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#999999; font-size:11px;")
        layout.addWidget(hint)

        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self._on_item_renamed)
        layout.addWidget(self.list_widget, stretch=1)

        add_row = QHBoxLayout()
        self.new_input = QLineEdit()
        self.new_input.setPlaceholderText("New class name")
        self.new_input.returnPressed.connect(self._add_class)
        add_row.addWidget(self.new_input, stretch=1)

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_class)
        add_row.addWidget(add_btn)
        layout.addLayout(add_row)

        action_row = QHBoxLayout()
        remove_btn = QPushButton("Remove Last")
        remove_btn.setStyleSheet("background-color:#dc3545;")
        remove_btn.clicked.connect(self._remove_last)
        action_row.addWidget(remove_btn)

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet("background-color:#198754;")
        save_btn.clicked.connect(self._save)
        action_row.addWidget(save_btn)
        layout.addLayout(action_row)

    def _refresh(self):
        """Rebuild the list widget from the in-memory class names."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for class_id, name in enumerate(self.names):
            item = QListWidgetItem(f"{class_id}: {name}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setForeground(class_qcolor(class_id))
            item.setData(Qt.ItemDataRole.UserRole, class_id)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

        if not self.names:
            self.list_widget.addItem("No classes defined yet.")

    def _on_item_renamed(self, item: QListWidgetItem):
        """Apply an inline rename, keeping the id prefix intact.

        Args:
            item: The edited list item.
        """
        class_id = item.data(Qt.ItemDataRole.UserRole)
        if class_id is None:
            return

        text = item.text()
        new_name = text.split(":", 1)[1].strip() if ":" in text else text.strip()
        if new_name:
            self.names[class_id] = new_name
        self._refresh()

    def _add_class(self):
        """Append the name typed in the input field as a new class."""
        name = self.new_input.text().strip()
        if not name:
            return
        if name in self.names:
            QMessageBox.warning(self, "Duplicate", f"'{name}' is already defined.")
            return
        self.names.append(name)
        self.new_input.clear()
        self._refresh()

    def _remove_last(self):
        """Remove the highest class id, after confirming with the user."""
        if not self.names:
            return
        last_id = len(self.names) - 1
        confirm = QMessageBox.question(
            self,
            "Remove Last Class",
            f"Remove class {last_id} ('{self.names[last_id]}')?\n\n"
            "Any existing label using this id will keep it and be exported under a "
            "placeholder name.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.names.pop()
            self._refresh()

    def _save(self):
        """Persist the class list and close the dialog."""
        save_classes(self.names)
        parent = self.parent()
        if parent and hasattr(parent, "append_log"):
            parent.append_log(f"[+] Saved {len(self.names)} class(es) to data/classes.json\n")
        self.accept()


class AutoLabelingApp(QMainWindow):
    """Main window: pipeline controls, annotation canvas and console output."""

    def __init__(self):
        """Build the window and show the current review queue count."""
        super().__init__()
        self.setWindowTitle("Auto Labeling Tool - Pro Canvas GUI")
        self.resize(1100, 800)
        self.current_image_path = None
        self.setup_ui()
        self.refresh_class_combo()
        self.update_review_badge()

    def setup_ui(self):
        """Lay out the sidebar, canvas and console panels."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(220)
        sidebar_frame.setStyleSheet("""
            QFrame { background-color: #2b2b2b; border-right: 1px solid #1e1e1e; }
            QLabel { color: #ffffff; font-size: 18px; font-weight: bold; }
            QPushButton {
                background-color: #0d6efd; color: white; border: none; padding: 10px;
                border-radius: 5px; font-size: 14px; margin: 5px 10px;
            }
            QPushButton:hover { background-color: #0b5ed7; }
            QPushButton:disabled { background-color: #5c636a; color: #ced4da; }
            QPushButton#loadBtn { background-color: #198754; }
            QPushButton#loadBtn:hover { background-color: #157347; }
            QPushButton#reviewBtn { background-color: #fd7e14; }
            QPushButton#reviewBtn:hover { background-color: #dc6502; }
            QPushButton#classesBtn { background-color: #6c757d; font-size: 13px; }
            QPushButton#classesBtn:hover { background-color: #5c636a; }
            QPushButton#exportBtn { background-color: #6f42c1; }
            QPushButton#exportBtn:hover { background-color: #5a32a3; }
            QComboBox {
                background-color: #3b3b3b; color: white; border: 1px solid #555;
                padding: 5px; border-radius: 3px; font-size: 14px; margin: 5px 10px;
            }
            QComboBox::drop-down { border: 0px; }
            QLineEdit {
                background-color: #3b3b3b; color: white; border: 1px solid #555;
                padding: 6px; border-radius: 3px; font-size: 14px; margin: 5px 10px;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)

        logo_label = QLabel("Auto Labeling")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo_label)

        self.class_combo = QComboBox()
        sidebar_layout.addWidget(self.class_combo)

        self.btn_classes = QPushButton("Manage Classes")
        self.btn_classes.setObjectName("classesBtn")
        self.btn_classes.clicked.connect(self.open_class_manager)
        sidebar_layout.addWidget(self.btn_classes)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter prompt (e.g., pallet)")
        sidebar_layout.addWidget(self.prompt_input)

        self.btn_load = QPushButton("Load Image")
        self.btn_load.setObjectName("loadBtn")
        self.btn_load.clicked.connect(self.open_image_dialog)
        sidebar_layout.addWidget(self.btn_load)
        sidebar_layout.addSpacing(20)

        self.buttons = []
        steps = [
            ("1. Deduplication", "src.core.step1_deduplication"),
            ("2. Embedding (VDB)", "src.core.step2_embedding"),
            ("3a. Text Prompting", "src.core.step3a_text_prompting"),
            ("3b. Manual Seeding", "src.core.step3b_manual_seeding"),
            ("4. Propagation", "src.core.step4_propagation"),
        ]

        for text, path in steps:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, p=path: self.run_script(p))
            sidebar_layout.addWidget(btn)
            self.buttons.append(btn)

        self.btn_review = QPushButton("5. Review Queue")
        self.btn_review.setObjectName("reviewBtn")
        self.btn_review.clicked.connect(self.open_review_queue)
        sidebar_layout.addWidget(self.btn_review)
        self.buttons.append(self.btn_review)

        self.btn_export = QPushButton("6. Export Dataset")
        self.btn_export.setObjectName("exportBtn")
        self.btn_export.clicked.connect(lambda: self.run_script("src.core.step5_export"))
        sidebar_layout.addWidget(self.btn_export)
        self.buttons.append(self.btn_export)

        sidebar_layout.addStretch()

        splitter = QSplitter(Qt.Orientation.Vertical)

        self.canvas = ZoomableGraphicsView()
        self.canvas.box_drawn_signal.connect(self.on_box_drawn)
        self.canvas.status_msg_signal.connect(self.append_log_newline)
        splitter.addWidget(self.canvas)

        console_frame = QFrame()
        console_frame.setStyleSheet("background-color: #1e1e1e;")
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(10, 10, 10, 10)

        console_label = QLabel("Console Output")
        console_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 14px;")
        console_layout.addWidget(console_label)

        self.log_textbox = QTextEdit()
        self.log_textbox.setReadOnly(True)
        self.log_textbox.setStyleSheet("""
            background-color: #252526; color: #cccccc; border: 1px solid #3e3e42;
            font-family: Consolas, monospace; font-size: 13px; padding: 10px;
        """)
        self.log_textbox.append("[*] System ready. Waiting for commands...\n")
        console_layout.addWidget(self.log_textbox)

        splitter.addWidget(console_frame)
        splitter.setSizes([600, 200])

        main_layout.addWidget(sidebar_frame)
        main_layout.addWidget(splitter)

    def open_image_dialog(self):
        """Ask the user for an image and load it onto the canvas."""
        default_dir = os.path.abspath("data/deduplicated")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Image", default_dir, "Images (*.png *.jpg *.jpeg)"
        )

        if file_path:
            self.current_image_path = file_path
            self.canvas.load_image(file_path)
            self.append_log(f"\n[*] Image loaded: {os.path.basename(file_path)}\n")
            self.append_log("[*] TIP: Use Scroll to Zoom, Right-Click to Pan.\n")
            self.canvas.setFocus()

    def on_box_drawn(self, rect: QRect):
        """Write a confirmed canvas box to the file Step 3b reads.

        The box is handed over as JSON rather than by calling into the step, so
        the GUI and the step stay decoupled.

        Args:
            rect: Confirmed box in image pixel coordinates.
        """
        class_id = self.selected_class_id()
        if class_id is None:
            self.append_log("[-] No classes defined. Use 'Manage Classes' to add one before drawing a box.\n")
            return

        seed_data = {
            "image_path": self.current_image_path,
            "class_id": class_id,
            "bbox": [rect.x(), rect.y(), rect.width(), rect.height()],
        }

        os.makedirs("data", exist_ok=True)
        with open("data/temp_seed.json", "w") as f:
            json.dump(seed_data, f)

        self.append_log(f"[+] Box Confirmed! Class: {self.class_combo.currentText()}\n")
        self.append_log(
            f"    Coordinates -> X:{rect.x()}, Y:{rect.y()}, W:{rect.width()}, H:{rect.height()}\n"
        )
        self.append_log("[*] Data saved! You can now click '3b. Manual Seeding' to generate the YOLO mask.\n")

    def set_buttons_state(self, enabled: bool):
        """Enable or disable every action button.

        Args:
            enabled: False while a step is running, to prevent overlapping runs.
        """
        for btn in self.buttons:
            btn.setEnabled(enabled)
        self.btn_load.setEnabled(enabled)

    def append_log_newline(self, text):
        """Append a console line, adding the trailing newline.

        Args:
            text: Message without a trailing newline.
        """
        self.append_log(text + "\n")

    def append_log(self, text):
        """Append text to the console and keep the view scrolled to the end.

        Args:
            text: Raw text to insert, newlines included.
        """
        cursor = self.log_textbox.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_textbox.setTextCursor(cursor)
        self.log_textbox.ensureCursorVisible()

    def run_script(self, module_name: str):
        """Launch a pipeline step, writing any prompt file it depends on first.

        Args:
            module_name: Dotted module path of the step to execute.
        """
        if "step3a" in module_name:
            prompt_text = self.prompt_input.text().strip()
            if not prompt_text:
                self.append_log(
                    "[-] Warning: Prompt box is empty! Please type a prompt "
                    "(e.g., 'pallet') before running Step 3a.\n"
                )
                return
            if not self.current_image_path:
                self.append_log("[-] Warning: No image loaded! Please load an image first.\n")
                return

            class_id = self.selected_class_id()
            if class_id is None:
                self.append_log("[-] No classes defined. Use 'Manage Classes' to add one first.\n")
                return

            os.makedirs("data", exist_ok=True)
            with open("data/current_prompt.json", "w") as f:
                json.dump(
                    {"prompt": prompt_text, "image_path": self.current_image_path, "class_id": class_id}, f
                )
            self.append_log(f"[*] Prompt saved for Step 3a: '{prompt_text}'\n")

        self.append_log(f"\n[{'=' * 40}]\n")
        self.append_log(f"[*] Executing: {module_name}\n")

        self.set_buttons_state(False)
        self.worker = WorkerThread(module_name)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_script_finished)
        self.worker.start()

    def on_script_finished(self, returncode):
        """Re-enable the UI and refresh the queue badge once a step exits.

        Args:
            returncode: Process exit code, zero on success.
        """
        if returncode == 0:
            self.append_log("\n[+] Process finished successfully.\n")
        else:
            self.append_log(f"\n[!] Process ended with error code: {returncode}\n")

        self.set_buttons_state(True)
        self.update_review_badge()

    def refresh_class_combo(self):
        """Rebuild the class dropdown from data/classes.json, preserving the selection.

        The dropdown is data-driven so the tool can be pointed at any project's
        classes without editing the source.
        """
        previous = self.class_combo.currentText()
        names = load_classes()

        self.class_combo.clear()
        if names:
            self.class_combo.addItems(f"{i} - {name}" for i, name in enumerate(names))
            index = self.class_combo.findText(previous)
            self.class_combo.setCurrentIndex(max(index, 0))
        else:
            self.class_combo.addItem("No classes - use 'Manage Classes'")

        self.class_combo.setEnabled(bool(names))

    def open_class_manager(self):
        """Open the class editor and refresh anything that depends on the classes."""
        ClassManagerDialog(self).exec()
        self.refresh_class_combo()
        if self.current_image_path:
            self.canvas.load_image(self.current_image_path)

    def selected_class_id(self) -> int | None:
        """Return the class id chosen in the dropdown.

        Returns:
            int | None: The selected id, or None when no classes are defined.
        """
        if not self.class_combo.isEnabled():
            return None
        return self.class_combo.currentIndex()

    def open_review_queue(self):
        """Open the review dialog and refresh the badge when it closes."""
        dialog = ReviewQueueDialog(self)
        dialog.exec()
        self.update_review_badge()

    def update_review_badge(self):
        """Show the number of pending reviews on the Review Queue button."""
        count = 0
        review_path = Path("data/review_queue.json")
        if review_path.exists():
            try:
                with open(review_path) as f:
                    data = json.load(f)
                count = len(data.get("pending", {}))
            except (json.JSONDecodeError, OSError):
                count = 0

        self.btn_review.setText(f"5. Review Queue ({count})" if count else "5. Review Queue")


if __name__ == "__main__":
    # Direct execution fallback. The supported entry point is `uv run main.py`,
    # which also anchors the working directory to the project root.
    os.chdir(PROJECT_ROOT)

    app = QApplication(sys.argv)
    window = AutoLabelingApp()
    window.show()
    sys.exit(app.exec())

"""Main desktop GUI for the Auto Labeling Tool.

Provides the two things the pipeline cannot do on its own: a canvas for drawing
seed boxes by hand, and a review screen where a human accepts or rejects the
borderline labels Step 4 produced.

The GUI never imports a step's internals. It launches each one as a separate
process and communicates through the same files the steps use between
themselves, so the decoupled architecture holds across the UI boundary too.

What it does import are the shared utilities: the class definitions, the review
queue format, the label format and the confidence thresholds. All four are
deliberately free of heavy dependencies, so this process never loads torch. It
runs no model of its own.
"""

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
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
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The src.* imports below must follow the sys.path setup above, so E402 is
# expected here. Running this file directly puts only src/gui on the path.
from src.core.class_config import (  # noqa: E402
    class_description,
    class_name,
    load_class_records,
    load_classes,
    save_class_records,
)
from src.core.dataset_summary import summarise  # noqa: E402
from src.core.review_queue import (  # noqa: E402
    SESSION_LOG_PATH,
    accept,
    add_pending,
    append_session_record,
    clear_rejections,
    load_queue,
    reject,
    save_queue,
)
from src.core.tiers import (  # noqa: E402
    AUTO_ACCEPT_THRESHOLD,
    MIN_EXAMPLES_TO_TRUST,
    REVIEW_THRESHOLD,
)
from src.core.yolo_format import (  # noqa: E402
    read_yolo_boxes,
    write_yolo_boxes,
    yolo_box_to_pixels,
)
from src.gui.label_editor import (  # noqa: E402
    LabelEditorView,
    class_qcolor,
    padded_scene_rect,
    zoom_at_cursor,
)


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
            # The child is told to write UTF-8 and its output is read as UTF-8.
            # Without both, Python decodes the pipe with the console's locale
            # codepage, which on a Turkish or Western European Windows is cp1254
            # or cp1252 and cannot represent the box-drawing characters
            # Ultralytics uses for its progress bars. Training then died on its
            # first progress bar with a UnicodeDecodeError, and only from inside
            # the GUI: run from a terminal there is no pipe and no decoding.
            #
            # errors="replace" so that one unexpected byte from any future
            # dependency shows as a question mark instead of killing a run that
            # may already be twenty minutes in.
            environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}

            process = subprocess.Popen(
                ["uv", "run", "python", "-m", self.module_name],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
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

    Confirmed boxes stay on the canvas so several objects can be marked in one
    frame, and so returning to a frame shows what is already labelled instead of
    inviting a duplicate.

    Attributes:
        box_drawn_signal: Emitted with the confirmed box in scene coordinates.
        box_removed_signal: Emitted when the user deletes the most recent box.
        status_msg_signal: Emitted with short status text for the console panel.
    """

    box_drawn_signal = pyqtSignal(QRect)
    box_removed_signal = pyqtSignal()
    status_msg_signal = pyqtSignal(str)

    def __init__(self):
        """Configure the scene, render hints and interaction state."""
        super().__init__()
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

        self.image_item = None
        self.current_rect_item = None
        self.pending_rect_item = None
        self.confirmed_items: list = []
        self.start_pos = None
        self.mouse_pos = None
        self.image_size = (0, 0)
        self._fit_scale = 1.0

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
        self.confirmed_items = []

        pixmap = QPixmap(image_path)
        self.image_item = self.scene.addPixmap(pixmap)

        self.image_size = (pixmap.width(), pixmap.height())
        # Padded so the view always has somewhere to scroll to; without that, Qt
        # centres a scene that fits the viewport and zooming cannot follow the
        # cursor. The image itself, not the padding, is what gets fitted.
        self.setSceneRect(padded_scene_rect(*self.image_size))
        self.resetTransform()
        self.fitInView(QRectF(0, 0, *self.image_size), Qt.AspectRatioMode.KeepAspectRatio)
        self._fit_scale = self.transform().m11() or 1.0
        self.centerOn(pixmap.width() / 2, pixmap.height() / 2)

    def add_confirmed_box(self, rect: QRect, class_id: int) -> None:
        """Draw a box that is already part of this image's label set.

        Args:
            rect: Box in image pixel coordinates.
            class_id: Class the box belongs to, used to colour it.
        """
        pen = QPen(class_qcolor(class_id), 2)
        pen.setCosmetic(True)
        item = self.scene.addRect(QRectF(rect), pen)
        self.confirmed_items.append(item)

    def remove_last_confirmed(self) -> bool:
        """Delete the most recently confirmed box from the canvas.

        Returns:
            bool: True when a box was removed, False when there was none.
        """
        if not self.confirmed_items:
            return False
        self.scene.removeItem(self.confirmed_items.pop())
        return True

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

        zoom_at_cursor(self, event, self._fit_scale)

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
            # Clamped to the image, not to the scene: the scene is padded so the
            # view can scroll, and a seed box must not stray into that padding.
            x = max(0, min(self.mouse_pos.x(), self.image_size[0]))
            y = max(0, min(self.mouse_pos.y(), self.image_size[1]))

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
        """Confirm with Enter, discard with Escape, delete the last box with Backspace.

        Args:
            event: Qt key event.
        """
        if self.pending_rect_item:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                rect = self.pending_rect_item.rect()
                final_rect = QRect(int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height()))

                self.scene.removeItem(self.pending_rect_item)
                self.pending_rect_item = None

                # The listener adds the box back in its class colour, so it is
                # drawn once, by whoever knows which class was selected.
                self.box_drawn_signal.emit(final_rect)

            elif event.key() == Qt.Key.Key_Escape:
                self.scene.removeItem(self.pending_rect_item)
                self.pending_rect_item = None
                self.viewport().update()
                self.status_msg_signal.emit("[-] Box canceled.")

        elif event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.box_removed_signal.emit()

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

        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("color:#bbbbbb; font-size:11px; border:none;")
        self.summary_label.setWordWrap(True)
        info_layout.addWidget(self.summary_label)
        self.refresh_summary()

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

    def refresh_summary(self) -> None:
        """Restate what the label file currently holds, per class.

        The card is a list entry, not an editor: it reports what is on disk so
        the queue can be scanned, while the boxes themselves are corrected in the
        editor pane, which is the only place big enough to judge them.
        """
        names = load_classes()
        counts: dict[int, int] = {}
        for class_id, *_ in read_yolo_boxes(self.entry.get("label_path", "")):
            counts[class_id] = counts.get(class_id, 0) + 1

        if not counts:
            # Two different frames arrive here: one the detector found nothing
            # in, and one a human emptied. Accepting either is a claim that the
            # frame really is empty, which the exporter then trains on.
            self.summary_label.setText("no boxes — accept to confirm empty, or add the missed one")
            self.summary_label.setStyleSheet("color:#e0a030; font-size:11px; border:none;")
            return

        self.summary_label.setText(
            "  ".join(f"{class_name(names, cid)} ×{count}" for cid, count in sorted(counts.items()))
        )
        self.summary_label.setStyleSheet("color:#bbbbbb; font-size:11px; border:none;")

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
        self._order: list[str] = []
        self._selected_key = None

        self.session_accepted = 0
        self.session_rejected = 0
        # Wall-clock time of each decision. What the review loop costs per frame
        # is the number that decides whether the pipeline needs a model in the
        # loop or just a faster screen, and guessing at it has no value.
        self.decision_times: list[float] = []
        # Decisions are made at a keystroke each, so a mistaken one is certain
        # rather than unlikely. Every decision is reversible until the screen is
        # closed.
        self.undo_stack: list[dict] = []
        # Every decision, in order, as "accept key" or "reject key".
        self.decided_keys: list[str] = []

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
        # Always visible rather than on demand. A queue of three hundred frames
        # gives no clue how far down it goes if the only way to move is the
        # wheel, and the bar doubles as the position indicator.
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll_area.setStyleSheet("""
            QScrollArea { border:none; background-color:#1e1e1e; }
            QScrollBar:vertical {
                background-color:#1e1e1e; width:12px; margin:0px; border:none;
            }
            QScrollBar::handle:vertical {
                background-color:#4a4a4a; border-radius:6px; min-height:30px;
            }
            QScrollBar::handle:vertical:hover { background-color:#5f5f5f; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; }
        """)
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

        editor_title = QLabel("Editor — drag to move or resize, drag on empty space to add")
        editor_title.setStyleSheet("font-size:15px; font-weight:bold;")
        right_panel.addWidget(editor_title)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(8)

        class_caption = QLabel("Selected box:")
        class_caption.setStyleSheet("color:#999999; font-size:12px;")
        tool_row.addWidget(class_caption)

        self.editor_class_combo = QComboBox()
        self.editor_class_combo.addItems(load_classes() or ["(no classes defined)"])
        self.editor_class_combo.setEnabled(False)
        self.editor_class_combo.currentIndexChanged.connect(self._on_editor_class_changed)
        tool_row.addWidget(self.editor_class_combo)

        self.delete_box_btn = QPushButton("Delete Box")
        self.delete_box_btn.setEnabled(False)
        self.delete_box_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_box_btn.setStyleSheet("""
            QPushButton {
                background-color:#3b3b3b; color:#dddddd; border:1px solid #555;
                padding:5px 12px; border-radius:5px; font-size:12px;
            }
            QPushButton:hover { background-color:#5a2b2b; }
            QPushButton:disabled { color:#666666; border-color:#3a3a3a; }
        """)
        self.delete_box_btn.clicked.connect(lambda: self.editor.delete_selected())
        tool_row.addWidget(self.delete_box_btn)
        tool_row.addStretch()

        hint = QLabel("keys: 1-9 class · Del remove · A accept · R reject · Ctrl+Z undo")
        hint.setStyleSheet("color:#777777; font-size:11px;")
        tool_row.addWidget(hint)
        right_panel.addLayout(tool_row)

        self.editor = LabelEditorView()
        self.editor.setMinimumSize(560, 440)
        self.editor.selection_changed.connect(self._on_editor_selection)
        self.editor.boxes_changed.connect(self._on_boxes_changed)
        self.editor.status_message.connect(self._log)
        right_panel.addWidget(self.editor, stretch=1)

        self.preview_info_label = QLabel("Select an item from the left list to edit it.")
        self.preview_info_label.setStyleSheet("""
            color:#cccccc; font-size:13px; background-color:#252526;
            border-radius:6px; padding:10px;
        """)
        self.preview_info_label.setWordWrap(True)
        right_panel.addWidget(self.preview_info_label)

        decision_row = QHBoxLayout()
        decision_row.setSpacing(8)

        self.accept_next_btn = QPushButton("✓ Accept && Next")
        self.accept_next_btn.setEnabled(False)
        self.accept_next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.accept_next_btn.setStyleSheet("""
            QPushButton {
                background-color:#198754; color:white; border:none;
                padding:10px; border-radius:5px; font-weight:bold;
            }
            QPushButton:hover { background-color:#157347; }
            QPushButton:disabled { background-color:#2f4a3c; color:#8a8a8a; }
        """)
        self.accept_next_btn.clicked.connect(self._accept_selected)

        self.reject_next_btn = QPushButton("✗ Reject && Next")
        self.reject_next_btn.setEnabled(False)
        self.reject_next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reject_next_btn.setStyleSheet("""
            QPushButton {
                background-color:#dc3545; color:white; border:none;
                padding:10px; border-radius:5px; font-weight:bold;
            }
            QPushButton:hover { background-color:#bb2d3b; }
            QPushButton:disabled { background-color:#4a2f33; color:#8a8a8a; }
        """)
        self.reject_next_btn.clicked.connect(self._reject_selected)

        self.undo_btn = QPushButton("↶ Undo")
        self.undo_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.undo_btn.setStyleSheet("""
            QPushButton {
                background-color:#3b3b3b; color:#dddddd; border:1px solid #555;
                padding:10px 16px; border-radius:5px; font-weight:bold;
            }
            QPushButton:hover { background-color:#4a4a4a; }
        """)
        self.undo_btn.clicked.connect(self._undo_last)

        decision_row.addWidget(self.accept_next_btn)
        decision_row.addWidget(self.reject_next_btn)
        decision_row.addWidget(self.undo_btn)
        right_panel.addLayout(decision_row)

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

        self._order = [entry["image_key"] for entry in entries]

        if not entries:
            empty_label = QLabel("🎉 No pending images in queue. All clear!")
            empty_label.setStyleSheet("color:#4caf50; font-size:14px; padding:24px;")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.scroll_layout.insertWidget(0, empty_label)
            self._clear_editor()
            return

        for entry in entries:
            card = ReviewCardWidget(entry)
            card.selected_signal.connect(self._show_preview)
            card.accepted_signal.connect(self._accept_one)
            card.rejected_signal.connect(self._reject_one)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, card)
            self.cards[entry["image_key"]] = card

    def _clear_editor(self):
        """Empty the editor pane and disable the controls that act on it."""
        self.editor.clear_image()
        self.editor_class_combo.setEnabled(False)
        self.delete_box_btn.setEnabled(False)
        self.accept_next_btn.setEnabled(False)
        self.reject_next_btn.setEnabled(False)
        self.preview_info_label.setText("Select an item from the left list to edit it.")
        self._selected_key = None

    def _show_preview(self, entry: dict):
        """Open the selected image in the editor with its propagated boxes.

        Args:
            entry: The queue record backing the clicked card.
        """
        self._selected_key = entry.get("image_key")
        img_path = entry.get("image_path")
        label_path = entry.get("label_path")

        if not img_path or not os.path.exists(img_path):
            self._clear_editor()
            self.preview_info_label.setText("⚠ Image file not found: " + str(img_path))
            return

        # Per-box scores let the editor draw the boxes propagation was unsure
        # about differently from the ones it would have accepted outright, which
        # is where a reviewer's attention belongs.
        scores = [float(s) for s in entry.get("box_scores", [])]
        self.editor.default_class_id = int(entry.get("class_id", 0))
        self.editor.load(img_path, str(label_path), scores)
        self.editor.setFocus()

        self.accept_next_btn.setEnabled(True)
        self.reject_next_btn.setEnabled(True)

        score = float(entry.get("score", 0.0))
        self.preview_info_label.setText(
            f"<b>{entry.get('image_key')}</b><br>"
            f"Best similarity: <b style='color:{ReviewCardWidget._score_color(score)}'>{score:.4f}</b> "
            f"&nbsp;|&nbsp; Thresholds: Review ≥ {REVIEW_THRESHOLD} · Auto ≥ {AUTO_ACCEPT_THRESHOLD}<br>"
            f"Source Seed: {entry.get('seed_source', '-')} &nbsp;|&nbsp; "
            f"Flagged At: {entry.get('flagged_at', '-')}"
        )

    def _on_editor_selection(self, index: int):
        """Point the class dropdown and the delete button at the selected box.

        Args:
            index: Index of the selected box, or -1 when none is selected.
        """
        selected = index >= 0
        self.delete_box_btn.setEnabled(selected)
        self.editor_class_combo.setEnabled(selected)

        if not selected:
            return

        class_id = self.editor.selected_class_id() or 0
        self.editor_class_combo.blockSignals(True)
        self.editor_class_combo.setCurrentIndex(min(class_id, self.editor_class_combo.count() - 1))
        self.editor_class_combo.blockSignals(False)

    def _on_editor_class_changed(self, class_id: int):
        """Apply the dropdown's class to the box selected in the editor.

        Args:
            class_id: Class id chosen in the dropdown.
        """
        self.editor.set_selected_class(class_id)

    def _on_boxes_changed(self):
        """Keep the card list in step with an edit made in the editor.

        The editor writes the label file itself, so nothing needs saving here.
        What does need updating is the card, which reports what the file holds.
        """
        card = self.cards.get(self._selected_key)
        if card:
            card.refresh_summary()

    def _accept_selected(self):
        """Accept the entry open in the editor and move to the next one."""
        if self._selected_key:
            self._accept_one(self._selected_key)

    def _reject_selected(self):
        """Reject the entry open in the editor and move to the next one."""
        if self._selected_key:
            self._reject_one(self._selected_key)

    def _select_after(self, image_key: str):
        """Open whichever entry followed the one just decided.

        Reviewing is a queue, not a browse: after a decision the next frame
        should already be on screen, otherwise every image costs an extra click.

        Args:
            image_key: Key of the entry that was just accepted or rejected.
        """
        pending = self.queue_data.get("pending", {})
        if not pending:
            self._clear_editor()
            return

        try:
            position = self._order.index(image_key)
        except ValueError:
            position = -1

        for key in self._order[position + 1 :] + self._order[: max(position, 0)]:
            entry = pending.get(key)
            if entry is not None:
                self._show_preview(entry)
                return

        self._clear_editor()

    def keyPressEvent(self, event):
        """Bind accept and reject to single keys so a queue can be worked quickly.

        Args:
            event: Qt key event.
        """
        if self._selected_key and event.key() == Qt.Key.Key_A:
            self._accept_selected()
            return
        if self._selected_key and event.key() == Qt.Key.Key_R:
            self._reject_selected()
            return
        if event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._undo_last()
            return
        super().keyPressEvent(event)

    def _accept_one(self, image_key: str):
        """Accept a queued label, leaving the file on disk and clearing the entry.

        Args:
            image_key: Key of the entry to accept.
        """
        entry = accept(self.queue_data, image_key)
        if entry is None:
            return
        self.undo_stack.append({"action": "accept", "entry": entry})
        self.decided_keys.append(f"accept {image_key}")
        self.decision_times.append(time.monotonic())
        self.session_accepted += 1
        self._save_queue()
        self._remove_card(image_key)
        self._log(f"[+] Accepted: {image_key} (score: {entry.get('score')})")

    def _median_seconds_per_frame(self) -> float | None:
        """Return the median gap between decisions, in seconds.

        The median rather than the mean, because a review session is not
        continuous: the reviewer answers the door, reads a message, thinks hard
        about one difficult frame. A mean over those gaps measures the
        interruptions. A median measures the work.

        Returns:
            float | None: Seconds per frame, or None before four decisions, when
            a median over two gaps would not be a median.
        """
        if len(self.decision_times) < 4:
            return None

        gaps = sorted(
            later - earlier
            for earlier, later in zip(self.decision_times, self.decision_times[1:], strict=False)
        )
        return gaps[len(gaps) // 2]

    def _pace_summary(self) -> str | None:
        """Describe how long a frame is taking and what the rest of the queue will cost.

        Returns:
            str | None: A one-line summary, or None before enough decisions have
            been made for a median to mean anything.
        """
        median = self._median_seconds_per_frame()
        if median is None:
            return None

        decisions = len(self.decision_times) - 1
        remaining = len(self.queue_data.get("pending", {}))

        summary = f"[*] Median {median:.1f} s per frame over {decisions} decision(s)."
        if remaining:
            summary += (
                f" At that pace the remaining {remaining} would take {median * remaining / 60:.0f} min."
            )
        return summary

    def _discard_outputs(self, entry: dict):
        """Delete everything propagation produced for a rejected frame.

        Deleting the label matters most: one left on disk would be picked up as a
        seed prototype by the next run, so a rejected box would come back as
        something the tool believes in. The mask is deleted for a duller reason —
        nothing else will ever remove it, and orphans accumulate silently.

        Args:
            entry: The queue record being rejected.
        """
        key = entry.get("image_key", "?")

        paths = [entry.get("label_path"), entry.get("mask_path")]
        if not entry.get("mask_path"):
            # Entries queued before masks were recorded still have one on disk.
            paths.append(os.path.join("data/masks", f"{key}.png"))

        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    self._log(f"[!] Could not remove {os.path.basename(path)} ({key}): {e}")

    def _reject_one(self, image_key: str):
        """Reject a queued label, delete what it produced and remember the rejection.

        Args:
            image_key: Key of the entry to reject.
        """
        entry = reject(self.queue_data, image_key)
        if entry is None:
            return
        # Read the boxes before they are deleted; a rejection that cannot be
        # taken back is a rejection nobody makes quickly.
        boxes = read_yolo_boxes(entry.get("label_path", ""))
        self.undo_stack.append({"action": "reject", "entry": entry, "boxes": boxes})
        self._discard_outputs(entry)
        self.decided_keys.append(f"reject {image_key}")
        self.decision_times.append(time.monotonic())
        self.session_rejected += 1
        self._save_queue()
        self._remove_card(image_key)
        self._log(f"[-] Rejected and label removed: {image_key} (score: {entry.get('score')})")

    def _undo_last(self):
        """Put the most recent decision back the way it was.

        Review runs at a keystroke per frame, so a mistaken one is not an edge
        case, it is a certainty. Accepting only cleared the queue entry, and
        rejecting also deleted the label file, which is why the boxes are read
        before that happens.

        The mask is not restored. Nothing downstream reads it — the exporter
        works from labels — and it is regenerated by the next propagation run.
        """
        if not self.undo_stack:
            self._log("[*] Nothing to undo.")
            return

        record = self.undo_stack.pop()
        entry = record["entry"]
        key = entry["image_key"]

        if record["action"] == "reject":
            self.queue_data.setdefault("rejected", {}).pop(key, None)
            label_path = entry.get("label_path")
            if label_path:
                write_yolo_boxes(label_path, record["boxes"])
            self.session_rejected = max(self.session_rejected - 1, 0)
        else:
            self.session_accepted = max(self.session_accepted - 1, 0)

        add_pending(self.queue_data, key, entry)
        if self.decision_times:
            self.decision_times.pop()
        if self.decided_keys:
            self.decided_keys.pop()

        self._save_queue()
        self._populate_cards()
        self._show_preview(entry)
        self._log(f"[*] Undone: {key} is back in the queue.")

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
            self._select_after(image_key)

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
                self._discard_outputs(entry)
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
                f"Accepted: {self.session_accepted} | Rejected: {self.session_rejected}"
            )
            pace = self._pace_summary()
            self._log(pace + "\n" if pace else "")
            self._record_session()
        super().closeEvent(event)

    def _record_session(self) -> None:
        """Persist what this sitting cost, outside the window that measured it.

        The console panel is cleared when the application closes, so the first
        time review was timed the number was lost with it. A sitting is appended
        to data/review_sessions.jsonl and echoed to stdout, both of which outlive
        the dialog.
        """
        median = self._median_seconds_per_frame()
        record = {
            "finished_at": datetime.now(UTC).isoformat(),
            "accepted": self.session_accepted,
            "rejected": self.session_rejected,
            # In decision order. Accepting a frame does not touch its label
            # file, so without this list there is no record anywhere of which
            # frames were accepted — and "which one did I just wave through?" is
            # the first thing anyone asks after a mis-keyed decision.
            "decided": self.decided_keys,
            "median_seconds_per_frame": round(median, 2) if median is not None else None,
            "active_minutes": (
                round((self.decision_times[-1] - self.decision_times[0]) / 60, 1)
                if len(self.decision_times) > 1
                else 0.0
            ),
            "still_pending": len(self.queue_data.get("pending", {})),
        }

        try:
            append_session_record(record)
        except OSError as e:
            print(f"[!] Could not write the review session record: {e}")
            return

        print(f"[+] Review session recorded in {SESSION_LOG_PATH}: {json.dumps(record)}")


class LabelEditorDialog(QDialog):
    """Correct the labels of one image, outside the review queue.

    The review screen could already do this, but only for frames a propagation
    or prediction run had queued. Once a frame is accepted it leaves the queue,
    and there was then no way to fix it short of deleting its label and starting
    again — which is a strange thing for a labelling tool to be unable to do.

    A class scheme that changes after labelling has begun makes this concrete:
    splitting one class into two means revisiting the frames that used the old
    one, and none of them are in any queue.
    """

    def __init__(self, image_path: str, label_path: str, parent=None):
        """Open one image and its label file for editing.

        Args:
            image_path: Image to display.
            label_path: YOLO label file to rewrite in place.
            parent: Optional Qt parent, used to reach the console for logging.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Edit Labels — {os.path.basename(image_path)}")
        self.resize(1000, 720)
        self.setStyleSheet("""
            QDialog { background-color:#1e1e1e; }
            QLabel { color:white; }
            QComboBox {
                background-color:#2b2b2b; color:white;
                border:1px solid #444; padding:5px; border-radius:4px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        tool_row = QHBoxLayout()
        caption = QLabel("Selected box:")
        caption.setStyleSheet("color:#999999; font-size:12px;")
        tool_row.addWidget(caption)

        self.class_combo = QComboBox()
        self.class_combo.addItems(load_classes() or ["(no classes defined)"])
        self.class_combo.setEnabled(False)
        self.class_combo.currentIndexChanged.connect(lambda i: self.editor.set_selected_class(i))
        tool_row.addWidget(self.class_combo)

        self.delete_btn = QPushButton("Delete Box")
        self.delete_btn.setEnabled(False)
        self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color:#3b3b3b; color:#dddddd; border:1px solid #555;
                padding:5px 12px; border-radius:5px; font-size:12px;
            }
            QPushButton:hover { background-color:#5a2b2b; }
            QPushButton:disabled { color:#666666; border-color:#3a3a3a; }
        """)
        self.delete_btn.clicked.connect(lambda: self.editor.delete_selected())
        tool_row.addWidget(self.delete_btn)
        tool_row.addStretch()

        hint = QLabel("keys: 1-9 class · Del remove · drag to move or resize · drag empty space to add")
        hint.setStyleSheet("color:#777777; font-size:11px;")
        tool_row.addWidget(hint)
        layout.addLayout(tool_row)

        self.editor = LabelEditorView()
        self.editor.selection_changed.connect(self._on_selection)
        self.editor.status_message.connect(self._log)
        layout.addWidget(self.editor, stretch=1)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("""
            color:#cccccc; font-size:12px; background-color:#252526;
            border-radius:6px; padding:8px;
        """)
        layout.addWidget(self.status_label)

        if self.editor.load(image_path, label_path):
            self.editor.setFocus()
            self._refresh_status()
        else:
            self.status_label.setText(f"⚠ Could not open {image_path}")

        # Every edit is written to the label file as it happens, so there is no
        # save button to forget and nothing a close can lose.
        self.editor.boxes_changed.connect(self._refresh_status)

    def _refresh_status(self):
        """Restate what the label file now holds, per class."""
        names = load_classes()
        counts: dict[int, int] = {}
        for box in self.editor.boxes:
            counts[box["class_id"]] = counts.get(box["class_id"], 0) + 1

        if not counts:
            self.status_label.setText("No boxes. This frame will be exported as a confirmed-empty image.")
            return

        summary = "  ".join(f"{class_name(names, cid)} ×{n}" for cid, n in sorted(counts.items()))
        self.status_label.setText(f"Saved: {summary}")

    def _on_selection(self, index: int):
        """Point the class dropdown and the delete button at the selected box.

        Args:
            index: Index of the selected box, or -1 when none is selected.
        """
        selected = index >= 0
        self.delete_btn.setEnabled(selected)
        self.class_combo.setEnabled(selected)
        if not selected:
            return

        self.class_combo.blockSignals(True)
        self.class_combo.setCurrentIndex(
            min(self.editor.selected_class_id() or 0, self.class_combo.count() - 1)
        )
        self.class_combo.blockSignals(False)

    def _log(self, message: str):
        """Forward a message to the main window's console, if there is one.

        Args:
            message: Text to append, without a trailing newline.
        """
        parent = self.parent()
        if parent and hasattr(parent, "append_log"):
            parent.append_log(message + "\n")


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
        self.resize(620, 500)
        self.records = load_class_records()
        self._build_ui()
        self._refresh()

    @property
    def names(self) -> list[str]:
        """Class names in class-id order.

        Returns:
            list[str]: The name of each defined class.
        """
        return [r["name"] for r in self.records]

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
            "order. Renaming is always safe; only the last class can be removed.\n"
            "Write down the rule that decides what belongs in a class. A convention "
            "nobody wrote down is the main reason datasets end up labelled inconsistently."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#999999; font-size:11px;")
        layout.addWidget(hint)

        self.list_widget = QListWidget()
        self.list_widget.itemChanged.connect(self._on_item_renamed)
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget, stretch=1)

        desc_label = QLabel("Labelling rule for the selected class")
        desc_label.setStyleSheet("color:#cccccc; font-size:12px; font-weight:bold;")
        layout.addWidget(desc_label)

        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText("e.g. Pallet carrying three stacked rows of egg trays")
        self.description_input.setFixedHeight(64)
        self.description_input.setStyleSheet("""
            background-color:#2b2b2b; color:white; border:1px solid #444;
            border-radius:4px; font-size:12px; padding:4px;
        """)
        self.description_input.textChanged.connect(self._on_description_changed)
        layout.addWidget(self.description_input)

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
        """Rebuild the list widget from the in-memory class records."""
        selected = self.list_widget.currentRow()

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for class_id, record in enumerate(self.records):
            suffix = "" if record["description"].strip() else "   (no rule written)"
            item = QListWidgetItem(f"{class_id}: {record['name']}{suffix}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setForeground(class_qcolor(class_id))
            item.setData(Qt.ItemDataRole.UserRole, class_id)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

        if not self.records:
            self.list_widget.addItem("No classes defined yet.")
            self.description_input.setEnabled(False)
            return

        self.description_input.setEnabled(True)
        self.list_widget.setCurrentRow(min(max(selected, 0), len(self.records) - 1))

    def _on_selection_changed(self, row: int):
        """Show the selected class's rule in the description box.

        Args:
            row: Index of the newly selected class, or -1 when none is selected.
        """
        if not 0 <= row < len(self.records):
            return
        self.description_input.blockSignals(True)
        self.description_input.setPlainText(self.records[row]["description"])
        self.description_input.blockSignals(False)

    def _on_description_changed(self):
        """Store edits to the rule against the class currently selected."""
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.records):
            self.records[row]["description"] = self.description_input.toPlainText().strip()

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
        new_name = new_name.replace("(no rule written)", "").strip()
        if new_name:
            self.records[class_id]["name"] = new_name
        self._refresh()

    def _add_class(self):
        """Append the name typed in the input field as a new class."""
        name = self.new_input.text().strip()
        if not name:
            return
        if name in self.names:
            QMessageBox.warning(self, "Duplicate", f"'{name}' is already defined.")
            return
        self.records.append({"name": name, "description": ""})
        self.new_input.clear()
        self._refresh()
        self.list_widget.setCurrentRow(len(self.records) - 1)
        self.description_input.setFocus()

    def _remove_last(self):
        """Remove the highest class id, after confirming with the user."""
        if not self.records:
            return
        last_id = len(self.records) - 1
        confirm = QMessageBox.question(
            self,
            "Remove Last Class",
            f"Remove class {last_id} ('{self.records[last_id]['name']}')?\n\n"
            "Any existing label using this id will keep it and be exported under a "
            "placeholder name.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.records.pop()
            self._refresh()

    def _save(self):
        """Persist the class list and close the dialog."""
        save_class_records(self.records)
        undocumented = [r["name"] for r in self.records if not r["description"].strip()]

        parent = self.parent()
        if parent and hasattr(parent, "append_log"):
            parent.append_log(f"[+] Saved {len(self.records)} class(es) to data/classes.json\n")
            if undocumented:
                parent.append_log(f"[*] No labelling rule written for: {', '.join(undocumented)}\n")
        self.accept()


class DatasetPanel(QFrame):
    """A standing count of what the dataset holds, per class.

    Every step reports what it just did and none of them says where that leaves
    the dataset, so the number that decides the next move — which class is thin,
    which has enough examples for the detector to auto-accept it — used to be
    recovered from an export log or by counting files. It belongs on screen.

    The bar under each name is the class's share of all boxes. Imbalance is
    invisible in a column of counts and is the failure mode this project has
    actually hit, so it is drawn rather than left to be worked out.
    """

    def __init__(self, parent=None):
        """Build an empty panel; call refresh() to fill it.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.row_widgets: list[QWidget] = []
        self.setObjectName("datasetPanel")
        # Maximum, not Preferred: the panel takes the height its contents need and
        # gives the rest back to the pipeline above it. Left to expand it absorbed
        # every spare pixel in the sidebar and pushed the step buttons off screen.
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.setStyleSheet("""
            QFrame#datasetPanel { background-color: #232323; border: 1px solid #383838;
                                  border-radius: 6px; margin: 0px 8px 8px 8px; }
            QLabel { font-size: 11px; font-weight: normal; }
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical { background: transparent; width: 6px; }
            QScrollBar::handle:vertical { background: #4a4a4a; border-radius: 3px; min-height: 20px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(4)

        title = QLabel("DATASET")
        title.setStyleSheet("color:#8a8a8a; font-size:10px; font-weight:bold; letter-spacing:1px;")
        layout.addWidget(title)

        self.totals_label = QLabel("")
        self.totals_label.setWordWrap(True)
        self.totals_label.setStyleSheet("color:#d0d0d0; font-size:11px;")
        layout.addWidget(self.totals_label)

        # The rows are rebuilt from scratch on every refresh rather than updated
        # in place, because the class list itself can change while the window is
        # open and a stale row is worse than a redrawn one.
        self.rows_container = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.setContentsMargins(0, 4, 0, 0)
        self.rows_layout.setSpacing(6)

        # The rows scroll rather than growing the panel. A project may define far
        # more classes than this one does, and a panel that grows to fit them
        # would push the pipeline buttons out of the window — which is the exact
        # complaint this panel was added into.
        rows_scroll = QScrollArea()
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setFrameShape(QFrame.Shape.NoFrame)
        rows_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rows_scroll.setMaximumHeight(200)
        rows_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        rows_scroll.setWidget(self.rows_container)
        layout.addWidget(rows_scroll)

        self.refresh()

    def refresh(self) -> None:
        """Recount the labels on disk and redraw the panel."""
        summary = summarise()

        labelled = summary.frames - summary.empty
        frames = "frame" if labelled == 1 else "frames"
        boxes = "box" if summary.boxes == 1 else "boxes"
        self.totals_label.setText(
            f"{summary.boxes} {boxes} in {labelled} {frames}\n{summary.empty} confirmed empty"
        )

        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.row_widgets = []

        if not summary.classes:
            hint = QLabel("No classes yet.")
            hint.setStyleSheet("color:#8a8a8a; font-size:11px;")
            self.rows_layout.addWidget(hint)
            self.row_widgets.append(hint)
        else:
            for row in summary.classes:
                widget = self._build_row(row)
                self.rows_layout.addWidget(widget)
                self.row_widgets.append(widget)

        # Without this the rows share any spare height between them and a project
        # with two classes draws two very tall rows.
        self.rows_layout.addStretch()

    def _build_row(self, row) -> QWidget:
        """Draw one class: its name, its box count and its share of the dataset.

        Args:
            row: A ClassRow from the dataset summary.

        Returns:
            QWidget: The assembled row.
        """
        widget = QWidget()
        box = QVBoxLayout(widget)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        colour = class_qcolor(row.class_id).name()

        name = QLabel(row.name)
        name.setStyleSheet(f"color:{colour}; font-size:11px;")
        header.addWidget(name)
        header.addStretch()

        # An untrusted class is not a warning about the labels, it is a statement
        # about what pre-labelling is allowed to do with them, so it is marked
        # rather than coloured like an error.
        count = QLabel(f"{row.boxes}" if row.trusted else f"{row.boxes} !")
        count.setStyleSheet(
            f"color:{'#c8c8c8' if row.trusted else '#e0a458'}; font-size:11px; font-weight:bold;"
        )
        count.setToolTip(
            f"{row.boxes} boxes, {row.share:.0%} of the dataset.\n"
            + (
                "Pre-labelling may auto-accept this class."
                if row.trusted
                else f"Fewer than {MIN_EXAMPLES_TO_TRUST} boxes, so pre-labelling sends every "
                "detection of this class to review instead of accepting it."
            )
        )
        header.addWidget(count)
        box.addLayout(header)

        # The share is drawn by giving two frames proportional stretch, so the bar
        # stays correct at any sidebar width without any pixel arithmetic.
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(0)

        filled = QFrame()
        filled.setFixedHeight(3)
        filled.setStyleSheet(f"background-color:{colour}; border:none; border-radius:1px;")
        rest = QFrame()
        rest.setFixedHeight(3)
        rest.setStyleSheet("background-color:#3a3a3a; border:none; border-radius:1px;")

        bar.addWidget(filled, max(int(row.share * 1000), 1 if row.boxes else 0))
        bar.addWidget(rest, max(int((1 - row.share) * 1000), 0))
        box.addLayout(bar)

        return widget


class AutoLabelingApp(QMainWindow):
    """Main window: pipeline controls, annotation canvas and console output."""

    def __init__(self):
        """Build the window and show the current review queue count."""
        super().__init__()
        self.setWindowTitle("Auto Labeling Tool - Pro Canvas GUI")
        self.resize(1100, 800)
        self.current_image_path = None
        self.pending_boxes: list[dict] = []
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
        sidebar_frame.setObjectName("sidebar")
        sidebar_frame.setFixedWidth(250)
        # The background and border are scoped to #sidebar rather than left on
        # QFrame, because the scroll area and the dataset panel are QFrames too
        # and would otherwise each draw their own right-hand border.
        sidebar_frame.setStyleSheet("""
            QFrame#sidebar { background-color: #2b2b2b; border-right: 1px solid #1e1e1e; }
            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            QScrollBar:vertical {
                background: transparent; width: 8px; margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #4a4a4a; border-radius: 4px; min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #5c5c5c; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
            QLabel { color: #ffffff; font-size: 18px; font-weight: bold; }
            QPushButton {
                background-color: #0d6efd; color: white; border: none; padding: 10px;
                border-radius: 5px; font-size: 14px; margin: 4px 8px;
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
            QPushButton#trainBtn { background-color: #0f8b8d; }
            QPushButton#trainBtn:hover { background-color: #0c7071; }
            QComboBox {
                background-color: #3b3b3b; color: white; border: 1px solid #555;
                padding: 5px; border-radius: 3px; font-size: 14px; margin: 4px 8px;
            }
            QComboBox::drop-down { border: 0px; }
            QLineEdit {
                background-color: #3b3b3b; color: white; border: 1px solid #555;
                padding: 6px; border-radius: 3px; font-size: 14px; margin: 4px 8px;
            }
        """)

        sidebar_outer = QVBoxLayout(sidebar_frame)
        sidebar_outer.setContentsMargins(0, 0, 0, 0)
        sidebar_outer.setSpacing(0)

        # The controls scroll. Every addition to the pipeline used to squeeze the
        # column until the lower buttons were clipped and then invisible, and
        # enlarging the window did not help because the sidebar is a fixed width
        # and the shortfall is vertical. Scrolling means the next button costs
        # nothing rather than costing the last one.
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(8, 16, 8, 12)
        sidebar_scroll.setWidget(sidebar_content)
        # Stretch 1 so the spare height goes to the scrolling controls rather than
        # to the dataset panel pinned below them.
        sidebar_outer.addWidget(sidebar_scroll, 1)

        logo_label = QLabel("Auto Labeling")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo_label)

        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self.update_class_rule)
        sidebar_layout.addWidget(self.class_combo)

        # The rule sits next to the dropdown rather than only in the class editor,
        # because the moment it is needed is while a box is being drawn.
        self.class_rule_label = QLabel("")
        self.class_rule_label.setWordWrap(True)
        self.class_rule_label.setStyleSheet(
            "color:#9fb8d4; font-size:11px; font-weight:normal; margin:0px 12px 4px 12px;"
        )
        sidebar_layout.addWidget(self.class_rule_label)

        self.btn_classes = QPushButton("Manage Classes")
        self.btn_classes.setObjectName("classesBtn")
        self.btn_classes.clicked.connect(self.open_class_manager)
        sidebar_layout.addWidget(self.btn_classes)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter prompt (e.g., pallet)")
        sidebar_layout.addWidget(self.prompt_input)

        self.btn_edit = QPushButton("Edit Labels")
        self.btn_edit.setObjectName("editBtn")
        self.btn_edit.clicked.connect(self.open_label_editor)
        sidebar_layout.addWidget(self.btn_edit)

        self.btn_load = QPushButton("Load Image")
        self.btn_load.setObjectName("loadBtn")
        self.btn_load.clicked.connect(self.open_image_dialog)
        sidebar_layout.addWidget(self.btn_load)

        self.box_count_label = QLabel("")
        self.box_count_label.setStyleSheet(
            "color:#9fd4a8; font-size:11px; font-weight:normal; margin:2px 12px 0px 12px;"
        )
        sidebar_layout.addWidget(self.box_count_label)
        sidebar_layout.addSpacing(16)

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

        self.btn_train = QPushButton("7. Train YOLO")
        self.btn_train.setObjectName("trainBtn")
        self.btn_train.clicked.connect(lambda: self.run_script("src.core.step6_train"))
        sidebar_layout.addWidget(self.btn_train)
        self.buttons.append(self.btn_train)

        # Numbered last because it needs a trained model, but it feeds the review
        # queue exactly like step 4 does. The pipeline is a loop, not a list.
        self.btn_predict = QPushButton("8. Pre-label with YOLO")
        self.btn_predict.setObjectName("predictBtn")
        self.btn_predict.clicked.connect(lambda: self.run_script("src.core.step7_predict"))
        sidebar_layout.addWidget(self.btn_predict)
        self.buttons.append(self.btn_predict)

        sidebar_layout.addStretch()

        # Pinned below the scroll area rather than inside it. It is the one thing
        # here that is read rather than clicked, and a number that has to be
        # scrolled to is a number nobody looks at.
        self.dataset_panel = DatasetPanel()
        sidebar_outer.addWidget(self.dataset_panel, 0)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self.canvas = ZoomableGraphicsView()
        self.canvas.box_drawn_signal.connect(self.on_box_drawn)
        self.canvas.box_removed_signal.connect(self.on_box_removed)
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
            self.append_log("[*] TIP: scroll to zoom, right-drag to pan, Backspace to undo a box.\n")
            self.load_existing_boxes()
            self.canvas.setFocus()

    def open_label_editor(self):
        """Open a labelled image in the editor, whether or not it is queued.

        Asks for the image rather than using whatever is on the seeding canvas,
        because the frames that need correcting are usually a list someone worked
        out elsewhere — the ones holding a class that is being split, say.
        """
        default_dir = os.path.abspath("data/deduplicated")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select an image to edit", default_dir, "Images (*.png *.jpg *.jpeg)"
        )
        if not file_path:
            return

        stem = os.path.splitext(os.path.basename(file_path))[0]
        label_path = os.path.join("data/labels", f"{stem}.txt")

        # A frame with no label file opens empty rather than being refused.
        # Drawing the first box on a fresh frame is labelling, not correcting,
        # and there is no reason the same canvas should not do both. The seeding
        # step remains the better route when the box wants SAM to tighten it;
        # this one is faster when the hand is enough.
        if not os.path.exists(label_path):
            self.append_log(f"\n[*] {stem} has no labels yet. Draw the first box.\n")
        else:
            self.append_log(f"\n[*] Editing labels for {stem}\n")
        LabelEditorDialog(file_path, label_path, parent=self).exec()
        self.dataset_panel.refresh()

        # The canvas may be showing this same frame with the boxes as they were.
        if self.current_image_path == file_path:
            self.canvas.load_image(file_path)
            self.load_existing_boxes()

    def load_existing_boxes(self):
        """Show the boxes already labelled on this image and adopt them for editing.

        Without this, returning to a frame would show a blank canvas and the next
        Step 3b run would overwrite its label file, silently discarding whatever
        was already there.
        """
        self.pending_boxes = []
        if not self.current_image_path:
            return

        stem = os.path.splitext(os.path.basename(self.current_image_path))[0]
        label_path = os.path.join("data/labels", f"{stem}.txt")
        if not os.path.exists(label_path):
            self.update_box_count()
            return

        pixmap = QPixmap(self.current_image_path)
        width, height = pixmap.width(), pixmap.height()

        for class_id, xc, yc, w, h in read_yolo_boxes(label_path):
            x0, y0, x1, y1 = yolo_box_to_pixels((xc, yc, w, h), width, height)
            rect = QRect(x0, y0, x1 - x0, y1 - y0)
            self.pending_boxes.append({"class_id": class_id, "bbox": [x0, y0, x1 - x0, y1 - y0]})
            self.canvas.add_confirmed_box(rect, class_id)

        self.append_log(f"[*] {len(self.pending_boxes)} existing box(es) loaded for this image.\n")
        self.write_seed_file()
        self.update_box_count()

    def write_seed_file(self):
        """Hand the current box list to Step 3b through data/temp_seed.json.

        The boxes travel as JSON rather than by calling into the step, so the GUI
        and the step stay decoupled.
        """
        os.makedirs("data", exist_ok=True)
        with open("data/temp_seed.json", "w") as f:
            json.dump({"image_path": self.current_image_path, "boxes": self.pending_boxes}, f)

    def update_box_count(self):
        """Show how many boxes are staged for the current image."""
        count = len(self.pending_boxes)
        self.box_count_label.setText(f"{count} box(es) on this image" if count else "")

    def on_box_drawn(self, rect: QRect):
        """Add a confirmed canvas box to this image's box list.

        Args:
            rect: Confirmed box in image pixel coordinates.
        """
        class_id = self.selected_class_id()
        if class_id is None:
            self.append_log("[-] No classes defined. Use 'Manage Classes' to add one before drawing a box.\n")
            return

        self.pending_boxes.append(
            {"class_id": class_id, "bbox": [rect.x(), rect.y(), rect.width(), rect.height()]}
        )
        self.canvas.add_confirmed_box(rect, class_id)
        self.write_seed_file()
        self.update_box_count()

        self.append_log(
            f"[+] Box {len(self.pending_boxes)} confirmed: {self.class_combo.currentText()} "
            f"at X:{rect.x()} Y:{rect.y()} W:{rect.width()} H:{rect.height()}\n"
        )
        self.append_log(
            "[*] Draw more boxes for other objects in this frame, then run '3b. Manual Seeding'.\n"
        )

    def on_box_removed(self):
        """Drop the most recently added box, in response to Backspace."""
        if not self.pending_boxes:
            return
        if not self.canvas.remove_last_confirmed():
            return

        removed = self.pending_boxes.pop()
        self.write_seed_file()
        self.update_box_count()
        self.append_log(
            f"[-] Removed the last box ({class_name(load_classes(), removed['class_id'])}). "
            f"{len(self.pending_boxes)} left.\n"
        )

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
        self.dataset_panel.refresh()

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
        self.update_class_rule()

    def update_class_rule(self):
        """Show the labelling rule for the class currently selected."""
        class_id = self.selected_class_id()
        if class_id is None:
            self.class_rule_label.setText("")
            return

        rule = class_description(load_class_records(), class_id)
        self.class_rule_label.setText(rule or "No labelling rule written for this class.")

    def open_class_manager(self):
        """Open the class editor and refresh anything that depends on the classes."""
        ClassManagerDialog(self).exec()
        self.refresh_class_combo()
        self.dataset_panel.refresh()
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
        self.dataset_panel.refresh()

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

"""
Module: app.py
Description: Main Desktop GUI for the Auto Labeling Tool.
             Features an industry-standard QGraphicsView canvas and 
             saves confirmed bounding boxes to a JSON file for Step 3b.
"""

import sys
import os
import subprocess
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFrame, QSplitter, QFileDialog,
    QGraphicsView, QGraphicsScene, QComboBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint, QRect, QRectF, QPointF
from PyQt6.QtGui import QPixmap, QPen, QColor, QPainter

class WorkerThread(QThread):
    """Runs the heavy AI scripts in the background and sends output to the GUI."""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int)

    def __init__(self, script_path):
        super().__init__()
        self.script_path = script_path

    def run(self):
        try:
            process = subprocess.Popen(
                ["uv", "run", "python", self.script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in process.stdout:
                self.log_signal.emit(line)
            process.wait()
            self.finished_signal.emit(process.returncode)
        except Exception as e:
            self.log_signal.emit(f"\n[!] Critical Error: {str(e)}\n")
            self.finished_signal.emit(-1)

class ZoomableGraphicsView(QGraphicsView):
    """Professional Canvas with Zoom, Pan, Crosshairs, and Confirmation mechanics."""
    box_drawn_signal = pyqtSignal(QRect)
    status_msg_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # High-quality rendering
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
        self.scene.clear()
        self.current_rect_item = None
        self.pending_rect_item = None
        
        pixmap = QPixmap(image_path)
        self.image_item = self.scene.addPixmap(pixmap)
        
        self.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def drawForeground(self, painter, rect):
        """Draws the crosshair lines overlay on top of everything."""
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
        """Handles zooming with the mouse wheel."""
        if not self.image_item:
            return
            
        zoom_in_factor = 1.15
        zoom_out_factor = 1.0 / zoom_in_factor
        
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
            
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if not self.image_item:
            return
            
        # Panning
        if event.button() in [Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton]:
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
            
        # Drawing
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


class AutoLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Labeling Tool - Pro Canvas GUI")
        self.resize(1100, 800)
        self.current_image_path = None
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sidebar Setup ---
        sidebar_frame = QFrame()
        sidebar_frame.setFixedWidth(220)
        sidebar_frame.setStyleSheet("""
            QFrame { background-color: #2b2b2b; border-right: 1px solid #1e1e1e; }
            QLabel { color: #ffffff; font-size: 18px; font-weight: bold; }
            QPushButton { background-color: #0d6efd; color: white; border: none; padding: 10px; border-radius: 5px; font-size: 14px; margin: 5px 10px; }
            QPushButton:hover { background-color: #0b5ed7; }
            QPushButton:disabled { background-color: #5c636a; color: #ced4da; }
            QPushButton#loadBtn { background-color: #198754; }
            QPushButton#loadBtn:hover { background-color: #157347; }
            QComboBox { background-color: #3b3b3b; color: white; border: 1px solid #555; padding: 5px; border-radius: 3px; font-size: 14px; margin: 5px 10px; }
            QComboBox::drop-down { border: 0px; }
        """)
        
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)

        logo_label = QLabel("Auto Labeling")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo_label)
        
        # Class Selector
        self.class_combo = QComboBox()
        self.class_combo.addItems(["0 - Box (Koli)", "1 - Pallet (Palet)", "2 - Other"])
        sidebar_layout.addWidget(self.class_combo)

        self.btn_load = QPushButton("Load Image")
        self.btn_load.setObjectName("loadBtn")
        self.btn_load.clicked.connect(self.open_image_dialog)
        sidebar_layout.addWidget(self.btn_load)
        sidebar_layout.addSpacing(20)

        # Core Buttons
        self.buttons = []
        steps = [
            ("1. Deduplication", "src/core/step1_deduplication.py"),
            ("2. Embedding (VDB)", "src/core/step2_embedding.py"),
            ("3a. Text Prompting", "src/core/step3a_text_prompting.py"),
            ("3b. Manual Seeding", "src/core/step3b_manual_seeding.py"),
            ("4. Propagation", "src/core/step4_propagation.py")
        ]

        for text, path in steps:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, p=path: self.run_script(p))
            sidebar_layout.addWidget(btn)
            self.buttons.append(btn)

        sidebar_layout.addStretch()

        # --- Main Workspace Setup ---
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
        selected_class_text = self.class_combo.currentText()
        class_id = int(selected_class_text.split(" - ")[0])
        
        # Prepare and save JSON data
        seed_data = {
            "image_path": self.current_image_path,
            "class_id": class_id,
            "bbox": [rect.x(), rect.y(), rect.width(), rect.height()]
        }
        
        os.makedirs("data", exist_ok=True)
        with open("data/temp_seed.json", "w") as f:
            json.dump(seed_data, f)
            
        self.append_log(f"[+] Box Confirmed! Class: {class_id} ({selected_class_text})\n")
        self.append_log(f"    Coordinates -> X:{rect.x()}, Y:{rect.y()}, W:{rect.width()}, H:{rect.height()}\n")
        self.append_log("[*] Data saved! You can now click '3b. Manual Seeding' to generate the YOLO mask.\n")

    def set_buttons_state(self, enabled: bool):
        for btn in self.buttons:
            btn.setEnabled(enabled)
        self.btn_load.setEnabled(enabled)

    def append_log_newline(self, text):
        self.append_log(text + "\n")

    def append_log(self, text):
        cursor = self.log_textbox.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_textbox.setTextCursor(cursor)
        self.log_textbox.ensureCursorVisible()

    def run_script(self, script_path):
        self.append_log(f"\n[{'='*40}]\n")
        self.append_log(f"[*] Executing: {script_path}\n")
        
        self.set_buttons_state(False)
        self.worker = WorkerThread(script_path)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_script_finished)
        self.worker.start()

    def on_script_finished(self, returncode):
        if returncode == 0:
            self.append_log(f"\n[+] Process finished successfully.\n")
        else:
            self.append_log(f"\n[!] Process ended with error code: {returncode}\n")
            
        self.set_buttons_state(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoLabelingApp()
    window.show()
    sys.exit(app.exec())
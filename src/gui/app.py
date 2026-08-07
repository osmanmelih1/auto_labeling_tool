"""
Module: app.py
Description: Main Desktop GUI for the Auto Labeling Tool.
             Features an interactive Canvas with precise coordinate mapping (scaling fix)
             for drawing bounding boxes, and a background thread for running AI scripts.
"""

import sys
import os
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLabel, QFrame, QSplitter, QFileDialog
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint, QRect
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor

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

class InteractiveCanvas(QLabel):
    """Custom Label that handles image display and precise mouse coordinate mapping."""
    box_drawn_signal = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("No Image Loaded\n(Click 'Load Image' to start)")
        self.setStyleSheet("background-color: #121212; color: #555555; font-size: 20px;")
        
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.original_pixmap = None

    def load_image(self, image_path):
        self.original_pixmap = QPixmap(image_path)
        self.update_display()

    def update_display(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            scaled_pixmap = self.original_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)

    def get_image_metrics(self):
        """Calculates scaling factors and offsets to map widget coordinates to real image pixels."""
        if not self.original_pixmap:
            return 0, 0, 1.0, 1.0, 0, 0
            
        lbl_w, lbl_h = self.width(), self.height()
        orig_w, orig_h = self.original_pixmap.width(), self.original_pixmap.height()
        
        scaled_size = self.original_pixmap.scaled(
            lbl_w, lbl_h, Qt.AspectRatioMode.KeepAspectRatio
        ).size()
        
        s_w, s_h = scaled_size.width(), scaled_size.height()
        
        # Calculate black borders (offsets)
        x_offset = (lbl_w - s_w) // 2
        y_offset = (lbl_h - s_h) // 2
        
        # Calculate scale ratio (Real / Scaled)
        scale_x = orig_w / s_w if s_w > 0 else 1.0
        scale_y = orig_h / s_h if s_h > 0 else 1.0
        
        return x_offset, y_offset, scale_x, scale_y, s_w, s_h

    def clamp_point(self, point, x_offset, y_offset, s_w, s_h):
        """Prevents drawing outside the actual image area (black borders)."""
        x = max(x_offset, min(point.x(), x_offset + s_w))
        y = max(y_offset, min(point.y(), y_offset + s_h))
        return QPoint(x, y)

    def mousePressEvent(self, event):
        if self.original_pixmap and event.button() == Qt.MouseButton.LeftButton:
            x_off, y_off, _, _, s_w, s_h = self.get_image_metrics()
            self.drawing = True
            self.start_point = self.clamp_point(event.pos(), x_off, y_off, s_w, s_h)
            self.end_point = self.start_point

    def mouseMoveEvent(self, event):
        if self.drawing and self.original_pixmap:
            x_off, y_off, _, _, s_w, s_h = self.get_image_metrics()
            self.end_point = self.clamp_point(event.pos(), x_off, y_off, s_w, s_h)
            self.draw_temp_box()

    def mouseReleaseEvent(self, event):
        if self.drawing and self.original_pixmap and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            x_off, y_off, scale_x, scale_y, s_w, s_h = self.get_image_metrics()
            self.end_point = self.clamp_point(event.pos(), x_off, y_off, s_w, s_h)
            self.draw_temp_box()
            
            # --- THE FIX: Map widget coordinates directly to ORIGINAL image pixels ---
            real_start_x = (self.start_point.x() - x_off) * scale_x
            real_start_y = (self.start_point.y() - y_off) * scale_y
            real_end_x = (self.end_point.x() - x_off) * scale_x
            real_end_y = (self.end_point.y() - y_off) * scale_y
            
            real_rect = QRect(
                QPoint(int(real_start_x), int(real_start_y)), 
                QPoint(int(real_end_x), int(real_end_y))
            ).normalized()
            
            self.box_drawn_signal.emit(real_rect)

    def draw_temp_box(self):
        """Draws the visual red box on the UI."""
        if not self.original_pixmap:
            return
            
        scaled_pixmap = self.original_pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        
        painter = QPainter(scaled_pixmap)
        pen = QPen(QColor(255, 0, 0), 3, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        
        # Adjust drawing coordinates to the scaled pixmap (removing the label offsets)
        x_off, y_off, _, _, _, _ = self.get_image_metrics()
        draw_start = QPoint(self.start_point.x() - x_off, self.start_point.y() - y_off)
        draw_end = QPoint(self.end_point.x() - x_off, self.end_point.y() - y_off)
        
        rect = QRect(draw_start, draw_end)
        painter.drawRect(rect)
        painter.end()
        
        self.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_display()


class AutoLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Labeling Tool - Interactive GUI (Calibrated)")
        self.resize(1100, 800)
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
        """)
        sidebar_layout = QVBoxLayout(sidebar_frame)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)

        logo_label = QLabel("Auto Labeling")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo_label)
        sidebar_layout.addSpacing(20)

        self.btn_load = QPushButton("Load Image")
        self.btn_load.setObjectName("loadBtn")
        self.btn_load.clicked.connect(self.open_image_dialog)
        sidebar_layout.addWidget(self.btn_load)
        sidebar_layout.addSpacing(20)

        # --- Core Buttons Configuration ---
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
        
        self.canvas = InteractiveCanvas()
        self.canvas.box_drawn_signal.connect(self.on_box_drawn)
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
        splitter.setSizes([500, 200])

        main_layout.addWidget(sidebar_frame)
        main_layout.addWidget(splitter)

    def open_image_dialog(self):
        default_dir = os.path.abspath("data/deduplicated")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Image", default_dir, "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.canvas.load_image(file_path)
            self.append_log(f"\n[*] Image loaded into canvas: {os.path.basename(file_path)}\n")

    def on_box_drawn(self, rect: QRect):
        self.append_log(f"[*] REAL Original Image Coordinates -> X:{rect.x()}, Y:{rect.y()}, Width:{rect.width()}, Height:{rect.height()}\n")
        self.append_log("[*] (Ready to be sent to Step 3b...)\n")

    def set_buttons_state(self, enabled: bool):
        for btn in self.buttons:
            btn.setEnabled(enabled)
        self.btn_load.setEnabled(enabled)

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
"""Shared fixtures for the test suite.

Two things every test here needs. A Qt application, because most of what is
worth testing in this project is in the two canvases and neither can be
instantiated without one. And an empty project directory, because the pipeline
addresses its data with relative ``data/...`` paths by design, so a test that
did not move out of the real project would read the operator's actual classes
and could delete their actual labels.

The offscreen platform is selected before Qt is imported anywhere, which is why
it is set here rather than in a test file.
"""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt6.QtGui import QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.core.yolo_format import write_yolo_boxes  # noqa: E402

# Fixed classes, so a test asserting on a caption does not depend on whatever
# project the checkout happens to be configured for.
TEST_CLASSES = [
    {"name": "pallet_1", "description": "one row"},
    {"name": "pallet_2", "description": "two rows"},
    {"name": "pallet_3", "description": "three rows"},
    {"name": "carton", "description": "cardboard rather than trays"},
    {"name": "irregular", "description": "anything else"},
]


@pytest.fixture(scope="session")
def qapp():
    """Provide the one QApplication the whole session shares.

    Qt permits only one per process, and creating it per test leaks native
    resources until the run falls over.

    Returns:
        QApplication: The running application object.
    """
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def project_sandbox(tmp_path, monkeypatch):
    """Run each test inside a throwaway project directory.

    Args:
        tmp_path: Pytest's per-test temporary directory.
        monkeypatch: Pytest's patching fixture, used for the working directory.

    Returns:
        pathlib.Path: Root of the sandboxed project.
    """
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "classes.json").write_text(
        json.dumps({"classes": TEST_CLASSES}, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def make_frame(project_sandbox):
    """Return a factory that writes an image and its label file.

    Args:
        project_sandbox: The sandboxed project root.

    Returns:
        callable: ``make_frame(key, boxes, size=(320, 240))`` returning
        ``(image_path, label_path)`` as strings.
    """

    def factory(key: str, boxes: list, size: tuple[int, int] = (320, 240)) -> tuple[str, str]:
        image_path = project_sandbox / f"{key}.png"
        QImage(size[0], size[1], QImage.Format.Format_RGB32).save(str(image_path))
        label_path = project_sandbox / f"{key}.txt"
        write_yolo_boxes(str(label_path), boxes)
        return str(image_path), str(label_path)

    return factory

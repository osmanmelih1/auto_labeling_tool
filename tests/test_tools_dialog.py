"""The dialog that puts the inspection tools inside the application.

What matters here is the handover: the dialog decides on a command and the main
window runs it through the same worker as a pipeline step. If that seam is
wrong the tools appear to do nothing, which is indistinguishable from them
having run and found nothing.
"""

import json

import pytest

pytest.importorskip("PyQt6")

from src.gui.app import ToolsDialog, open_folder  # noqa: E402
from src.gui.tool_catalog import find_tool  # noqa: E402


@pytest.fixture
def classes(project_sandbox):
    """Define a class scheme for the dialog's dropdowns to read.

    Args:
        project_sandbox: The sandboxed project root.

    Returns:
        pathlib.Path: The sandboxed project root.
    """
    (project_sandbox / "data" / "classes.json").write_text(
        json.dumps({"classes": ["palet_1li", "koli"]}), encoding="utf-8"
    )
    return project_sandbox


def test_running_a_tool_records_the_command_rather_than_launching_it(classes, qapp):
    """The dialog closes before the tool starts so its output reaches the console.

    Were the dialog to run the subprocess itself, the log would stream into a
    window that is being dismissed.

    Args:
        classes: The sandboxed project root with classes defined.
        qapp: The shared QApplication.
    """
    dialog = ToolsDialog()

    dialog._run(find_tool("preview"), None, apply=False)

    assert dialog.command == ("src.tools.preview_labels", [])


def test_a_tool_needing_a_class_is_refused_rather_than_started(classes, qapp, monkeypatch):
    """An argparse failure inside a subprocess reads like the tool is broken.

    Args:
        classes: The sandboxed project root with classes defined.
        qapp: The shared QApplication.
        monkeypatch: Used to silence the warning dialog.
    """
    warned = []
    monkeypatch.setattr("src.gui.app.QMessageBox.warning", lambda *args, **kwargs: warned.append(args[2]))

    dialog = ToolsDialog()
    dialog._run(find_tool("find"), None, apply=False)

    assert dialog.command is None
    assert warned and "needs a class" in warned[0]


def test_clearing_the_project_does_nothing_without_confirmation(classes, qapp, monkeypatch):
    """Declining the confirmation must leave no command behind to be run later.

    Args:
        classes: The sandboxed project root with classes defined.
        qapp: The shared QApplication.
        monkeypatch: Used to answer the confirmation with No.
    """
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        "src.gui.app.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )

    dialog = ToolsDialog()
    dialog._run(find_tool("new_project"), None, apply=True)

    assert dialog.command is None


def test_confirming_produces_the_apply_flag(classes, qapp, monkeypatch):
    """And answering Yes has to actually reach the tool, or the button is a lie.

    Args:
        classes: The sandboxed project root with classes defined.
        qapp: The shared QApplication.
        monkeypatch: Used to answer the confirmation with Yes.
    """
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        "src.gui.app.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    dialog = ToolsDialog()
    dialog._run(find_tool("new_project"), None, apply=True)

    assert dialog.command == ("src.tools.new_project", ["--apply"])


def test_opening_a_folder_that_does_not_exist_is_declined_not_attempted(classes, qapp):
    """A tool that has never run has no output directory, and the button says so.

    Args:
        classes: The sandboxed project root with classes defined.
        qapp: The shared QApplication.
    """
    assert open_folder("data/never_written_to") is False

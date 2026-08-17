"""Turning a tool and a dropdown into a command line.

The tools are launched as subprocesses, so a wrong argument is not a type error
here — it is a traceback in the console minutes later, or worse, a destructive
tool doing what it says on the tin. The building of the argument list is
therefore tested away from the dialog.
"""

import pytest

from src.gui.tool_catalog import TOOLS, build_command, find_tool


def test_a_class_filter_becomes_the_flag_the_tool_expects():
    """audit_labels and preview_labels take --class; getting it wrong aborts the run."""
    assert build_command(find_tool("audit"), "duzensiz_istif") == ["--class", "duzensiz_istif"]
    assert build_command(find_tool("preview"), "koli") == ["--class", "koli"]


def test_a_class_becomes_a_positional_where_the_tool_wants_one():
    """find_class_examples takes the class as its first argument, not as a flag."""
    assert build_command(find_tool("find"), "palet_1li") == ["palet_1li"]


def test_no_class_means_every_class():
    """The dialog's "All classes" entry has to produce no filter at all, not an empty one."""
    assert build_command(find_tool("audit"), None) == []
    assert build_command(find_tool("audit"), "") == []


def test_a_tool_that_needs_a_class_refuses_to_run_without_one():
    """Searching for examples of no particular class is not a request anyone makes.

    Without this the subprocess starts and argparse fails, which reads in the
    console like the tool is broken.
    """
    with pytest.raises(ValueError, match="needs a class"):
        build_command(find_tool("find"), None)


def test_apply_is_only_ever_passed_to_a_tool_declared_destructive():
    """--apply on the wrong module is the one mistake here that destroys work.

    The guard is in build_command rather than only in the dialog, so a future
    caller cannot reach past the confirmation and delete a project.
    """
    assert build_command(find_tool("new_project"), None, apply=True) == ["--apply"]
    assert build_command(find_tool("audit"), None, apply=True) == []


def test_reporting_is_the_default_for_the_destructive_tool():
    """Its dry run has to be what an unmodified click produces."""
    assert build_command(find_tool("new_project"), None) == []


def test_every_tool_is_reachable_by_its_key():
    """The dialog and these tests address tools by key, so a typo must fail loudly."""
    for tool in TOOLS:
        assert find_tool(tool.key) is tool

    with pytest.raises(KeyError):
        find_tool("no_such_tool")


def test_the_catalog_agrees_with_the_tools_own_output_directories():
    """The catalog repeats these paths instead of importing them, and drift is silent.

    Importing the tools would pull torch and ultralytics into the GUI process,
    which is why the strings are duplicated. This test is the price of that: it
    imports the tools, which the GUI never does, and checks the copies still
    match. A tool that moved its output would otherwise leave the "Open output
    folder" button opening yesterday's previews — a mistake already made once by
    hand.
    """
    pytest.importorskip("PIL")
    from src.tools import audit_labels, preview_labels

    assert find_tool("audit").output_dir == audit_labels.OUTPUT_DIR
    assert find_tool("preview").output_dir == preview_labels.OUTPUT_DIR

    pytest.importorskip("ultralytics")
    from src.tools import find_class_examples

    assert find_tool("find").output_dir == find_class_examples.OUTPUT_DIR

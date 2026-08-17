"""The inspection tools the GUI is allowed to launch, and how to call them.

The tools under ``src/tools`` were reachable only from a terminal, which meant a
labelling session ran in two windows and a demonstration of the pipeline had a
PowerShell prompt in the middle of it. This is the list the tool dialog is built
from.

The module paths and output directories are written out here rather than
imported from the tools themselves. Importing them would be more honest but
would pull ``torch`` and ``ultralytics` into the GUI's own process — which is
why ``find_latest_weights`` is imported inside ``main()`` over there in the first
place. The duplication is kept in check by a test that compares these strings
against the tools' own constants.

Nothing here runs anything. It builds an argument list; the GUI runs it through
the same worker thread the pipeline steps use, so tool output lands in the same
console.
"""

from dataclasses import dataclass

CLASS_FLAG = "flag"
CLASS_POSITIONAL = "positional"


@dataclass(frozen=True)
class Tool:
    """One launchable inspection tool.

    Attributes:
        key: Stable identifier, used by tests and by the dialog's button map.
        label: Name shown in the dialog.
        module: Dotted module path to run with ``python -m``.
        blurb: One or two sentences on what it answers and what it costs.
        output_dir: Where it writes, so the dialog can offer to open it, or None
            when the tool only prints.
        class_argument: How it takes a class name — as ``--class NAME``, as a
            positional argument, or not at all.
        class_required: Whether the tool refuses to run without a class.
        destructive: Whether running it with ``--apply`` deletes work.
        slow: Whether it loads a model, and so takes minutes rather than seconds.
    """

    key: str
    label: str
    module: str
    blurb: str
    output_dir: str | None = None
    class_argument: str | None = None
    class_required: bool = False
    destructive: bool = False
    slow: bool = False


TOOLS: list[Tool] = [
    Tool(
        key="audit",
        label="Audit labels",
        module="src.tools.audit_labels",
        blurb=(
            "Runs the trained detector over every labelled box and shortlists the ones it "
            "disagrees with. A clean audit means no new contradictions, never that the labels "
            "are right — a mistake made the same way every time is one the model has learned."
        ),
        output_dir="data/audit",
        class_argument=CLASS_FLAG,
        slow=True,
    ),
    Tool(
        key="preview",
        label="Preview labels",
        module="src.tools.preview_labels",
        blurb=(
            "Draws the current boxes onto the frames so they can be checked by eye. "
            "No model, no GPU: this is the fastest way to see what a class actually contains."
        ),
        output_dir="data/previews",
        class_argument=CLASS_FLAG,
    ),
    Tool(
        key="find",
        label="Find class examples",
        module="src.tools.find_class_examples",
        blurb=(
            "Searches unlabelled frames for a scarce class and proposes candidates. "
            "It proposes only — nothing is written to data/labels. Note that it needs a model "
            "that already knows the class, so a class with a handful of boxes is hard to grow "
            "this way."
        ),
        output_dir="data/candidates",
        class_argument=CLASS_POSITIONAL,
        class_required=True,
        slow=True,
    ),
    Tool(
        key="gpu",
        label="Check GPU",
        module="src.tools.gpu_check",
        blurb="Reports whether CUDA is visible and how fast this machine actually is.",
    ),
    Tool(
        key="new_project",
        label="Start a new project",
        module="src.tools.new_project",
        blurb=(
            "Empties this project's data so the tool can be pointed at another dataset, keeping "
            "the ~700 MB of model weights. Reports by default and deletes only on confirmation. "
            "Copy the trained checkpoint out of runs/ first: it is the point of the exercise and "
            "it is not in Git."
        ),
        destructive=True,
    ),
]


def build_command(tool: Tool, class_name: str | None = None, apply: bool = False) -> list[str]:
    """Turn a tool and the dialog's choices into arguments for ``python -m``.

    Args:
        tool: The tool to run.
        class_name: Class to restrict to, or None for all classes.
        apply: Whether a destructive tool should act rather than report.

    Returns:
        list[str]: Arguments to append after the module path.

    Raises:
        ValueError: If a tool that requires a class was given none.
    """
    arguments: list[str] = []

    if tool.class_required and not class_name:
        raise ValueError(f"{tool.label} needs a class to look for.")

    if class_name and tool.class_argument == CLASS_FLAG:
        arguments += ["--class", class_name]
    elif class_name and tool.class_argument == CLASS_POSITIONAL:
        arguments.append(class_name)

    # --apply is only ever added for a tool declared destructive, so a future
    # tool cannot be made to delete by passing apply=True to the wrong one.
    if apply and tool.destructive:
        arguments.append("--apply")

    return arguments


def find_tool(key: str) -> Tool:
    """Look a tool up by its key.

    Args:
        key: The tool's stable identifier.

    Returns:
        Tool: The matching tool.

    Raises:
        KeyError: If no tool has that key.
    """
    for tool in TOOLS:
        if tool.key == key:
            return tool
    raise KeyError(key)

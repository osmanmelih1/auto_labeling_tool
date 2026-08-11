"""Shared class-definition store.

This is a utility, not a pipeline step.

The tool is meant to label any dataset, so the set of object classes belongs to
the data, not to the source code. Class names therefore live in a JSON file under
``data/`` that the user edits through the GUI. Nothing in the codebase names a
concrete class: labelling pallets and labelling tumours must be the same program
with a different ``data/classes.json``.

The file is a plain list where a class's position is its YOLO class id, which is
the same convention the exported ``data.yaml`` uses::

    {"classes": ["pallet", "forklift"]}

    -> class id 0 is "pallet", class id 1 is "forklift"

Reordering the list therefore renumbers existing labels, so the GUI only ever
appends and renames.
"""

import colorsys
import json
import os
from pathlib import Path

CLASSES_FILE = "data/classes.json"

# Golden-ratio hue stepping keeps consecutive class colours far apart for any
# number of classes, so palettes never have to be hardcoded per project.
_HUE_STEP = 0.618033988749895
_HUE_OFFSET = 0.08


def load_classes(path: str = CLASSES_FILE) -> list[str]:
    """Read the class names, returning an empty list when none are defined yet.

    A missing or malformed file is not an error: a fresh checkout legitimately
    has no classes until the user defines some.

    Args:
        path: Location of the class definition file.

    Returns:
        list[str]: Class names, where the index is the YOLO class id.
    """
    if not os.path.exists(path):
        return []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[!] Could not read {path} ({e}). Treating it as empty.")
        return []

    names = data.get("classes", []) if isinstance(data, dict) else data
    if not isinstance(names, list):
        print(f"[!] {path} does not contain a list of classes. Treating it as empty.")
        return []

    return [str(name) for name in names]


def save_classes(names: list[str], path: str = CLASSES_FILE) -> None:
    """Write the class names, creating the parent directory if needed.

    Args:
        names: Class names in class-id order.
        path: Location of the class definition file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"classes": names}, f, indent=2, ensure_ascii=False)


def class_name(names: list[str], class_id: int) -> str:
    """Return a display name for a class id, even one with no definition.

    Label files can legitimately reference an id whose name was never defined,
    for example after a class was removed. Those ids still need to be shown and
    exported rather than crashing the caller.

    Args:
        names: Class names in class-id order.
        class_id: The id to name.

    Returns:
        str: The defined name, or a generated placeholder.
    """
    if 0 <= class_id < len(names):
        return names[class_id]
    return f"class_{class_id}"


def class_color(class_id: int) -> tuple[int, int, int]:
    """Generate a distinct RGB colour for a class id.

    Colours are computed rather than stored so that any number of classes gets a
    readable palette without a per-project colour table.

    Args:
        class_id: The class id to colour.

    Returns:
        tuple: ``(red, green, blue)`` in the range 0-255.
    """
    hue = (_HUE_OFFSET + class_id * _HUE_STEP) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)

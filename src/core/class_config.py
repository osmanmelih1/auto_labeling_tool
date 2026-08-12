"""Shared class-definition store.

This is a utility, not a pipeline step.

The tool is meant to label any dataset, so the set of object classes belongs to
the data, not to the source code. Class names therefore live in a JSON file under
``data/`` that the user edits through the GUI. Nothing in the codebase names a
concrete class: labelling pallets and labelling tumours must be the same program
with a different ``data/classes.json``.

A class's position in the file is its YOLO class id, which is the same convention
the exported ``data.yaml`` uses. Reordering therefore renumbers every existing
label, so the GUI only ever appends and renames.

Each class may also carry a description: the rule that decides whether an object
belongs to it. Where a class boundary is not obvious from its name, that rule
lives here rather than in the labeller's head, because a convention nobody wrote
down is the main reason datasets end up labelled inconsistently::

    {"classes": [
        {"name": "palet_3lu", "description": "Three rows of trays on the pallet"},
        {"name": "koli",      "description": "Cardboard boxes on a pallet"}
    ]}

The plain form is still accepted and still means the same thing::

    {"classes": ["pallet", "forklift"]}
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


def load_class_records(path: str = CLASSES_FILE) -> list[dict]:
    """Read the full class definitions, names and descriptions together.

    Entries may be plain strings or objects; both are normalised to a dict with
    ``name`` and ``description`` keys so callers never have to check which form
    the file happens to use.

    Args:
        path: Location of the class definition file.

    Returns:
        list[dict]: One ``{"name": ..., "description": ...}`` per class, in
        class-id order.
    """
    if not os.path.exists(path):
        return []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[!] Could not read {path} ({e}). Treating it as empty.")
        return []

    entries = data.get("classes", []) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        print(f"[!] {path} does not contain a list of classes. Treating it as empty.")
        return []

    records = []
    for entry in entries:
        if isinstance(entry, dict):
            records.append(
                {
                    "name": str(entry.get("name", "")),
                    "description": str(entry.get("description", "")),
                }
            )
        else:
            records.append({"name": str(entry), "description": ""})

    return records


def save_class_records(records: list[dict], path: str = CLASSES_FILE) -> None:
    """Write class definitions, keeping descriptions where they exist.

    Classes without a description are written in the plain string form so a file
    that never needed descriptions stays as simple as it started.

    Args:
        records: One ``{"name": ..., "description": ...}`` per class.
        path: Location of the class definition file.
    """
    entries: list = []
    for record in records:
        description = record.get("description", "").strip()
        if description:
            entries.append({"name": record["name"], "description": description})
        else:
            entries.append(record["name"])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"classes": entries}, f, indent=2, ensure_ascii=False)


def class_description(records: list[dict], class_id: int) -> str:
    """Return the labelling rule recorded for a class, if any.

    Args:
        records: Class records in class-id order.
        class_id: The id to describe.

    Returns:
        str: The description, or an empty string when none was written.
    """
    if 0 <= class_id < len(records):
        return records[class_id].get("description", "")
    return ""


def load_classes(path: str = CLASSES_FILE) -> list[str]:
    """Read just the class names, for callers that do not need the descriptions.

    A missing or malformed file is not an error: a fresh checkout legitimately
    has no classes until the user defines some.

    Args:
        path: Location of the class definition file.

    Returns:
        list[str]: Class names, where the index is the YOLO class id.
    """
    return [record["name"] for record in load_class_records(path)]


def save_classes(names: list[str], path: str = CLASSES_FILE) -> None:
    """Write class names, preserving any descriptions already on disk.

    Renaming a class through this function must not silently discard the rule
    that says what belongs in it, so existing descriptions are carried over by
    position.

    Args:
        names: Class names in class-id order.
        path: Location of the class definition file.
    """
    existing = load_class_records(path)
    records = [
        {
            "name": name,
            "description": existing[i]["description"] if i < len(existing) else "",
        }
        for i, name in enumerate(names)
    ]
    save_class_records(records, path)


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

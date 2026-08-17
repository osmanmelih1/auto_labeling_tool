"""What the dataset holds right now, per class.

This is the question the operator asks most often and the pipeline answered
least directly. Every step prints what it just did; none of them says where the
dataset stands afterwards, so the count that decides the next move — which class
is thin, which has earned the detector's trust — had to be recovered by reading
the class distribution out of an export log or counting label files by hand.

The arithmetic lives here rather than in the GUI so it can be tested without a
running Qt application, and so the numbers on screen come from the same place
the pipeline's own thresholds do.

Reads ``data/labels/`` and ``data/classes.json``; writes nothing.
"""

from dataclasses import dataclass
from pathlib import Path

from src.core.class_config import CLASSES_FILE, load_classes
from src.core.tiers import MIN_EXAMPLES_TO_TRUST
from src.core.yolo_format import count_boxes_per_class


@dataclass(frozen=True)
class ClassRow:
    """One class's standing in the dataset.

    Attributes:
        class_id: Position in the class list, which is its YOLO id.
        name: Class name.
        boxes: How many boxes carry this class.
        trusted: Whether it has enough examples for the detector to auto-accept
            it during pre-labelling.
        share: Fraction of all boxes belonging to this class, 0.0 to 1.0.
    """

    class_id: int
    name: str
    boxes: int
    trusted: bool
    share: float


@dataclass(frozen=True)
class Summary:
    """The dataset at a glance.

    Attributes:
        frames: Label files on disk, each one a frame a human has decided on.
        empty: Frames confirmed to hold nothing, which export as background images.
        boxes: Total boxes across every class.
        classes: One row per defined class, in id order.
    """

    frames: int
    empty: int
    boxes: int
    classes: list[ClassRow]


def summarise(
    label_dir: str = "data/labels",
    classes_file: str = CLASSES_FILE,
    trust_at: int = MIN_EXAMPLES_TO_TRUST,
) -> Summary:
    """Count what has been labelled so far.

    A class defined but never used is still reported, at zero. Its absence is
    the very thing worth seeing: a class nobody has drawn yet cannot be learned,
    and the row saying so is more useful than a missing line.

    Args:
        label_dir: Directory of YOLO label files.
        classes_file: Class definition file, or None for the default.
        trust_at: Boxes a class needs before the detector may auto-accept it.

    Returns:
        Summary: Frame and box counts, and one row per class.
    """
    names = load_classes(classes_file)
    counts = count_boxes_per_class(label_dir)

    files = sorted(Path(label_dir).glob("*.txt")) if Path(label_dir).is_dir() else []
    empty = 0
    for path in files:
        try:
            if not path.read_text(encoding="utf-8").strip():
                empty += 1
        except OSError:
            continue

    total = sum(counts.values())
    rows = [
        ClassRow(
            class_id=class_id,
            name=name,
            boxes=counts.get(class_id, 0),
            trusted=counts.get(class_id, 0) >= trust_at,
            share=(counts.get(class_id, 0) / total) if total else 0.0,
        )
        for class_id, name in enumerate(names)
    ]

    return Summary(frames=len(files), empty=empty, boxes=total, classes=rows)

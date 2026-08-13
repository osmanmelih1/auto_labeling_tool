"""Reorganise a project's class scheme without abandoning its labels.

A class list is not settled at the start of a project. It is settled after a few
hundred frames have been looked at, when it becomes clear that one class was
really three, or that two are never told apart, or that a catch-all defined as
"everything else" has no consistent appearance and no detector will ever learn
it.

Until now that discovery was expensive. A class's position in the list is its
YOLO class id, so removing one from the middle silently changes the meaning of
every label above it, and the class editor therefore refused to do it. The
refusal was right; the missing piece was a tool that renumbers the labels to
match.

Nothing is written without ``--apply``. Run it once to see the report, once more
to act on it.

Input:  ``data/classes.json``, ``data/labels/``
Output: the same two, rewritten together
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.core.class_config import CLASSES_FILE, load_class_records, save_class_records
from src.core.yolo_format import count_boxes_per_class, read_yolo_boxes, write_yolo_boxes

LABEL_DIR = "data/labels"


def report(records: list[dict], counts: Counter) -> None:
    """Print the current class scheme and how much of the dataset uses it.

    Args:
        records: Class records in id order.
        counts: Box count keyed by class id.
    """
    total = sum(counts.values()) or 1
    print(f"[*] {len(records)} class(es), {sum(counts.values())} box(es) on disk.\n")
    print(f"    {'id':>3}  {'name':<20} {'boxes':>7}  {'share':>6}")
    for class_id, record in enumerate(records):
        boxes = counts.get(class_id, 0)
        print(f"    {class_id:>3}  {record['name']:<20} {boxes:>7}  {boxes / total:>5.1%}")

    unknown = sorted(cid for cid in counts if cid >= len(records))
    if unknown:
        print(f"\n[!] Label files reference undefined class id(s): {unknown}")


def build_remap(records: list[dict], removed: int, reassign_to: int | None) -> dict[int, int | None]:
    """Work out the new id of every existing class id.

    Args:
        records: Class records before the change.
        removed: Class id being taken out of the list.
        reassign_to: Class id its boxes become, or None to delete them.

    Returns:
        dict: Old class id to new class id, or to None where the boxes go.
    """
    remap: dict[int, int | None] = {}
    for class_id in range(len(records)):
        if class_id == removed:
            remap[class_id] = None if reassign_to is None else reassign_to
        else:
            remap[class_id] = class_id

    # Everything above the removed class shifts down by one, including the
    # destination of a reassignment when that destination sat above it.
    for class_id, target in remap.items():
        if target is not None and target > removed:
            remap[class_id] = target - 1

    return remap


def apply_remap(remap: dict[int, int | None], label_dir: str = LABEL_DIR) -> tuple[int, int]:
    """Rewrite every label file under a new numbering.

    Args:
        remap: Old class id to new class id, or None to drop the box.
        label_dir: Directory of YOLO label files.

    Returns:
        tuple: How many files were rewritten and how many boxes were dropped.
    """
    rewritten = dropped = 0

    for path in sorted(Path(label_dir).glob("*.txt")):
        boxes = read_yolo_boxes(str(path))
        if not boxes:
            continue

        updated = []
        for class_id, *box in boxes:
            target = remap.get(class_id, class_id)
            if target is None:
                dropped += 1
                continue
            updated.append((target, *box))

        if updated != boxes:
            write_yolo_boxes(str(path), updated)
            rewritten += 1

    return rewritten, dropped


def resolve(records: list[dict], name: str) -> int:
    """Find a class by name.

    Args:
        records: Class records in id order.
        name: Class name to look for.

    Returns:
        int: The class id.

    Raises:
        SystemExit: If no class has that name.
    """
    for class_id, record in enumerate(records):
        if record["name"] == name:
            return class_id

    print(f"[!] No class named '{name}'. Known: {', '.join(r['name'] for r in records)}")
    raise SystemExit(1)


def main(argv: list[str] | None = None) -> int:
    """Report on, or reorganise, the project's class scheme.

    Args:
        argv: Command line arguments. ``sys.argv[1:]`` when omitted.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--delete", metavar="NAME", help="remove a class that no label still uses")
    parser.add_argument(
        "--merge",
        nargs=2,
        metavar=("FROM", "INTO"),
        help="give every box of FROM the class INTO, then remove FROM",
    )
    parser.add_argument("--labels", default=LABEL_DIR, help="directory of label files")
    parser.add_argument("--classes", default=CLASSES_FILE, help="class definition file")
    parser.add_argument("--apply", action="store_true", help="write the change; otherwise report only")
    args = parser.parse_args(argv)

    records = load_class_records(args.classes)
    counts = count_boxes_per_class(args.labels)

    if not args.delete and not args.merge:
        report(records, counts)
        print("\n[*] Nothing to change. Pass --delete NAME or --merge FROM INTO.")
        return 0

    if args.delete and args.merge:
        print("[!] Use one of --delete or --merge, not both.")
        return 1

    if args.merge:
        source, destination = args.merge
        removed = resolve(records, source)
        reassign_to = resolve(records, destination)
        if removed == reassign_to:
            print("[!] A class cannot be merged into itself.")
            return 1
        action = f"merge '{source}' ({counts.get(removed, 0)} box(es)) into '{destination}'"
    else:
        removed = resolve(records, args.delete)
        reassign_to = None
        held = counts.get(removed, 0)
        if held:
            print(
                f"[!] '{args.delete}' still holds {held} box(es). Deleting it would throw them away.\n"
                f"    Reassign them first with --merge {args.delete} <other class>, or relabel them"
                f" in the review editor."
            )
            return 1
        action = f"delete '{args.delete}' (no boxes use it)"

    remap = build_remap(records, removed, reassign_to)
    moved = {old: new for old, new in remap.items() if new != old}

    print(f"[*] Planned: {action}.\n")
    print(f"    {'was':>3}  {'name':<20} -> {'now':>3}  name")
    for old, new in sorted(moved.items()):
        target = "dropped" if new is None else f"{new:>3}  {records[old if new == old else removed]['name']}"
        name = records[old]["name"]
        print(f"    {old:>3}  {name:<20} -> {target}")

    if not args.apply:
        print("\n[*] Nothing was written. Re-run with --apply to make it so.")
        return 0

    rewritten, dropped = apply_remap(remap, args.labels)
    save_class_records([r for i, r in enumerate(records) if i != removed], args.classes)

    print(f"\n[+] {rewritten} label file(s) rewritten, {dropped} box(es) dropped.")
    print(f"[+] {args.classes} now defines {len(records) - 1} class(es).")
    print("[!] The exported dataset is now out of date. Run Step 6 (Export) before training again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

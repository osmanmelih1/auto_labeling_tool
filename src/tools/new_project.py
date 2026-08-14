"""Clear one project's data so the tool can be pointed at the next one.

Nothing in the source is tied to a particular dataset: class names live in
``data/classes.json`` and are edited from the GUI, colours are generated for any
number of classes, and every path resolves under ``data/``. What ties the tool
to a project is the contents of that directory, and until now there was no
documented way to empty it.

Left in place, the leftovers do not raise — they quietly corrupt the next
project. ``step1`` adds to ``data/deduplicated/`` rather than replacing it, so
the old frames stay and are exported alongside the new. ``step7`` picks the most
recent checkpoint under ``runs/``, so a detector trained on the last project
confidently pre-labels this one with the last project's classes. Neither failure
announces itself.

What is *not* cleared matters as much. ``data/models/`` holds DINOv3, SAM and
the YOLO starting weights — around 700 MB that has nothing to do with any
project and would only have to be downloaded again.

Nothing is written without ``--apply``. Run it once to see what would go, once
more to make it go.

Input:  ``data/``, ``datasets/``, ``runs/``
Output: the same, emptied of one project
"""

import argparse
import shutil
from pathlib import Path

# Everything a project writes. Directories are emptied, files are deleted; the
# directory itself stays so no step has to create it on the next run.
PROJECT_DIRECTORIES = [
    "data/raw",
    "data/deduplicated",
    "data/labels",
    "data/masks",
    "data/embeddings",
    "data/debug",
    "data/audit",
    "data/candidates",
    "data/previews",
    "data/spotcheck",
    "datasets",
    "runs",
]

PROJECT_FILES = [
    "data/review_queue.json",
    "data/review_sessions.jsonl",
    "data/temp_seed.json",
    "data/current_prompt.json",
]

# The class scheme is the one piece a second project on the same domain might
# want to keep, so it is cleared by default but exempted by a flag.
CLASSES_FILE = "data/classes.json"

# Model weights are downloaded, identical for every project, and large.
KEPT = ["data/models"]


def measure(path: Path) -> tuple[int, int]:
    """Count what a path holds.

    Args:
        path: File or directory to measure.

    Returns:
        tuple: Number of files and total bytes.
    """
    if path.is_file():
        return 1, path.stat().st_size
    if not path.is_dir():
        return 0, 0

    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def human(size: int) -> str:
    """Render a byte count at a readable scale.

    Args:
        size: Number of bytes.

    Returns:
        str: Something like ``1.2 GB``.
    """
    scaled = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if scaled < 1024 or unit == "GB":
            return f"{scaled:.0f} {unit}" if unit == "B" else f"{scaled:.1f} {unit}"
        scaled /= 1024
    return f"{scaled:.1f} GB"


def survey(targets: list[str]) -> list[tuple[Path, int, int]]:
    """Measure every target that exists.

    Args:
        targets: Paths to look at.

    Returns:
        list: ``(path, file count, bytes)`` for those that are present.
    """
    found = []
    for name in targets:
        path = Path(name)
        files, size = measure(path)
        if files:
            found.append((path, files, size))
    return found


def clear(path: Path) -> None:
    """Empty a directory or delete a file, leaving the directory itself.

    A step that finds its output directory missing has to create it, and not
    every step does. Emptying rather than removing keeps that from mattering.

    Args:
        path: File or directory to clear.
    """
    if path.is_file():
        path.unlink()
        return

    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main(argv: list[str] | None = None) -> int:
    """Report on, or carry out, clearing the current project.

    Args:
        argv: Command line arguments. ``sys.argv[1:]`` when omitted.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--keep-classes",
        action="store_true",
        help="leave data/classes.json alone, for a second project on the same domain",
    )
    parser.add_argument("--apply", action="store_true", help="delete; otherwise report only")
    args = parser.parse_args(argv)

    targets = PROJECT_DIRECTORIES + PROJECT_FILES
    if not args.keep_classes:
        targets = targets + [CLASSES_FILE]

    going = survey(targets)
    staying = survey(KEPT)

    if not going:
        print("[+] Nothing to clear. This is already a fresh project.")
        return 0

    print(f"    {'would clear':<26} {'files':>7}  {'size':>9}")
    for path, files, size in going:
        print(f"    {str(path):<26} {files:>7}  {human(size):>9}")
    print(f"    {'':<26} {sum(f for _, f, _ in going):>7}  {human(sum(s for _, _, s in going)):>9}\n")

    if staying:
        print(f"    {'kept':<26} {'files':>7}  {'size':>9}")
        for path, files, size in staying:
            print(f"    {str(path):<26} {files:>7}  {human(size):>9}")
        print("    Model weights are the same for every project and slow to fetch again.\n")

    if args.keep_classes:
        print(f"[*] {CLASSES_FILE} is being kept. Clear it too if the next project has other classes.")
    else:
        print(f"[*] {CLASSES_FILE} will go. Pass --keep-classes to reuse this class scheme.")

    trained = sorted(Path("runs").glob("*/weights/best.pt")) if Path("runs").exists() else []
    if trained:
        print(f"\n[!] {len(trained)} trained detector(s) under runs/ will be deleted, including")
        print(f"    {trained[-1]}.")
        print("    Copy out anything worth keeping before applying: this project's model is")
        print("    the whole point of the labelling, and it is not in Git.")

    if not args.apply:
        print("\n[*] Nothing was deleted. Re-run with --apply to make it so.")
        return 0

    for path, _, _ in going:
        clear(path)

    print(f"\n[+] Cleared {len(going)} location(s). Put the new images in data/raw and start at Step 1.")
    if not args.keep_classes:
        print("[+] Define the new classes from 'Manage Classes' in the GUI before seeding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

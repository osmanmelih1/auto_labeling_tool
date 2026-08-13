"""Draw the labels as they are now, so a dataset can be checked by eye.

The debug images the pipeline writes are snapshots of what a machine proposed at
the moment it proposed it. They are frozen there: a box moved, deleted or
reclassified afterwards still appears in them exactly as the machine first drew
it. Reading them back as if they showed the dataset is how a corrected frame gets
"corrected" a second time.

This renders the label files instead. Whatever is on disk right now is what
appears, which makes it the only honest way to look at a dataset that people have
been editing.

Writes nothing but images, and never to ``data/labels/``.

Input:  ``data/deduplicated/``, ``data/labels/``, ``data/classes.json``
Output: ``data/previews/``
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from src.core.class_config import class_color, class_name, load_classes
from src.core.yolo_format import read_yolo_boxes, yolo_box_to_pixels

IMAGE_DIR = "data/deduplicated"
LABEL_DIR = "data/labels"
OUTPUT_DIR = "data/previews"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def resolve_image(image_dir: Path, key: str) -> Path | None:
    """Find the image file for an extensionless key.

    Args:
        image_dir: Directory to search.
        key: Filename without extension.

    Returns:
        Path | None: The image, or None when no file matches.
    """
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{key}{extension}"
        if candidate.exists():
            return candidate
    return None


def summarise(names: list[str], boxes: list) -> str:
    """Describe a frame's contents compactly enough for a filename.

    Args:
        names: Class names in id order.
        boxes: Boxes read from the label file.

    Returns:
        str: Something like ``2-palet_3lu_1-koli``, or ``empty``.
    """
    if not boxes:
        return "empty"

    counts: dict[int, int] = {}
    for class_id, *_ in boxes:
        counts[class_id] = counts.get(class_id, 0) + 1

    return "_".join(f"{count}-{class_name(names, class_id)}" for class_id, count in sorted(counts.items()))


def render(image_path: Path, boxes: list, names: list[str], destination: Path) -> None:
    """Draw a frame's current labels onto a copy of it.

    Args:
        image_path: Source image.
        boxes: Boxes read from its label file.
        names: Class names in id order.
        destination: Where to write the annotated copy.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for class_id, *box in boxes:
        x0, y0, x1, y1 = yolo_box_to_pixels(tuple(box), image.width, image.height)
        colour = class_color(class_id)
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=4)
        draw.text((x0 + 6, max(0, y0 - 16)), class_name(names, class_id), fill=colour)

    image.save(destination, quality=80)


def main(argv: list[str] | None = None) -> int:
    """Render the current labels of some or all frames.

    Args:
        argv: Command line arguments. ``sys.argv[1:]`` when omitted.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--keys", help="comma-separated image keys to render")
    parser.add_argument("--keys-file", help="file with one image key per line")
    parser.add_argument("--class", dest="target", help="only frames containing this class")
    parser.add_argument("--empty", action="store_true", help="only frames whose label file is empty")
    parser.add_argument("--images", default=IMAGE_DIR, help="directory of source images")
    parser.add_argument("--labels", default=LABEL_DIR, help="directory of label files")
    parser.add_argument("--out", default=OUTPUT_DIR, help="directory to write previews into")
    args = parser.parse_args(argv)

    names = load_classes()
    class_id_wanted = None
    if args.target:
        if args.target not in names:
            print(f"[!] No class named '{args.target}'. Known: {', '.join(names)}")
            return 1
        class_id_wanted = names.index(args.target)

    wanted: set[str] | None = None
    if args.keys:
        wanted = {key.strip() for key in args.keys.split(",") if key.strip()}
    if args.keys_file:
        text = Path(args.keys_file).read_text(encoding="utf-8")
        wanted = (wanted or set()) | {line.strip() for line in text.splitlines() if line.strip()}

    image_dir = Path(args.images)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.jpg"):
        stale.unlink()

    written = missing = 0

    for label_path in sorted(Path(args.labels).glob("*.txt")):
        key = label_path.stem
        if wanted is not None and key not in wanted:
            continue

        boxes = read_yolo_boxes(str(label_path))
        if args.empty and boxes:
            continue
        if class_id_wanted is not None and not any(cid == class_id_wanted for cid, *_ in boxes):
            continue

        image_path = resolve_image(image_dir, key)
        if image_path is None:
            print(f"  [-] {key}: no image found in {image_dir}")
            missing += 1
            continue

        try:
            render(image_path, boxes, names, output_dir / f"{summarise(names, boxes)}_{key}.jpg")
            written += 1
        except Exception as e:
            print(f"  [-] {key}: could not render ({e})")
            missing += 1

    print(f"[+] {written} preview(s) written to {output_dir}")
    if missing:
        print(f"[!] {missing} frame(s) could not be rendered.")
    if wanted:
        absent = wanted - {p.stem for p in Path(args.labels).glob("*.txt")}
        if absent:
            print(f"[!] {len(absent)} requested key(s) have no label file: {', '.join(sorted(absent))}")
    print("[*] These are the labels as they are on disk now, corrections included.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

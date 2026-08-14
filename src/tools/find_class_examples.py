"""Find frames that probably contain a class there are too few examples of.

Every dataset has a rare class. Here it is a single-row pallet: three boxes in
three hundred and thirty-nine. A detector cannot learn from three, and labelling
more random frames will not help, because a class that appears in one frame in a
hundred still appears in one frame in a hundred however many are labelled.

They have to be sought out, and opening frames at random to look for them is
hours of work. The detector already has an opinion — it is simply below the
threshold anything acts on. Asking it at a confidence nobody would trust turns
a search into a shortlist.

Nothing is written to ``data/labels/``. This tool proposes reading, not
labelling: it ranks the frames where the class might be and copies annotated
previews somewhere they can be flipped through. What to do with them is a
judgement, and it is made in the seeding canvas or the label editor.

Input:  ``runs/*/weights/best.pt``, ``data/deduplicated/``, ``data/labels/``
Output: ``data/candidates/<class>/`` and a ranked list on the console
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from src.core.class_config import load_classes
from src.core.step7_predict import IMAGE_EXTENSIONS, find_latest_weights
from src.core.yolo_format import read_yolo_boxes, yolo_box_to_pixels

IMAGE_DIR = "data/deduplicated"
LABEL_DIR = "data/labels"
OUTPUT_DIR = "data/candidates"

# Far below anything the pipeline would act on. At this level the detector is not
# claiming to have found the object; it is reporting where it would look.
SEARCH_CONFIDENCE = 0.05

BATCH_SIZE = 16
DEFAULT_LIMIT = 40


class ClassExampleFinder:
    """Ranks images by how likely a detector thinks one class is present."""

    def __init__(
        self,
        target: str,
        weights: str | None = None,
        image_dir: str = IMAGE_DIR,
        label_dir: str = LABEL_DIR,
        output_dir: str = OUTPUT_DIR,
        confidence: float = SEARCH_CONFIDENCE,
    ) -> None:
        """Load the detector and resolve the class being searched for.

        Args:
            target: Name of the class to look for.
            weights: Checkpoint to search with. The latest run is used when omitted.
            image_dir: Directory of source images.
            label_dir: Directory of existing labels, read only to report coverage.
            output_dir: Directory receiving the previews.
            confidence: Floor to ask the detector for.

        Raises:
            FileNotFoundError: If no trained checkpoint can be found.
            ValueError: If no class has that name.
        """
        names = load_classes()
        if target not in names:
            raise ValueError(f"[!] No class named '{target}'. Known: {', '.join(names)}")

        self.target = target
        self.class_id = names.index(target)
        self.names = names

        self.weights = weights or find_latest_weights()
        if not self.weights or not Path(self.weights).exists():
            raise FileNotFoundError("[!] No trained detector found. Train one before searching.")

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir) / target
        self.confidence = confidence

        from ultralytics import YOLO

        print(f"[*] Loading detector from {self.weights}...")
        self.model = YOLO(self.weights)
        print("[+] Detector ready.")

    def images(self) -> list[Path]:
        """List every image to search.

        Labelled frames are included on purpose. A frame labelled as one class may
        hold an unlabelled instance of another, and a missed rare object is
        exactly what is being hunted.

        Returns:
            list: Paths to every image in the source directory.
        """
        return [p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]

    def already_labelled(self, stem: str) -> bool:
        """Report whether a frame already carries the class being searched for.

        Args:
            stem: Image key.

        Returns:
            bool: True when its label file already holds a box of this class.
        """
        path = self.label_dir / f"{stem}.txt"
        return any(class_id == self.class_id for class_id, *_ in read_yolo_boxes(str(path)))

    def search(self, paths: list[Path]) -> list[tuple[Path, float, list]]:
        """Score every image by the detector's best guess at the target class.

        Args:
            paths: Images to search.

        Returns:
            list: ``(path, best_confidence, boxes)`` for images where the class
            was suspected at all, ranked most confident first.
        """
        found: list[tuple[Path, float, list]] = []

        for start in range(0, len(paths), BATCH_SIZE):
            batch = paths[start : start + BATCH_SIZE]
            results = self.model.predict(
                [str(p) for p in batch],
                conf=self.confidence,
                verbose=False,
            )

            for path, result in zip(batch, results, strict=True):
                boxes = [
                    (float(box.conf.item()), tuple(box.xywhn[0].tolist()))
                    for box in result.boxes
                    if int(box.cls.item()) == self.class_id
                ]
                if boxes:
                    boxes.sort(reverse=True)
                    found.append((path, boxes[0][0], boxes))

        found.sort(key=lambda item: item[1], reverse=True)
        return found

    def save_preview(self, path: Path, boxes: list, rank: int) -> None:
        """Write a copy of the image with the suspected boxes drawn on it.

        Args:
            path: Source image.
            boxes: ``(confidence, box)`` pairs for the target class.
            rank: Position in the ranked list, used to order the filenames.
        """
        try:
            image = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(image)

            for confidence, box in boxes:
                x0, y0, x1, y1 = yolo_box_to_pixels(box, image.width, image.height)
                draw.rectangle([x0, y0, x1, y1], outline=(255, 176, 0), width=4)
                draw.text((x0 + 6, max(0, y0 - 16)), f"{self.target} {confidence:.2f}", fill=(255, 176, 0))

            image.save(self.output_dir / f"{rank:03d}_{boxes[0][0]:.2f}_{path.stem}.jpg", quality=80)
        except Exception as e:
            print(f"  [-] {path.stem}: could not write preview ({e}).")

    def run(self, limit: int = DEFAULT_LIMIT) -> None:
        """Search every image and write the top candidates as previews.

        Args:
            limit: How many candidates to keep.
        """
        paths = self.images()
        if not paths:
            print(f"[-] No images in {self.image_dir}.")
            return

        print(f"[*] Searching {len(paths)} image(s) for '{self.target}' at confidence >= {self.confidence}.")
        found = self.search(paths)

        if not found:
            print(f"[-] The detector never suspects '{self.target}' anywhere, even at {self.confidence}.")
            print("[*] It has too few examples to have an opinion. Seed some by hand from frames you")
            print("    know contain it, then train again and search once more.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for existing in self.output_dir.glob("*.jpg"):
            existing.unlink()

        fresh = kept = 0
        print(f"\n[+] {len(found)} image(s) where '{self.target}' is suspected. Top {limit}:\n")

        for path, confidence, boxes in found[:limit]:
            labelled = self.already_labelled(path.stem)
            self.save_preview(path, boxes, kept)
            kept += 1
            if not labelled:
                fresh += 1
            marker = "already labelled" if labelled else "NEW"
            print(f"  {confidence:.3f}  {path.stem}  [{marker}]")

        print(f"\n[+] {kept} preview(s) written to {self.output_dir}")
        print(f"[*] {fresh} of them do not yet carry a '{self.target}' box.")
        print("[*] Nothing was labelled. Open the ones that look right in the seeding canvas")
        print("    (Step 3b) or in Edit Labels, and add the box yourself.")


def main(argv: list[str] | None = None) -> int:
    """Search the dataset for a class that needs more examples.

    Args:
        argv: Command line arguments. ``sys.argv[1:]`` when omitted.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("target", help="name of the class to look for")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="how many candidates to keep")
    parser.add_argument(
        "--confidence",
        type=float,
        default=SEARCH_CONFIDENCE,
        help="how faint a suspicion to accept",
    )
    parser.add_argument("--weights", help="checkpoint to search with")
    parser.add_argument(
        "--image-dir",
        default=IMAGE_DIR,
        help=(
            "where to search. Deduplication can discard the very frame a rare class "
            "appears in, so point this at data/raw when the deduplicated pool comes up empty"
        ),
    )
    args = parser.parse_args(argv)

    try:
        finder = ClassExampleFinder(
            args.target,
            weights=args.weights,
            image_dir=args.image_dir,
            confidence=args.confidence,
        )
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return 1

    finder.run(limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())

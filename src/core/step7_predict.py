"""Step 7 - Pre-labelling with the trained detector.

This is the step that closes the active learning loop. Until now the loop ran
one way: seeds produced labels, labels produced a dataset, the dataset produced
a model, and the model was never heard from again. Every new image cost the same
human attention as the first one.

The detector is a better proposer than similarity matching for anything it has
actually seen. It reports a calibrated confidence rather than a cosine distance,
it separates classes that differ by extent rather than by texture — which patch
prototypes provably cannot — and it runs in milliseconds per image instead of
seconds. What it cannot do is recognise a class it was barely trained on, which
is why Step 4 stays: propagation is the cold start, this is what replaces it once
there is a model.

The output is the same review queue Step 4 writes, so the review screen neither
knows nor cares which of the two proposed a box.

**Read the queue least-confident first.** A frame the detector scores at 0.95
teaches nobody anything; the frames worth a human's time are the ones it found
difficult, and correcting those is what makes the next model better. The review
screen's sort control has a "Score: Low to High" option for exactly this.

Input:  ``runs/*/weights/best.pt``, ``data/deduplicated/``, ``data/labels/``
Output: ``data/labels/``, ``data/review_queue.json``, ``data/debug/``
"""

import os
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

from src.core.class_config import class_name, load_classes
from src.core.review_queue import REVIEW_QUEUE_PATH, add_pending, is_suppressed, load_queue, save_queue
from src.core.tiers import DETECTOR_AUTO_ACCEPT, DETECTOR_REVIEW
from src.core.yolo_format import write_yolo_boxes, yolo_box_to_pixels

IMAGE_DIR = "data/deduplicated"
LABEL_DIR = "data/labels"
DEBUG_DIR = "data/debug"
RUNS_DIR = "runs"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Ultralytics batches internally; this only bounds how many images are held in
# memory at once on a 6 GB card.
BATCH_SIZE = 16


def find_latest_weights(runs_dir: str = RUNS_DIR) -> str | None:
    """Locate the most recently trained detector.

    Args:
        runs_dir: Directory Ultralytics writes runs into.

    Returns:
        str | None: Path to ``best.pt``, or None when nothing has been trained.
    """
    candidates = sorted(
        Path(runs_dir).glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


class DetectorPreLabeller:
    """Runs a trained detector over unlabelled images and tiers its output."""

    def __init__(
        self,
        weights: str | None = None,
        image_dir: str = IMAGE_DIR,
        label_dir: str = LABEL_DIR,
        debug_dir: str = DEBUG_DIR,
        review_queue_path: str = REVIEW_QUEUE_PATH,
        auto_threshold: float = DETECTOR_AUTO_ACCEPT,
        review_threshold: float = DETECTOR_REVIEW,
    ) -> None:
        """Load the detector and prepare the output directories.

        Args:
            weights: Checkpoint to predict with. The latest run is used when omitted.
            image_dir: Directory of source images.
            label_dir: Directory holding existing labels and receiving new ones.
            debug_dir: Directory receiving annotated previews.
            review_queue_path: Queue file shared with the GUI.
            auto_threshold: Confidence at or above which a box needs no human.
            review_threshold: Confidence below which a box is not proposed at all.

        Raises:
            FileNotFoundError: If no trained checkpoint can be found.
        """
        self.weights = weights or find_latest_weights()
        if not self.weights or not os.path.exists(self.weights):
            raise FileNotFoundError(
                "[!] No trained detector found. Run Step 7 (Train YOLO) before this step."
            )

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.debug_dir = Path(debug_dir)
        self.review_queue_path = review_queue_path
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold

        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Loading detector from {self.weights}...")
        self.model = YOLO(self.weights)
        print("[+] Detector ready.")

    def unlabelled_images(self) -> list[Path]:
        """List images that no human has decided on yet.

        An image with a label file has either been reviewed or been auto-accepted
        on the operator's behalf. Re-proposing it would overwrite work.

        Returns:
            list: Paths to images without a label file.
        """
        images = [p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
        return [p for p in images if not (self.label_dir / f"{p.stem}.txt").exists()]

    def predict(self, paths: list[Path]) -> list[tuple[Path, list[tuple[int, float, tuple]]]]:
        """Run the detector over a list of images.

        Args:
            paths: Images to predict on.

        Returns:
            list: One ``(path, detections)`` pair per image, where a detection is
            ``(class_id, confidence, (x_center, y_center, width, height))``
            normalised to [0, 1].
        """
        output: list[tuple[Path, list[tuple[int, float, tuple]]]] = []

        for start in range(0, len(paths), BATCH_SIZE):
            batch = paths[start : start + BATCH_SIZE]
            results = self.model.predict(
                [str(p) for p in batch],
                conf=self.review_threshold,
                verbose=False,
            )

            for path, result in zip(batch, results, strict=True):
                detections = []
                for box in result.boxes:
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    detections.append(
                        (int(box.cls.item()), float(box.conf.item()), (x_center, y_center, width, height))
                    )
                detections.sort(key=lambda d: d[1], reverse=True)
                output.append((path, detections))

        return output

    def save_debug_preview(self, path: Path, detections: list, decision: str, score: float) -> None:
        """Draw the proposed boxes on a copy of the image.

        Args:
            path: Source image.
            detections: Detections for this image.
            decision: Tier name, used in the filename.
            score: Weakest confidence in the frame, used in the filename.
        """
        try:
            image = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(image)
            names = load_classes()

            for class_id, confidence, box in detections:
                x0, y0, x1, y1 = yolo_box_to_pixels(box, image.width, image.height)
                colour = (0, 255, 0) if confidence >= self.auto_threshold else (255, 176, 0)
                draw.rectangle([x0, y0, x1, y1], outline=colour, width=4)
                draw.text(
                    (x0 + 6, max(0, y0 - 16)),
                    f"{class_name(names, class_id)} {confidence:.2f}",
                    fill=colour,
                )

            name = f"det_{decision.lower()}_{score:.3f}_n{len(detections)}_{path.stem}.jpg"
            image.save(self.debug_dir / name, quality=80)
        except Exception as e:
            print(f"  [-] {path.stem}: could not write preview ({e}).")

    def run(self) -> None:
        """Pre-label every unlabelled image and tier the result."""
        targets = self.unlabelled_images()
        if not targets:
            print("[-] Every image already has a label. Nothing to pre-label.")
            return

        names = load_classes()
        print(f"[*] Pre-labelling {len(targets)} unlabelled image(s).")
        print(f"[*] Tiers -> AUTO >= {self.auto_threshold} | REVIEW >= {self.review_threshold}\n")

        queue = load_queue(self.review_queue_path)
        auto = review = empty = suppressed = objects = 0

        for path, detections in self.predict(targets):
            key = path.stem

            if not detections:
                print(f"  [-]      {key} | nothing found above {self.review_threshold}")
                empty += 1
                continue

            best = detections[0][1]
            weakest = min(confidence for _, confidence, _ in detections)

            skip, previous = is_suppressed(queue, key, best)
            if skip:
                print(f"  [SKIP  ] {key} | {best:.3f} | already rejected at {previous:.3f}")
                suppressed += 1
                continue

            # The same rule Step 4 applies: a frame is only accepted outright
            # when every box in it would be. One uncertain instance is reason
            # enough for a human to see the frame, because a missed object
            # teaches the next model that the object is background.
            decision = "AUTO" if weakest >= self.auto_threshold else "REVIEW"

            label_path = self.label_dir / f"{key}.txt"
            write_yolo_boxes(str(label_path), [(class_id, *box) for class_id, _, box in detections])
            self.save_debug_preview(path, detections, decision, weakest)

            if decision == "AUTO":
                queue.setdefault("pending", {}).pop(key, None)
                auto += 1
            else:
                add_pending(
                    queue,
                    key,
                    {
                        "score": round(best, 4),
                        "weakest_score": round(weakest, 4),
                        "object_count": len(detections),
                        "box_scores": [round(confidence, 4) for _, confidence, _ in detections],
                        "seed_source": os.path.basename(os.path.dirname(os.path.dirname(self.weights))),
                        "class_id": detections[0][0],
                        "label_path": os.path.abspath(label_path),
                        "mask_path": None,
                        "image_path": os.path.abspath(path),
                        "image_key": key,
                        "flagged_at": datetime.now(UTC).isoformat(),
                        "status": "pending_review",
                        "method": "detector_prediction",
                    },
                )
                review += 1

            objects += len(detections)
            summary = ", ".join(
                f"{class_name(names, class_id)} {confidence:.3f}" for class_id, confidence, _ in detections
            )
            print(f"  [{decision:6}] {key} | weakest {weakest:.3f} | {len(detections)} object(s): {summary}")

        save_queue(queue, self.review_queue_path)

        labelled = auto + review
        print("\n[+] Pre-labelling complete:")
        print(f"    - Detector      : {self.weights}")
        print(f"    - Auto-accepted : {auto}")
        print(f"    - Review queue  : {review} (see {self.review_queue_path})")
        print(f"    - Nothing found : {empty}")
        if suppressed:
            print(f"    - Skipped       : {suppressed} previously rejected by a human")
        if labelled:
            print(f"    - Objects boxed : {objects} across {labelled} image(s)")
            print(f"                      ({objects / labelled:.2f} per labelled image)")
        print(f"[*] Annotated previews written to {self.debug_dir} for visual inspection.")

        if review:
            print(f"[!] {review} image(s) awaiting approval in the GUI 'Review Queue'.")
            print("[*] Sort it by 'Score: Low -> High'. The frames the detector found hardest")
            print("    are the ones a correction teaches the most.")


if __name__ == "__main__":
    try:
        DetectorPreLabeller().run()
    except Exception as e:
        print(f"[!] An error occurred: {e}")

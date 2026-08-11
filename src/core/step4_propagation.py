"""Step 4 - Patch-Level Propagation with Localisation.

Spreads a handful of hand-made seed labels across the unlabelled dataset. For
every target image the object is *located*, not assumed: the seed's coordinates
are never copied.

The pipeline per target image is the one agreed on the whiteboard,
``compare DINOv3 embeddings -> check similarity -> SAM -> localize``:

1. Every seed box is reduced to a prototype vector by mean-pooling the DINOv3
   patch tokens that fall inside it. One prototype per seed box, grouped by class.
2. The target's patch grid is compared against every prototype, producing a
   cosine similarity heatmap per prototype. The best score over all prototypes is
   the detection confidence, and the winning seed is recorded as provenance.
3. The peak of that heatmap and its surrounding high-similarity region give a
   coarse box in the target's own pixel space.
4. SAM turns that coarse prompt into a precise mask, and the mask's bounding box
   becomes the YOLO label.

Why this replaces the previous design: the old step compared whole-image CLS
vectors and, on a match, copied the seed's label file verbatim with shutil.copy2.
Two images of the same scene therefore received identical coordinates regardless
of where the object actually was, and the global vector responded more to
lighting and viewpoint than to the object itself.

Confidence tiers gate the result:
  - AUTO-ACCEPT  (score >= AUTO_ACCEPT_THRESHOLD)
  - REVIEW QUEUE (REVIEW_THRESHOLD <= score < AUTO_ACCEPT_THRESHOLD)
  - REJECTED     (score < REVIEW_THRESHOLD)

Known limitation: one object per image. The peak region yields a single box, so
frames containing several instances of a class receive only the strongest one.

Input:  ``data/embeddings/patches/``, ``data/labels/`` (seeds), ``data/deduplicated/``
Output: ``data/labels/``, ``data/masks/``, ``data/review_queue.json``, ``data/debug/``
"""

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.core.review_queue import REVIEW_QUEUE_PATH, add_pending, is_suppressed, load_queue, save_queue
from src.core.sam_engine import SamEngine, box_area, mask_to_yolo_box, yolo_box_to_pixels

# --- Confidence Tier Configuration ---
# Calibrated against the first validation run, where images that genuinely
# contained the seeded object scored 0.82-0.91 and images that did not scored
# 0.69-0.76. The thresholds sit deliberately on the cautious side of that gap:
# with only a few seeds most true matches land in the review queue rather than
# being accepted outright. Tighten them once the score distribution over the full
# dataset is known, not before.
AUTO_ACCEPT_THRESHOLD = 0.86
REVIEW_THRESHOLD = 0.78

# The object region is every heatmap cell within this fraction of the peak's
# height above the heatmap floor. Lower values grow the coarse box.
REGION_RELATIVE_LEVEL = 0.75

# SAM occasionally latches onto the background instead of the prompted object.
# If its mask is this many times larger than the coarse box, the coarse box is
# kept instead and the case is flagged in the log.
MAX_SAM_AREA_GROWTH = 4.0

PATCH_DIR = "data/embeddings/patches"
IMAGE_DIR = "data/deduplicated"
LABEL_DIR = "data/labels"
MASK_DIR = "data/masks"
DEBUG_DIR = "data/debug"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def resolve_image_path(image_dir: str, image_key: str) -> str | None:
    """Find the image file matching an extensionless key.

    Args:
        image_dir: Directory to search.
        image_key: Filename without extension.

    Returns:
        str | None: Absolute path to the image, or None when no file matches.
    """
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(image_dir, image_key + ext)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def read_yolo_boxes(label_path: str) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into class ids and normalised boxes.

    Malformed lines are skipped rather than aborting the run: one bad label
    should not stop propagation across thousands of images.

    Args:
        label_path: Path to a YOLO .txt label file.

    Returns:
        list: One ``(class_id, x_center, y_center, width, height)`` per valid line.
    """
    boxes: list[tuple[int, float, float, float, float]] = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                xc, yc, w, h = (float(p) for p in parts[1:])
            except ValueError:
                continue
            boxes.append((class_id, xc, yc, w, h))
    return boxes


def unit(vectors: np.ndarray) -> np.ndarray:
    """Normalise vectors to unit length along their last axis.

    Args:
        vectors: Array whose last axis is the feature dimension.

    Returns:
        np.ndarray: The same shape, each feature vector scaled to length one.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / (norms + 1e-8)


@dataclass
class SeedPrototype:
    """A single annotated object reduced to one feature vector.

    Attributes:
        image_key: Source image the box was drawn on.
        class_id: YOLO class the box belongs to.
        vector: Unit-length mean of the patch tokens inside the box.
        cell_count: How many patch cells contributed, for diagnostics.
    """

    image_key: str
    class_id: int
    vector: np.ndarray
    cell_count: int


class PatchPropagator:
    """Locates seeded objects in unlabelled images and writes YOLO labels."""

    def __init__(
        self,
        patch_dir: str = PATCH_DIR,
        image_dir: str = IMAGE_DIR,
        label_dir: str = LABEL_DIR,
        mask_dir: str = MASK_DIR,
        review_queue_path: str = REVIEW_QUEUE_PATH,
        debug_dir: str = DEBUG_DIR,
        sam_engine: SamEngine | None = None,
    ) -> None:
        """Prepare directories and load SAM.

        Args:
            patch_dir: Directory of cached patch grids produced by Step 2.
            image_dir: Directory of source images.
            label_dir: Directory holding seed labels and receiving new ones.
            mask_dir: Directory receiving the generated masks.
            review_queue_path: JSON file backing the GUI review queue.
            debug_dir: Directory receiving heatmap overlays.
            sam_engine: Preloaded engine. Constructed on demand when omitted.

        Raises:
            FileNotFoundError: If Step 2 has not produced any patch grids.
        """
        self.patch_dir = Path(patch_dir)
        self.image_dir = image_dir
        self.label_dir = Path(label_dir)
        self.mask_dir = Path(mask_dir)
        self.review_queue_path = Path(review_queue_path)
        self.debug_dir = Path(debug_dir)

        if not self.patch_dir.exists():
            raise FileNotFoundError(
                f"[!] Patch grids not found at {self.patch_dir}. Run Step 2 first; "
                "propagation needs patch tokens, not just the global vector database."
            )

        self.image_keys = sorted(p.stem for p in self.patch_dir.glob("*.npy"))
        if not self.image_keys:
            raise FileNotFoundError(f"[!] No patch grids inside {self.patch_dir}.")

        print(f"[+] Found cached patch grids for {len(self.image_keys)} images.")

        for directory in (self.label_dir, self.mask_dir, self.debug_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.sam = sam_engine or SamEngine()

    def load_patch_grid(self, image_key: str) -> np.ndarray:
        """Load one cached patch grid as float32.

        Args:
            image_key: Filename without extension.

        Returns:
            np.ndarray: Grid of shape ``(rows, cols, dim)``.
        """
        return np.load(self.patch_dir / f"{image_key}.npy").astype(np.float32)

    def build_prototypes(self) -> list[SeedPrototype]:
        """Turn every existing label box into a prototype feature vector.

        Returns:
            list: One prototype per valid seed box across all label files.
        """
        prototypes: list[SeedPrototype] = []

        for image_key in self.image_keys:
            label_path = self.label_dir / f"{image_key}.txt"
            if not label_path.exists():
                continue

            grid = self.load_patch_grid(image_key)
            rows, cols, dim = grid.shape

            for class_id, xc, yc, w, h in read_yolo_boxes(str(label_path)):
                r0 = max(0, int(np.floor((yc - h / 2) * rows)))
                r1 = min(rows, int(np.ceil((yc + h / 2) * rows)))
                c0 = max(0, int(np.floor((xc - w / 2) * cols)))
                c1 = min(cols, int(np.ceil((xc + w / 2) * cols)))

                if r1 <= r0 or c1 <= c0:
                    print(f"  [-] {image_key}: box covers no patch cell, skipped as seed.")
                    continue

                region = grid[r0:r1, c0:c1].reshape(-1, dim)
                prototypes.append(
                    SeedPrototype(
                        image_key=image_key,
                        class_id=class_id,
                        vector=unit(region.mean(axis=0)),
                        cell_count=region.shape[0],
                    )
                )

        return prototypes

    def match(
        self, grid: np.ndarray, prototypes: list[SeedPrototype]
    ) -> tuple[float, SeedPrototype, np.ndarray]:
        """Score a target grid against every prototype and keep the best.

        Taking the maximum over prototypes rather than averaging them keeps the
        method robust to appearance variation: one seed shot in different light
        can still carry the match on its own.

        Args:
            grid: Target patch grid of shape ``(rows, cols, dim)``.
            prototypes: Candidate prototypes to compare against.

        Returns:
            tuple: Best score, the prototype that produced it, and its heatmap.
        """
        normalised = unit(grid)
        best_score = -1.0
        best_proto = prototypes[0]
        best_heatmap = np.zeros(grid.shape[:2], dtype=np.float32)

        for proto in prototypes:
            heatmap = normalised @ proto.vector
            score = float(heatmap.max())
            if score > best_score:
                best_score, best_proto, best_heatmap = score, proto, heatmap

        return best_score, best_proto, best_heatmap

    def peak_region_box(
        self, heatmap: np.ndarray
    ) -> tuple[tuple[float, float, float, float], tuple[int, int]]:
        """Derive a coarse normalised box from the heatmap's peak region.

        Only the connected region containing the peak is kept, so a second,
        unrelated hot spot elsewhere in the frame cannot stretch the box across
        the whole image.

        Args:
            heatmap: Cosine similarity per patch cell, shape ``(rows, cols)``.

        Returns:
            tuple: The normalised ``(x_center, y_center, width, height)`` box and
            the ``(row, col)`` index of the peak cell.
        """
        rows, cols = heatmap.shape
        peak_r, peak_c = np.unravel_index(int(heatmap.argmax()), heatmap.shape)

        floor, ceiling = float(heatmap.min()), float(heatmap.max())
        level = floor + (ceiling - floor) * REGION_RELATIVE_LEVEL
        binary = (heatmap >= level).astype(np.uint8)

        count, labelled = cv2.connectedComponents(binary, connectivity=8)
        if count > 1:
            binary = (labelled == labelled[peak_r, peak_c]).astype(np.uint8)

        occupied_rows = np.where(binary.any(axis=1))[0]
        occupied_cols = np.where(binary.any(axis=0))[0]
        r0, r1 = int(occupied_rows[0]), int(occupied_rows[-1]) + 1
        c0, c1 = int(occupied_cols[0]), int(occupied_cols[-1]) + 1

        box = (
            (c0 + c1) / 2.0 / cols,
            (r0 + r1) / 2.0 / rows,
            (c1 - c0) / cols,
            (r1 - r0) / rows,
        )
        return box, (int(peak_r), int(peak_c))

    def localize(
        self, image_key: str, heatmap: np.ndarray
    ) -> tuple[tuple[float, float, float, float], np.ndarray | None, str] | None:
        """Convert a heatmap into a precise YOLO box using SAM.

        Args:
            image_key: Target image key.
            heatmap: Similarity heatmap for the winning prototype.

        Returns:
            tuple | None: The final normalised box, the mask when SAM produced a
            usable one, and a note describing which result was kept. None when
            the source image cannot be read.
        """
        image_path = resolve_image_path(self.image_dir, image_key)
        if image_path is None:
            print(f"  [-] {image_key}: source image not found, cannot localise.")
            return None

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"  [-] {image_key}: image could not be decoded.")
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_rgb.shape[:2]

        coarse_box, (peak_r, peak_c) = self.peak_region_box(heatmap)
        rows, cols = heatmap.shape
        peak_xy = ((peak_c + 0.5) / cols * img_w, (peak_r + 0.5) / rows * img_h)

        self.sam.set_image(image_rgb)
        mask, sam_score = self.sam.mask_from_prompt(
            box_xyxy=yolo_box_to_pixels(coarse_box, img_w, img_h),
            point_xy=peak_xy,
        )

        refined = mask_to_yolo_box(mask)
        if refined is None:
            return coarse_box, None, "SAM returned an empty mask; kept the heatmap box"

        growth = box_area(refined) / max(box_area(coarse_box), 1e-8)
        if growth > MAX_SAM_AREA_GROWTH:
            return (
                coarse_box,
                None,
                f"SAM mask {growth:.1f}x larger than the prompt, likely background leak; "
                "kept the heatmap box",
            )

        return refined, mask, f"SAM mask kept (confidence {sam_score:.3f})"

    def save_debug_overlay(
        self,
        image_key: str,
        heatmap: np.ndarray,
        final_box: tuple[float, float, float, float],
        decision: str,
        score: float,
    ) -> None:
        """Write a heatmap overlay so every decision can be checked by eye.

        Args:
            image_key: Target image key.
            heatmap: Similarity heatmap for the winning prototype.
            final_box: The box that was written, normalised.
            decision: Tier name, used in the filename.
            score: Detection confidence, used in the filename.
        """
        image_path = resolve_image_path(self.image_dir, image_key)
        if image_path is None:
            return

        try:
            base = Image.open(image_path).convert("RGB")
            width, height = base.size

            floor, ceiling = float(heatmap.min()), float(heatmap.max())
            normalised = ((heatmap - floor) / (ceiling - floor + 1e-8)) ** 3

            colour = np.zeros((*heatmap.shape, 3), dtype=np.uint8)
            colour[..., 0] = (normalised * 255).astype(np.uint8)
            colour[..., 2] = ((1 - normalised) * 140).astype(np.uint8)
            overlay = Image.fromarray(colour).resize((width, height), Image.BICUBIC)

            blended = Image.blend(base, overlay, 0.4)
            draw = ImageDraw.Draw(blended)
            draw.rectangle(list(yolo_box_to_pixels(final_box, width, height)), outline=(0, 255, 0), width=4)

            blended.save(self.debug_dir / f"{decision.lower()}_{score:.3f}_{image_key}.jpg", quality=80)
        except Exception as e:
            print(f"  [-] {image_key}: could not write debug overlay ({e}).")

    def propagate(self) -> None:
        """Locate every seeded class across all unlabelled images."""
        prototypes = self.build_prototypes()
        if not prototypes:
            print("[-] No seeds found. Run Step 3a or 3b to annotate at least one image.")
            return

        seed_keys = {p.image_key for p in prototypes}
        targets = [key for key in self.image_keys if key not in seed_keys]

        by_class: dict[int, int] = {}
        for proto in prototypes:
            by_class[proto.class_id] = by_class.get(proto.class_id, 0) + 1

        print(f"\n[+] Built {len(prototypes)} prototype(s) from {len(seed_keys)} seed image(s).")
        for class_id, count in sorted(by_class.items()):
            print(f"    class {class_id}: {count} prototype(s)")
        print(f"[*] Locating them in {len(targets)} unlabelled image(s).")
        print(f"[*] Tiers -> AUTO >= {AUTO_ACCEPT_THRESHOLD} | REVIEW >= {REVIEW_THRESHOLD}\n")

        queue = load_queue(str(self.review_queue_path))
        auto = review = rejected = suppressed = 0

        for image_key in targets:
            grid = self.load_patch_grid(image_key)
            score, proto, heatmap = self.match(grid, prototypes)

            if score < REVIEW_THRESHOLD:
                print(f"  [-]      {image_key} | {score:.4f} | below review threshold")
                rejected += 1
                continue

            skip, previous = is_suppressed(queue, image_key, score)
            if skip:
                print(
                    f"  [SKIP  ] {image_key} | {score:.4f} | already rejected at "
                    f"{previous:.4f}, not improved enough to re-propose"
                )
                suppressed += 1
                continue

            result = self.localize(image_key, heatmap)
            if result is None:
                rejected += 1
                continue

            final_box, mask, note = result
            decision = "AUTO" if score >= AUTO_ACCEPT_THRESHOLD else "REVIEW"

            label_path = self.label_dir / f"{image_key}.txt"
            xc, yc, w, h = final_box
            with open(label_path, "w") as f:
                f.write(f"{proto.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            if mask is not None:
                cv2.imwrite(str(self.mask_dir / f"{image_key}.png"), mask.astype(np.uint8) * 255)

            self.save_debug_overlay(image_key, heatmap, final_box, decision, score)

            if decision == "AUTO":
                queue.setdefault("pending", {}).pop(image_key, None)
                auto += 1
            else:
                add_pending(
                    queue,
                    image_key,
                    {
                        "score": round(score, 4),
                        "seed_source": proto.image_key,
                        "class_id": proto.class_id,
                        "label_path": os.path.abspath(label_path),
                        "image_path": resolve_image_path(self.image_dir, image_key),
                        "image_key": image_key,
                        "flagged_at": datetime.now(UTC).isoformat(),
                        "status": "pending_review",
                        "method": "patch_prototype_sam",
                    },
                )
                review += 1

            print(
                f"  [{decision:6}] {image_key} | {score:.4f} | seed '{proto.image_key}' | "
                f"box ({xc:.3f}, {yc:.3f}, {w:.3f}, {h:.3f}) | {note}"
            )

        save_queue(queue, str(self.review_queue_path))

        print("\n[+] Propagation complete:")
        print(f"    - Prototypes used : {len(prototypes)}")
        print(f"    - Auto-accepted   : {auto}")
        print(f"    - Review queue    : {review} (see {self.review_queue_path})")
        print(f"    - Rejected        : {rejected}")
        if suppressed:
            print(f"    - Skipped         : {suppressed} previously rejected by a human")
        print(f"[*] Heatmap overlays written to {self.debug_dir} for visual inspection.")

        if review:
            print(f"[!] {review} image(s) awaiting approval in the GUI 'Review Queue'.")


if __name__ == "__main__":
    try:
        PatchPropagator().propagate()
    except Exception as e:
        print(f"[!] An error occurred: {e}")

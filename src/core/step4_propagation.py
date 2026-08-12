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

Every distinct hot region becomes its own box, not just the strongest one. A frame
holding three pallets must be labelled with three boxes: labelling only one tells
the detector that the other two are background, which is worse than leaving the
frame out of the dataset entirely.

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

from src.core.class_config import class_name, load_classes
from src.core.review_queue import REVIEW_QUEUE_PATH, add_pending, is_suppressed, load_queue, save_queue
from src.core.sam_engine import SamEngine, mask_to_yolo_box
from src.core.tiers import AUTO_ACCEPT_THRESHOLD, DETECTION_LEVEL, REVIEW_THRESHOLD
from src.core.yolo_format import (
    box_area,
    iou,
    read_yolo_boxes,
    write_yolo_boxes,
    yolo_box_to_pixels,
)

# A region qualifies on its similarity score, not on its size. A distant object
# can occupy a single patch cell, and discarding one-cell regions silently drops
# exactly the small instances multi-object support exists to catch. The absolute
# DETECTION_LEVEL above is what separates objects from noise; region size is not
# evidence either way.
MIN_REGION_CELLS = 1

# Upper bound on boxes emitted for one image, as a guard against a heatmap that
# fragments into dozens of specks.
MAX_OBJECTS_PER_IMAGE = 20

# Two detections overlapping by more than this are the same object, usually one
# object split into two heatmap regions by an occluder.
DUPLICATE_IOU = 0.55

# SAM occasionally latches onto the background instead of the prompted object.
# If its mask is this many times larger than the coarse box, the coarse box is
# kept instead and the case is flagged in the log.
MAX_SAM_AREA_GROWTH = 4.0

# The growth ratio alone is too strict for small prompts. A distant object can
# occupy a single patch cell, which is 0.13% of the frame, and any correct mask
# for it necessarily exceeds four times that. Growth is therefore allowed up to
# this fraction of the frame regardless of the ratio, so the guard still catches
# a leak into the background without discarding every small object.
SMALL_PROMPT_AREA_ALLOWANCE = 0.05

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
class RegionCandidate:
    """One connected hot region of a heatmap, before SAM refinement.

    Attributes:
        score: Highest cosine similarity inside the region.
        box: Coarse normalised box covering the region.
        peak: ``(row, col)`` of the strongest cell, used as a SAM point prompt.
        cell_count: How many patch cells the region spans.
        proto_index: Prototype that won the peak cell, deciding the class.
    """

    score: float
    box: tuple[float, float, float, float]
    peak: tuple[int, int]
    cell_count: int
    proto_index: int


@dataclass
class Detection:
    """One object located in a target image.

    Attributes:
        box: Final normalised YOLO box.
        mask: SAM mask when one was kept, otherwise None.
        score: Patch similarity that produced this detection.
        note: Short description of how the box was decided.
        class_id: Class of the prototype that matched this object.
        seed_key: Seed image the matching prototype came from.
    """

    box: tuple[float, float, float, float]
    mask: np.ndarray | None
    score: float
    note: str
    class_id: int
    seed_key: str


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

    def match(self, grid: np.ndarray, prototypes: list[SeedPrototype]) -> tuple[np.ndarray, np.ndarray]:
        """Score a target grid against every prototype, cell by cell.

        The winner is decided per cell rather than once for the whole frame. A
        frame can hold objects of different classes, and picking a single best
        prototype for the image would then label every object with whichever
        class happened to match strongest somewhere.

        Taking the maximum over prototypes rather than averaging them keeps the
        method robust to appearance variation: one seed shot in different light
        can still carry the match on its own.

        Args:
            grid: Target patch grid of shape ``(rows, cols, dim)``.
            prototypes: Candidate prototypes to compare against.

        Returns:
            tuple: The per-cell best similarity, and the index of the prototype
            that achieved it, both shaped ``(rows, cols)``.
        """
        normalised = unit(grid)
        stack = np.stack([normalised @ proto.vector for proto in prototypes])
        return stack.max(axis=0), stack.argmax(axis=0)

    def find_object_regions(self, heatmap: np.ndarray, winners: np.ndarray) -> list[RegionCandidate]:
        """Find every distinct object candidate in a heatmap, not just the strongest.

        The heatmap is thresholded at an absolute cosine level rather than one
        relative to its own peak. A relative level defines "hot" in terms of the
        single best match in the frame, which hides a second, slightly dimmer
        instance of the same object; an absolute level asks the same question of
        every region independently: would this region's best patch pass review?

        Each connected region above that level becomes one candidate, scored by
        its own strongest patch.

        Args:
            heatmap: Best cosine similarity per patch cell, shape ``(rows, cols)``.
            winners: Index of the prototype achieving that similarity, same shape.

        Returns:
            list: Candidates ordered by score, strongest first.
        """
        rows, cols = heatmap.shape
        binary = (heatmap >= DETECTION_LEVEL).astype(np.uint8)

        if not binary.any():
            return []

        count, labelled = cv2.connectedComponents(binary, connectivity=8)
        candidates: list[RegionCandidate] = []

        for label in range(1, count):
            member = labelled == label
            if int(member.sum()) < MIN_REGION_CELLS:
                continue

            masked = np.where(member, heatmap, -np.inf)
            peak_r, peak_c = np.unravel_index(int(masked.argmax()), masked.shape)

            occupied_rows = np.where(member.any(axis=1))[0]
            occupied_cols = np.where(member.any(axis=0))[0]
            r0, r1 = int(occupied_rows[0]), int(occupied_rows[-1]) + 1
            c0, c1 = int(occupied_cols[0]), int(occupied_cols[-1]) + 1

            candidates.append(
                RegionCandidate(
                    score=float(heatmap[peak_r, peak_c]),
                    box=(
                        (c0 + c1) / 2.0 / cols,
                        (r0 + r1) / 2.0 / rows,
                        (c1 - c0) / cols,
                        (r1 - r0) / rows,
                    ),
                    peak=(int(peak_r), int(peak_c)),
                    cell_count=int(member.sum()),
                    proto_index=int(winners[peak_r, peak_c]),
                )
            )

        candidates.sort(key=lambda c: -c.score)
        return candidates[:MAX_OBJECTS_PER_IMAGE]

    def localize_regions(
        self,
        image_key: str,
        heatmap: np.ndarray,
        candidates: list[RegionCandidate],
        prototypes: list[SeedPrototype],
    ) -> list[Detection] | None:
        """Refine every candidate region into a precise box with SAM.

        Each region keeps the class of the prototype that won its peak cell, so a
        frame holding two different kinds of object is labelled with both.

        Args:
            image_key: Target image key.
            heatmap: Best-per-cell similarity map.
            candidates: Regions found in that map.
            prototypes: The prototype list the region indices refer to.

        Returns:
            list | None: Accepted detections, or None when the image cannot be read.
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
        rows, cols = heatmap.shape

        # The image encoding is the expensive part of SAM and is independent of the
        # prompt, so it is computed once and reused for every candidate.
        self.sam.set_image(image_rgb)

        detections: list[Detection] = []

        for candidate in candidates:
            peak_r, peak_c = candidate.peak
            peak_xy = ((peak_c + 0.5) / cols * img_w, (peak_r + 0.5) / rows * img_h)

            mask, sam_score = self.sam.mask_from_prompt(
                box_xyxy=yolo_box_to_pixels(candidate.box, img_w, img_h),
                point_xy=peak_xy,
            )

            refined = mask_to_yolo_box(mask)
            if refined is None:
                box, kept_mask, note = candidate.box, None, "empty SAM mask; kept heatmap box"
            else:
                prompt_area = box_area(candidate.box)
                allowed = max(prompt_area * MAX_SAM_AREA_GROWTH, SMALL_PROMPT_AREA_ALLOWANCE)
                if box_area(refined) > allowed:
                    box, kept_mask = candidate.box, None
                    note = (
                        f"SAM mask covers {box_area(refined):.1%} of the frame against an "
                        f"allowance of {allowed:.1%}, likely background leak; kept heatmap box"
                    )
                else:
                    box, kept_mask = refined, mask
                    note = f"SAM {sam_score:.3f}"

            proto = prototypes[candidate.proto_index]
            detection = Detection(
                box=box,
                mask=kept_mask,
                score=candidate.score,
                note=note,
                class_id=proto.class_id,
                seed_key=proto.image_key,
            )

            duplicate_of = next(
                (i for i, existing in enumerate(detections) if iou(box, existing.box) > DUPLICATE_IOU),
                None,
            )
            if duplicate_of is not None:
                # One object split across two heatmap regions, usually by an
                # occluder. Keep the stronger detection rather than emitting the
                # same object twice.
                if candidate.score > detections[duplicate_of].score:
                    detections[duplicate_of] = detection
                continue

            detections.append(detection)

        return detections

    def save_debug_overlay(
        self,
        image_key: str,
        heatmap: np.ndarray,
        detections: list[Detection],
        decision: str,
        score: float,
    ) -> None:
        """Write a heatmap overlay so every decision can be checked by eye.

        Args:
            image_key: Target image key.
            heatmap: Similarity heatmap for the winning prototype.
            detections: Every box written for this image.
            decision: Tier name, used in the filename.
            score: Best detection confidence, used in the filename.
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

            for detection in detections:
                x0, y0, x1, y1 = yolo_box_to_pixels(detection.box, width, height)
                # Green once the box would be accepted outright, amber while it
                # still needs a human, so a mixed image is obvious at a glance.
                colour = (0, 255, 0) if detection.score >= AUTO_ACCEPT_THRESHOLD else (255, 176, 0)
                draw.rectangle([x0, y0, x1, y1], outline=colour, width=4)
                draw.text((x0 + 6, max(0, y0 - 16)), f"{detection.score:.3f}", fill=colour)

            name = f"{decision.lower()}_{score:.3f}_n{len(detections)}_{image_key}.jpg"
            blended.save(self.debug_dir / name, quality=80)
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
        auto = review = rejected = suppressed = objects = 0

        for image_key in targets:
            grid = self.load_patch_grid(image_key)
            heatmap, winners = self.match(grid, prototypes)
            score = float(heatmap.max())

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

            candidates = self.find_object_regions(heatmap, winners)
            detections = self.localize_regions(image_key, heatmap, candidates, prototypes)
            if not detections:
                rejected += 1
                continue

            # An image is only accepted outright when every box in it would be.
            # One uncertain instance is enough reason for a human to look at the
            # frame, because a missed object teaches the detector that the object
            # is background.
            weakest = min(d.score for d in detections)
            best = max(detections, key=lambda d: d.score)
            decision = "AUTO" if weakest >= AUTO_ACCEPT_THRESHOLD else "REVIEW"

            label_path = self.label_dir / f"{image_key}.txt"
            write_yolo_boxes(
                str(label_path),
                [(d.class_id, *d.box) for d in detections],
            )

            masks = [d.mask for d in detections if d.mask is not None]
            if masks:
                combined = np.zeros_like(masks[0], dtype=np.uint8)
                for mask in masks:
                    combined |= mask.astype(np.uint8)
                cv2.imwrite(str(self.mask_dir / f"{image_key}.png"), combined * 255)

            self.save_debug_overlay(image_key, heatmap, detections, decision, score)

            if decision == "AUTO":
                queue.setdefault("pending", {}).pop(image_key, None)
                auto += 1
            else:
                add_pending(
                    queue,
                    image_key,
                    {
                        "score": round(score, 4),
                        "weakest_score": round(weakest, 4),
                        "object_count": len(detections),
                        # One score per label line, in the same order, so the
                        # review editor can show which individual boxes were
                        # uncertain rather than only the frame's best score.
                        "box_scores": [round(d.score, 4) for d in detections],
                        "seed_source": best.seed_key,
                        "class_id": best.class_id,
                        "label_path": os.path.abspath(label_path),
                        "image_path": resolve_image_path(self.image_dir, image_key),
                        "image_key": image_key,
                        "flagged_at": datetime.now(UTC).isoformat(),
                        "status": "pending_review",
                        "method": "patch_prototype_sam",
                    },
                )
                review += 1

            objects += len(detections)
            names = load_classes()
            summary = ", ".join(
                f"{class_name(names, d.class_id)} {d.score:.3f} [{d.note}]" for d in detections
            )
            print(
                f"  [{decision:6}] {image_key} | best {score:.4f} | seed "
                f"'{best.seed_key}' | {len(detections)} object(s): {summary}"
            )

        save_queue(queue, str(self.review_queue_path))

        labelled = auto + review
        print("\n[+] Propagation complete:")
        print(f"    - Prototypes used : {len(prototypes)}")
        print(f"    - Auto-accepted   : {auto}")
        print(f"    - Review queue    : {review} (see {self.review_queue_path})")
        print(f"    - Rejected        : {rejected}")
        if suppressed:
            print(f"    - Skipped         : {suppressed} previously rejected by a human")
        if labelled:
            print(f"    - Objects boxed   : {objects} across {labelled} image(s)")
            print(f"                        ({objects / labelled:.2f} per labelled image)")
        print(f"[*] Heatmap overlays written to {self.debug_dir} for visual inspection.")

        if review:
            print(f"[!] {review} image(s) awaiting approval in the GUI 'Review Queue'.")


if __name__ == "__main__":
    try:
        PatchPropagator().propagate()
    except Exception as e:
        print(f"[!] An error occurred: {e}")

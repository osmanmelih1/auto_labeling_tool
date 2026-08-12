"""Step 3b - Manual Seeding from Hand-Drawn Boxes.

Turns the rough boxes drawn on the GUI canvas into precise seed labels. The
rectangles only have to enclose their objects; SAM finds the actual boundaries,
and each mask's extent becomes one line of the YOLO label. That refinement is the
point of the step, because Step 4 mean-pools DINOv3 patch tokens inside each seed
box, and a box padded with floor and wall dilutes the prototype with background.

A frame may carry several boxes, of different classes. Labelling only one object
in a frame that holds three would teach the detector that the other two are
background, so the whole set is written together.

The GUI and this step never call into each other. The boxes arrive as JSON at
``data/temp_seed.json``, in line with the file-based contract used between all
pipeline stages.

Input:  ``data/temp_seed.json`` (written by the GUI)
Output: ``data/labels/<image>.txt`` and ``data/masks/seed_mask_<image>.png``
"""

import json
import os

import cv2
import numpy as np

from src.core.class_config import class_name, load_classes
from src.core.sam_engine import SamEngine, mask_to_yolo_box

SEED_FILE = "data/temp_seed.json"
SAM_MODEL_PATH = "data/models/sam_vit_b_01ec64.pth"
LABEL_DIR = "data/labels"
MASK_DIR = "data/masks"


def read_seed_file(path: str = SEED_FILE) -> tuple[str, list[dict]] | None:
    """Read the image path and box list the GUI staged.

    The single-box form written by earlier versions is still accepted, so a
    stale file left over from a previous session does not fail the step.

    Args:
        path: Location of the seed file.

    Returns:
        tuple | None: The image path and its boxes, or None when the file is
        missing, unreadable or empty.
    """
    if not os.path.exists(path):
        print("[!] Error: No seed data found. Draw and confirm a box in the GUI first.")
        return None

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[!] Error: {path} could not be read ({e}).")
        return None

    image_path = data.get("image_path")
    if not image_path:
        print(f"[!] Error: {path} does not name an image.")
        return None

    if "boxes" in data:
        boxes = data["boxes"]
    elif "bbox" in data:
        boxes = [{"class_id": data.get("class_id", 0), "bbox": data["bbox"]}]
    else:
        boxes = []

    if not boxes:
        print("[!] Error: No boxes staged for this image. Draw one and press Enter.")
        return None

    return image_path, boxes


def main() -> None:
    """Refine every staged box with SAM and write them as one label file."""
    print("[*] Starting Manual Seeding (Bounding Box Prompt)...")

    seed = read_seed_file()
    if seed is None:
        return
    img_path, boxes = seed

    names = load_classes()
    print(f"[*] Loaded {len(boxes)} box(es) for {os.path.basename(img_path)}")
    for i, box in enumerate(boxes, start=1):
        x, y, w, h = box["bbox"]
        label = class_name(names, box.get("class_id", 0))
        print(f"    {i}. {label:16} X:{x} Y:{y} W:{w} H:{h}")

    image = cv2.imread(img_path)
    if image is None:
        print(f"[!] Error: Could not read image from {img_path}")
        return

    try:
        sam = SamEngine(checkpoint_path=SAM_MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # The image encoding is the expensive part of SAM and does not depend on the
    # prompt, so it is computed once and reused for every box in the frame.
    sam.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    lines: list[str] = []
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for i, box in enumerate(boxes, start=1):
        class_id = box.get("class_id", 0)
        x, y, w, h = box["bbox"]

        mask, score = sam.mask_from_prompt(box_xyxy=(x, y, x + w, y + h))
        yolo_box = mask_to_yolo_box(mask)

        if yolo_box is None:
            print(f"  [-] Box {i}: SAM returned an empty mask, skipped.")
            continue

        xc, yc, nw, nh = yolo_box
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
        combined_mask |= mask.astype(np.uint8)
        print(f"  [+] Box {i}: {class_name(names, class_id):16} SAM confidence {score:.4f}")

    if not lines:
        print("[!] No usable masks were produced, so no label was written.")
        return

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    os.makedirs(LABEL_DIR, exist_ok=True)
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
    with open(label_path, "w") as f:
        f.writelines(lines)
    print(f"[+] {len(lines)} seed label(s) saved to: {label_path}")

    os.makedirs(MASK_DIR, exist_ok=True)
    mask_path = os.path.join(MASK_DIR, f"seed_mask_{base_name}.png")
    cv2.imwrite(mask_path, combined_mask * 255)
    print(f"[+] Mask saved successfully to: {mask_path}")

    print("\n[+] Process finished successfully.")


if __name__ == "__main__":
    main()

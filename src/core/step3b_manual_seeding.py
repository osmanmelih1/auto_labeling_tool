"""Step 3b - Manual Seeding from a Hand-Drawn Box.

Turns a rough box drawn on the GUI canvas into a precise seed label. The user's
rectangle only has to enclose the object; SAM finds its actual boundary, and the
mask's extent becomes the YOLO label. That refinement is the point of the step,
because Step 4 mean-pools DINOv3 patch tokens inside the seed box, and a box
padded with floor and wall dilutes the prototype vector with background.

The GUI and this step never call into each other. The box arrives as JSON at
``data/temp_seed.json``, in line with the file-based contract used between all
pipeline stages.

Input:  ``data/temp_seed.json`` (written by the GUI)
Output: ``data/labels/<image>.txt`` and ``data/masks/seed_mask_<image>.png``
"""

import json
import os

import cv2
import numpy as np

from src.core.sam_engine import SamEngine, mask_to_yolo_box

SEED_FILE = "data/temp_seed.json"
SAM_MODEL_PATH = "data/models/sam_vit_b_01ec64.pth"
LABEL_DIR = "data/labels"
MASK_DIR = "data/masks"


def main() -> None:
    """Read the GUI's box, refine it with SAM and write the seed label."""
    print("[*] Starting Manual Seeding (Bounding Box Prompt)...")

    if not os.path.exists(SEED_FILE):
        print("[!] Error: No seed data found. Please draw and confirm a box in the GUI first.")
        return

    with open(SEED_FILE) as f:
        data = json.load(f)

    img_path = data["image_path"]
    class_id = data["class_id"]
    x, y, w, h = data["bbox"]

    print(f"[*] Loaded seed for image: {os.path.basename(img_path)}")
    print(f"[*] Box Coordinates: X:{x}, Y:{y}, W:{w}, H:{h} | Class ID: {class_id}")

    image = cv2.imread(img_path)
    if image is None:
        print(f"[!] Error: Could not read image from {img_path}")
        return

    try:
        sam = SamEngine(checkpoint_path=SAM_MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    sam.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    print("[*] Refining the drawn box into a pixel-accurate mask...")
    mask, score = sam.mask_from_prompt(box_xyxy=(x, y, x + w, y + h))
    print(f"[+] Mask generated with confidence score: {score:.4f}")

    yolo_box = mask_to_yolo_box(mask)
    if yolo_box is None:
        print("[!] Warning: SAM returned an empty mask, so no label was written.")
        return

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    center_x, center_y, norm_w, norm_h = yolo_box

    os.makedirs(LABEL_DIR, exist_ok=True)
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
    with open(label_path, "w") as f:
        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
    print(f"[+] Seed label saved to: {label_path}")

    os.makedirs(MASK_DIR, exist_ok=True)
    mask_path = os.path.join(MASK_DIR, f"seed_mask_{base_name}.png")
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
    print(f"[+] Mask saved successfully to: {mask_path}")

    print("\n[+] Process finished successfully.")


if __name__ == "__main__":
    main()

"""Module: step3b_manual_seeding.py
Description: Reads the bounding box coordinates and class ID from the GUI's JSON file,
             feeds it to the Segment Anything Model (SAM) as a box prompt,
             and generates a perfect YOLO format label and mask.
"""

import json
import os

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


def main():
    print("[*] Starting Manual Seeding (Bounding Box Prompt)...")

    # 1. Load Data from GUI (JSON File)
    seed_file = "data/temp_seed.json"
    if not os.path.exists(seed_file):
        print("[!] Error: No seed data found. Please draw and confirm a box in the GUI first.")
        return

    # Read the JSON file created by app.py
    with open(seed_file) as f:
        data = json.load(f)

    img_path = data["image_path"]
    class_id = data["class_id"]
    x, y, w, h = data["bbox"]

    print(f"[*] Loaded seed for image: {os.path.basename(img_path)}")
    print(f"[*] Box Coordinates: X:{x}, Y:{y}, W:{w}, H:{h} | Class ID: {class_id}")

    # 2. Initialize SAM Model
    # Make sure the model exists in the data/models/ directory.
    sam_checkpoint = "data/models/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cpu"  # Can be changed to "cuda" if a compatible GPU is available

    print(f"[*] Initializing SAM ({model_type}) on {device}...")
    try:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
    except Exception as e:
        print(f"[!] Error loading SAM model: {e}")
        print(f"[!] Please ensure the model weights exist at: {sam_checkpoint}")
        return

    # 3. Read and Prepare the Image
    image = cv2.imread(img_path)
    if image is None:
        print(f"[!] Error: Could not read image from {img_path}")
        return

    # Convert BGR (OpenCV default) to RGB (SAM requirement)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # 4. Format Bounding Box for SAM
    # SAM expects the box in [x_min, y_min, x_max, y_max] format
    input_box = np.array([x, y, x + w, y + h])

    print("[*] Generating mask using bounding box prompt...")
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    mask = masks[0]
    score = scores[0]

    print(f"[+] Mask generated with confidence score: {score:.4f}")

    # 5. Convert Mask to YOLO Format
    # Find contours to get the exact bounding box of the generated mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour in case of minor artifacts
        c = max(contours, key=cv2.contourArea)
        mx, my, mw, mh = cv2.boundingRect(c)
        img_h, img_w = mask.shape

        # Normalize coordinates for YOLO (center_x, center_y, width, height)
        center_x = (mx + mw / 2.0) / img_w
        center_y = (my + mh / 2.0) / img_h
        norm_w = mw / img_w
        norm_h = mh / img_h

        # Setup output paths
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        os.makedirs("data/labels", exist_ok=True)
        label_path = f"data/labels/{base_name}.txt"

        # Write to YOLO .txt file
        with open(label_path, "w") as f:
            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        print(f"[+] Perfect YOLO label saved to: {label_path}")

        # Save Mask Image for visual verification and Step 4 (Propagation)
        os.makedirs("data/masks", exist_ok=True)
        mask_path = f"data/masks/seed_mask_{base_name}.jpg"
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_img)
        print(f"[+] Mask saved successfully to: {mask_path}")
    else:
        print("[!] Warning: No contours found in the generated mask.")

    print("\n[+] Process finished successfully.")


if __name__ == "__main__":
    main()

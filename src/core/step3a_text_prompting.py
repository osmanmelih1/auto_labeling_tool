"""Module: step3a_text_prompting.py
Description: Zero-Shot Text Prompting (Path A).
             Uses Grounding DINO to find objects based on a text prompt,
             passes the bounding boxes to SAM for precise masking,
             and finally converts that perfect mask into a YOLO format bounding box (.txt).
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


class TextToMaskPipeline:
    def __init__(self, sam_model_path: str, sam_type: str = "vit_b", device: str = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing pipeline on {self.device}...")

        # 1. Load Grounding DINO
        print("[*] Loading Grounding DINO (Text-to-Box)...")
        self.dino_id = "IDEA-Research/grounding-dino-base"
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_id).to(self.device)

        # 2. Load SAM (Box-to-Mask)
        print("[*] Loading SAM Base (Box-to-Mask)...")
        if not os.path.exists(sam_model_path):
            raise FileNotFoundError(f"[!] SAM model not found at {sam_model_path}")

        self.sam = sam_model_registry[sam_type](checkpoint=sam_model_path)
        self.sam.to(device=self.device)
        self.sam_predictor = SamPredictor(self.sam)

    def mask_to_yolo_format(self, mask_array: np.ndarray, img_width: int, img_height: int):
        """Converts a binary mask to YOLO bounding box format (x_center, y_center, width, height)."""
        # Find the rows and columns where the mask is True (object pixels)
        rows = np.any(mask_array, axis=1)
        cols = np.any(mask_array, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        # Get the extreme points of the mask
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Calculate normalized YOLO coordinates
        x_center = ((xmin + xmax) / 2.0) / img_width
        y_center = ((ymin + ymax) / 2.0) / img_height
        box_width = (xmax - xmin) / img_width
        box_height = (ymax - ymin) / img_height

        return x_center, y_center, box_width, box_height

    def generate_mask_from_text(
        self,
        image_path: str,
        text_prompt: str,
        mask_output_path: str,
        label_output_path: str,
        class_id: int = 0,
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
    ):
        print(f"\n[*] Processing image: {image_path}")
        print(f"[*] Searching for: '{text_prompt}'")

        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        img_width, img_height = image_pil.size

        # --- STEP 1: Grounding DINO (Find the Box) ---
        inputs = self.dino_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, text_threshold=text_threshold, target_sizes=[(img_height, img_width)]
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]

        # Manual threshold filtering
        keep_indices = scores > box_threshold
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]

        if len(boxes) == 0:
            print(
                f"[-] AI could not find any '{text_prompt}' in the image. Triggering Fallback (Path B) is recommended."
            )
            return False

        print(f"[+] Found {len(boxes)} matching object(s). Highest confidence: {scores[0]:.2f}")
        best_box = boxes[0].cpu().numpy()

        # --- STEP 2: SAM (Create the Perfect Mask) ---
        print("[*] Generating pixel-perfect mask with SAM...")
        self.sam_predictor.set_image(image_np)

        masks, _, _ = self.sam_predictor.predict(box=best_box, multimask_output=False)

        mask_array = masks[0]

        # --- STEP 3: Save the Mask Image ---
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))
        Path(mask_output_path).parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(mask_output_path)
        print(f"[+] Mask saved to: {mask_output_path}")

        # --- STEP 4: Generate and Save YOLO Label (.txt) ---
        yolo_coords = self.mask_to_yolo_format(mask_array, img_width, img_height)

        if yolo_coords:
            x_c, y_c, w, h = yolo_coords
            yolo_line = f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n"

            Path(label_output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(label_output_path, "w") as f:
                f.write(yolo_line)
            print(f"[+] Perfect YOLO label saved to: {label_output_path}")
        else:
            print("[-] Could not generate YOLO coordinates from the mask.")

        return True


if __name__ == "__main__":
    SAM_MODEL_PATH = "data/models/sam_vit_b_01ec64.pth"
    MASK_DIR = "data/masks"
    LABEL_DIR = "data/labels"

    PROMPT_FILE = "data/current_prompt.json"

    try:
        # Read the dynamic prompt file generated by the GUI
        if not os.path.exists(PROMPT_FILE):
            print("[!] Error: Prompt file missing. Please enter a prompt in the GUI.")
            exit(1)

        with open(PROMPT_FILE) as f:
            prompt_data = json.load(f)

        target_image_path = prompt_data.get("image_path")
        target_class_id = prompt_data.get("class_id", 0)
        text_prompt = prompt_data.get("prompt", "")

        if not target_image_path or not os.path.exists(target_image_path):
            print(f"[!] Error: Target image not found at {target_image_path}")
            exit(1)

        # Adjust output paths based on the current image
        image_filename = os.path.basename(target_image_path)
        test_mask_path = os.path.join(MASK_DIR, f"mask_{image_filename}")
        txt_filename = image_filename.rsplit(".", 1)[0] + ".txt"
        test_label_path = os.path.join(LABEL_DIR, txt_filename)

        # Execute the pipeline
        pipeline = TextToMaskPipeline(sam_model_path=SAM_MODEL_PATH)
        pipeline.generate_mask_from_text(
            image_path=target_image_path,
            text_prompt=text_prompt,
            mask_output_path=test_mask_path,
            label_output_path=test_label_path,
            class_id=target_class_id,
        )

    except Exception as e:
        print(f"[!] An error occurred: {e}")

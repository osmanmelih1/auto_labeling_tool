"""
Module: utils_manual_seeder
Description: Interactive SAM Seeder for Fallback (Path B). 
             Opens a UI window for the user to click on the target object 
             in the seed image, then generates and saves the mask.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor


class InteractiveSamSeeder:
    def __init__(self, model_path: str, model_type: str = "vit_b", device: str = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing SAM ({model_type}) on {self.device}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] SAM model not found at {model_path}")

        self.sam = sam_model_registry[model_type](checkpoint=model_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def select_point_and_generate_mask(self, image_path: str, output_path: str) -> None:
        """
        Displays the image, waits for user to click on the target object, 
        generates the mask based on that coordinate, and saves it.
        """
        print(f"\n[*] Loading seed image: {image_path}")
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        self.predictor.set_image(image_np)

        # 1. Interactive UI for Point Selection
        print("[*] PLEASE LOOK AT THE POP-UP WINDOW.")
        print("[*] Click once on the object you want to label (e.g., a pallet or box).")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image_np)
        ax.set_title("CLICK ON THE TARGET OBJECT (Wait for UI to close automatically)")
        ax.axis('off')
        
        # Get one click from the user
        clicked_points = plt.ginput(1, timeout=-1)
        plt.close(fig)

        if not clicked_points:
            print("[-] No point selected. Exiting.")
            return

        target_x, target_y = int(clicked_points[0][0]), int(clicked_points[0][1])
        center_point = np.array([[target_x, target_y]])
        point_labels = np.array([1])  # Foreground

        print(f"\n[*] Generating mask for user coordinate: [X:{target_x}, Y:{target_y}]")
        
        # 2. Predict the mask
        masks, scores, logits = self.predictor.predict(
            point_coords=center_point,
            point_labels=point_labels,
            multimask_output=False,
        )

        mask_array = masks[0]
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))

        # 3. Save the mask
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_img.save(output_path)
        
        print(f"[+] Mask generated with confidence score: {scores[0]:.4f}")
        print(f"[+] Mask saved successfully to: {output_path}")


if __name__ == "__main__":
    MODEL_PATH = "data/models/sam_vit_b_01ec64.pth"
    INPUT_DIR = "data/deduplicated"
    OUTPUT_DIR = "data/masks"

    try:
        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            test_image_name = image_files[0]
            test_image_path = os.path.join(INPUT_DIR, test_image_name)
            test_output_path = os.path.join(OUTPUT_DIR, f"seed_mask_{test_image_name}")

            seeder = InteractiveSamSeeder(model_path=MODEL_PATH)
            seeder.select_point_and_generate_mask(image_path=test_image_path, output_path=test_output_path)
        else:
            print(f"[-] No images found in {INPUT_DIR}.")
            
    except Exception as e:
        print(f"[!] An error occurred: {e}")
"""
Module: step3_text_prompting
Description: Zero-Shot Text Prompting (Path A).
             Uses Grounding DINO to find objects based on a text prompt,
             then passes the bounding boxes to SAM for precise masking.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor


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

    def generate_mask_from_text(self, image_path: str, text_prompt: str, output_path: str, box_threshold: float = 0.3, text_threshold: float = 0.3):
        print(f"\n[*] Processing image: {image_path}")
        print(f"[*] Searching for: '{text_prompt}'")
        
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)

        # --- STEP 1: Grounding DINO (Find the Box) ---
        inputs = self.dino_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        
        # Post-process to get bounding boxes (Removed box_threshold to fix HuggingFace API change)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]]
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        
        # Kütüphane hatasını aşmak için box_threshold filtrelemesini manuel yapıyoruz
        keep_indices = scores > box_threshold
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]

        if len(boxes) == 0:
            print(f"[-] AI could not find any '{text_prompt}' in the image. Triggering Fallback (Path B) is recommended.")
            return False

        print(f"[+] Found {len(boxes)} matching object(s). Highest confidence: {scores[0]:.2f}")
        
        # Take the best box (highest score) for the seed mask
        best_box = boxes[0].cpu().numpy()

        # --- STEP 2: SAM (Create the Mask) ---
        print("[*] Generating pixel-perfect mask with SAM...")
        self.sam_predictor.set_image(image_np)
        
        masks, _, _ = self.sam_predictor.predict(
            box=best_box,
            multimask_output=False
        )

        # --- STEP 3: Save the Result ---
        mask_array = masks[0]
        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_img.save(output_path)
        
        print(f"[+] Success! Mask saved to: {output_path}")
        return True


if __name__ == "__main__":
    SAM_MODEL_PATH = "data/models/sam_vit_b_01ec64.pth"
    INPUT_DIR = "data/deduplicated"
    OUTPUT_DIR = "data/masks"
    
    PROMPT = "box." 

    try:
        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            test_image = image_files[0]
            test_img_path = os.path.join(INPUT_DIR, test_image)
            test_out_path = os.path.join(OUTPUT_DIR, f"text_mask_{test_image}")

            pipeline = TextToMaskPipeline(sam_model_path=SAM_MODEL_PATH)
            pipeline.generate_mask_from_text(
                image_path=test_img_path, 
                text_prompt=PROMPT, 
                output_path=test_out_path
            )
        else:
            print(f"[-] No images found in {INPUT_DIR}.")
            
    except Exception as e:
        print(f"[!] An error occurred: {e}")
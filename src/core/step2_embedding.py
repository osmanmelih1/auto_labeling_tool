"""
Module: step2_embedding
Description: Extracts semantic features from deduplicated images using the state-of-the-art 
             DINOv3 Base model and stores the output embeddings as .npy files. 
             These embeddings are essential for fast similarity searches in the pipeline.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class DinoEmbedder:
    """
    Loads a DINOv3 Base vision model and extracts embeddings from images.

    Attributes:
        input_dir (Path): Directory containing the deduplicated images.
        output_dir (Path): Directory where the .npy embeddings will be saved.
        device (torch.device): Compute device (CPU or CUDA).
        model (torch.nn.Module): The pre-trained DINOv3 Base model.
        transform (transforms.Compose): The image preprocessing pipeline.
    """

    def __init__(self, input_dir: str, output_dir: str, device: Optional[str] = None) -> None:
        """
        Initializes the DinoEmbedder, sets up the compute device, and loads the DINOv3 Base model.

        Args:
            input_dir (str): Path to deduplicated source images.
            output_dir (str): Path to store the resulting .npy embeddings.
            device (Optional[str]): Force 'cpu' or 'cuda'. If None, auto-detects.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"[*] Compute device selected: {self.device}")

        # Loading the DINOv3 Base model directly from Meta's repository
        print("[*] Loading DINOv3 Base (vitb14) model from torch hub...")
        self.model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb14")
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def process_images(self) -> None:
        """
        Iterates over the input directory, processes each image through the DINOv3 model,
        and saves the resulting feature vector as a .npy file.
        """
        try:
            image_files = [
                f for f in os.listdir(self.input_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
        except FileNotFoundError:
            print(f"[!] Error: The input directory {self.input_dir} does not exist.")
            return

        if not image_files:
            print(f"[-] No images found in {self.input_dir} to process.")
            return

        print(f"[*] Starting DINOv3 embedding extraction for {len(image_files)} images...")

        for img_name in image_files:
            img_path = self.input_dir / img_name
            npy_filename = f"{os.path.splitext(img_name)[0]}.npy"
            npy_path = self.output_dir / npy_filename

            # Skip if the embedding already exists (allows resuming interrupted processes)
            if npy_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)

                features = self.model(input_tensor)
                embedding = features.cpu().numpy().squeeze()

                np.save(npy_path, embedding)

            except Exception as e:
                print(f"[!] Failed to process {img_name}. Error: {e}")

        print(f"[+] Task finished successfully. Embeddings are stored in {self.output_dir}.")


if __name__ == "__main__":
    INPUT_FOLDER = "data/deduplicated"
    OUTPUT_FOLDER = "data/embeddings"
    
    embedder = DinoEmbedder(input_dir=INPUT_FOLDER, output_dir=OUTPUT_FOLDER)
    embedder.process_images()
"""
Module: step2_embedding
Description: Extracts semantic features from images using DINOv3 Base.
             Loads the model from a local safetensors file to bypass network blocks,
             saves individual embeddings, and merges them into a VDB.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file


class DinoEmbedder:
    """
    Loads a local DINOv3 Base vision model and extracts embeddings from images.

    Attributes:
        input_dir (Path): Directory containing the deduplicated images.
        output_dir (Path): Directory where the .npy embeddings will be saved.
        device (torch.device): Compute device (CPU or CUDA).
        model (torch.nn.Module): The pre-trained DINOv3 Base model.
        transform (transforms.Compose): The image preprocessing pipeline.
    """

    def __init__(self, input_dir: str, output_dir: str, device: Optional[str] = None) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"[*] Compute device selected: {self.device}")

        # Download the architecture skeleton from PyTorch Hub (pretrained=False disables weight downloading)
        print("[*] Loading DINOv3 Base (vitb16) architecture from torch hub...")
        self.model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16", pretrained=False)
        
        # Load the downloaded weights from the local safetensors file
        local_model_path = "data/models/dinov3_vitb16.safetensors"
        print(f"[*] Loading local model weights from {local_model_path}...")
        
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"[!] Model file not found at {local_model_path}. Please download it and place it in the models directory.")
            
        # Safely load the state_dict using safetensors
        state_dict = load_file(local_model_path, device="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        
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
        and saves the resulting feature vector as an individual .npy file.
        Finally, it triggers the creation of the vector database.
        """
        image_files = [
            f for f in os.listdir(self.input_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not image_files:
            print(f"[-] No images found in {self.input_dir} to process.")
            return

        print(f"[*] Starting DINOv3 embedding extraction for {len(image_files)} images...")

        for img_name in image_files:
            img_path = self.input_dir / img_name
            npy_filename = f"{os.path.splitext(img_name)[0]}.npy"
            npy_path = self.output_dir / npy_filename

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
        
        print("[+] Individual embeddings saved. Creating single VDB file...")
        self.create_vector_database()

    def create_vector_database(self) -> None:
        """
        Merges all individual .npy files into a single vector database file (.npz)
        for incredibly fast similarity searches in the next pipeline steps.
        """
        all_embeddings = {}
        
        npy_files = [
            f for f in os.listdir(self.output_dir) 
            if f.endswith('.npy') and f != "embeddings_db.npz"
        ]
        
        for npy_file in npy_files:
            img_key = npy_file.replace('.npy', '')
            emb_array = np.load(self.output_dir / npy_file)
            all_embeddings[img_key] = emb_array
            
        db_path = self.output_dir / "embeddings_db.npz"
        np.savez_compressed(db_path, **all_embeddings)
        print(f"[+] Master Vector Database created successfully at: {db_path}")


if __name__ == "__main__":
    INPUT_FOLDER = "data/deduplicated"
    OUTPUT_FOLDER = "data/embeddings"
    
    embedder = DinoEmbedder(input_dir=INPUT_FOLDER, output_dir=OUTPUT_FOLDER)
    embedder.process_images()
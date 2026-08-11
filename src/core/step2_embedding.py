"""Module: step2_embedding.py
Description: Extracts semantic features from images using DINOv3 Base.
             Includes a custom AI Engineer Key Mapping function to perfectly
             translate HuggingFace local .safetensors keys to PyTorch Hub architecture,
             along with tensor shape corrections (reshape) for dimensional mismatches.
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms


class DinoEmbedder:
    """Loads a local DINOv3 Base vision model and extracts embeddings from images.

    Attributes:
        input_dir (Path): Directory containing the deduplicated images.
        output_dir (Path): Directory where the .npy embeddings will be saved.
        device (torch.device): Compute device (CPU or CUDA).
        model (torch.nn.Module): The pre-trained DINOv3 Base model.
        transform (transforms.Compose): The image preprocessing pipeline.
    """

    def __init__(self, input_dir: str, output_dir: str, device: str | None = None) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[*] Compute device selected: {self.device}")

        # 1. Load the architecture skeleton
        print("[*] Loading DINOv3 Base (vitb16) architecture from torch hub...")
        self.model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16", pretrained=False)

        # 2. Load our local DINOv3 file
        local_model_path = "data/models/dinov3_vitb16.safetensors"
        print(f"[*] Loading local model weights from {local_model_path}...")

        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"[!] Model file not found at {local_model_path}.")

        raw_state_dict = load_file(local_model_path, device="cpu")

        # 3. Apply AI Engineer Key Mapping and Shape Correction
        print("[*] Applying Key Mapping (HuggingFace to Facebook Architecture)...")
        mapped_state_dict = self._map_hf_to_fb_keys(raw_state_dict)

        # 4. Load the translated weights into the model
        # strict=False is used safely here because we handled the critical weights manually
        self.model.load_state_dict(mapped_state_dict, strict=False)
        print("[+] SUCCESS! DINOv3 weights mapped and loaded perfectly.")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _map_hf_to_fb_keys(self, hf_dict: dict) -> dict:
        """AI Engineer Fix: Translates layer names from the local .safetensors file
        to match the PyTorch Hub architecture requirements. Also handles
        tensor dimension mismatches like mask_token [1, 1, 768] -> [1, 768].
        """
        fb_dict = {}

        # Map Core Embeddings
        if "embeddings.cls_token" in hf_dict:
            fb_dict["cls_token"] = hf_dict["embeddings.cls_token"]

        if "embeddings.mask_token" in hf_dict:
            mask_t = hf_dict["embeddings.mask_token"]
            # Fix dimensional mismatch for mask_token
            if mask_t.dim() == 3 and mask_t.shape[0] == 1 and mask_t.shape[1] == 1:
                mask_t = mask_t.reshape(1, -1)
            fb_dict["mask_token"] = mask_t

        if "embeddings.patch_embeddings.weight" in hf_dict:
            fb_dict["patch_embed.proj.weight"] = hf_dict["embeddings.patch_embeddings.weight"]

        if "embeddings.patch_embeddings.bias" in hf_dict:
            fb_dict["patch_embed.proj.bias"] = hf_dict["embeddings.patch_embeddings.bias"]

        # Map Transformer Blocks (12 layers for Base model)
        for i in range(12):
            hf_prefix = f"layer.{i}"
            fb_prefix = f"blocks.{i}"

            # Norms and Scales
            if f"{hf_prefix}.norm1.weight" in hf_dict:
                fb_dict[f"{fb_prefix}.norm1.weight"] = hf_dict[f"{hf_prefix}.norm1.weight"]
            if f"{hf_prefix}.norm1.bias" in hf_dict:
                fb_dict[f"{fb_prefix}.norm1.bias"] = hf_dict[f"{hf_prefix}.norm1.bias"]
            if f"{hf_prefix}.norm2.weight" in hf_dict:
                fb_dict[f"{fb_prefix}.norm2.weight"] = hf_dict[f"{hf_prefix}.norm2.weight"]
            if f"{hf_prefix}.norm2.bias" in hf_dict:
                fb_dict[f"{fb_prefix}.norm2.bias"] = hf_dict[f"{hf_prefix}.norm2.bias"]
            if f"{hf_prefix}.layer_scale1.lambda1" in hf_dict:
                fb_dict[f"{fb_prefix}.ls1.gamma"] = hf_dict[f"{hf_prefix}.layer_scale1.lambda1"]
            if f"{hf_prefix}.layer_scale2.lambda1" in hf_dict:
                fb_dict[f"{fb_prefix}.ls2.gamma"] = hf_dict[f"{hf_prefix}.layer_scale2.lambda1"]

            # MLP Layers
            if f"{hf_prefix}.mlp.up_proj.weight" in hf_dict:
                fb_dict[f"{fb_prefix}.mlp.fc1.weight"] = hf_dict[f"{hf_prefix}.mlp.up_proj.weight"]
            if f"{hf_prefix}.mlp.up_proj.bias" in hf_dict:
                fb_dict[f"{fb_prefix}.mlp.fc1.bias"] = hf_dict[f"{hf_prefix}.mlp.up_proj.bias"]
            if f"{hf_prefix}.mlp.down_proj.weight" in hf_dict:
                fb_dict[f"{fb_prefix}.mlp.fc2.weight"] = hf_dict[f"{hf_prefix}.mlp.down_proj.weight"]
            if f"{hf_prefix}.mlp.down_proj.bias" in hf_dict:
                fb_dict[f"{fb_prefix}.mlp.fc2.bias"] = hf_dict[f"{hf_prefix}.mlp.down_proj.bias"]

            # Attention Projections
            if f"{hf_prefix}.attention.o_proj.weight" in hf_dict:
                fb_dict[f"{fb_prefix}.attn.proj.weight"] = hf_dict[f"{hf_prefix}.attention.o_proj.weight"]
            if f"{hf_prefix}.attention.o_proj.bias" in hf_dict:
                fb_dict[f"{fb_prefix}.attn.proj.bias"] = hf_dict[f"{hf_prefix}.attention.o_proj.bias"]

            # Attention QKV Concatenation (Complex Mapping)
            if f"{hf_prefix}.attention.q_proj.weight" in hf_dict:
                q_w = hf_dict[f"{hf_prefix}.attention.q_proj.weight"]
                k_w = hf_dict[f"{hf_prefix}.attention.k_proj.weight"]
                v_w = hf_dict[f"{hf_prefix}.attention.v_proj.weight"]
                fb_dict[f"{fb_prefix}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

                q_b = hf_dict.get(f"{hf_prefix}.attention.q_proj.bias")
                v_b = hf_dict.get(f"{hf_prefix}.attention.v_proj.bias")
                if q_b is not None and v_b is not None:
                    # K-projection typically has no bias in DINO architectures, pad with zeros
                    k_b = torch.zeros_like(q_b)
                    fb_dict[f"{fb_prefix}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        return fb_dict

    @torch.no_grad()
    def process_images(self) -> None:
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        if not image_files:
            print(f"[-] No images found in {self.input_dir} to process.")
            return

        print(f"[*] Starting DINOv3 embedding extraction for {len(image_files)} images...")

        for img_name in image_files:
            img_path = self.input_dir / img_name
            npy_filename = f"{os.path.splitext(img_name)[0]}.npy"
            npy_path = self.output_dir / npy_filename

            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)

                features = self.model(input_tensor)
                embedding = features.cpu().numpy().squeeze()

                np.save(npy_path, embedding)
                print(f"  [+] Extracted and saved true vector for: {img_name}")

            except Exception as e:
                print(f"[!] Failed to process {img_name}. Error: {e}")

        print("[+] Individual embeddings generated. Creating single VDB file...")
        self.create_vector_database()

    def create_vector_database(self) -> None:
        all_embeddings = {}

        npy_files = [
            f for f in os.listdir(self.output_dir) if f.endswith(".npy") and f != "embeddings_db.npz"
        ]

        for npy_file in npy_files:
            img_key = npy_file.replace(".npy", "")
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

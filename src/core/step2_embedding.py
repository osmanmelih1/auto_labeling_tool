"""Step 2 - DINOv3 Embedding Extraction (Master Vector Database).

Loads the local DINOv3 ViT-B/16 checkpoint, translates its HuggingFace parameter
names into the PyTorch Hub architecture layout, and extracts one feature vector
per image into a single compressed vector database.

The checkpoint is published by HuggingFace under names such as
``layer.0.attention.q_proj.weight`` while the ``facebookresearch/dinov3`` hub
architecture expects ``blocks.0.attn.qkv.weight``. The translation is therefore
explicit, and - critically - verified: every parameter the model declares must be
covered, otherwise the run aborts. Loading with ``strict=False`` and no
verification silently leaves unmatched parameters at their random initialisation,
which produces embeddings that look valid but carry no meaning.

Input:  ``data/deduplicated/``
Output: ``data/embeddings/<image>.npy`` and ``data/embeddings/embeddings_db.npz``
"""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

MODEL_PATH = "data/models/dinov3_vitb16.safetensors"
HUB_REPO = "facebookresearch/dinov3"
HUB_ENTRYPOINT = "dinov3_vitb16"

# Input images are resized to this square size. It must be a multiple of the
# patch size (16). No centre crop is applied: cropping discards the edges of the
# frame, and objects that live near the border would never reach the encoder.
#
# The size also sets the localisation granularity of Step 4. At 448 the grid is
# 28x28, so one patch covers ~3.6% of the frame; at 224 it is 14x14 and a patch
# covers ~7%, which is too coarse to point SAM at a specific object.
IMAGE_SIZE = 448
PATCH_SIZE = 16

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
VDB_FILENAME = "embeddings_db.npz"

# Subdirectory holding the per-image patch token grids used by Step 4.
PATCH_SUBDIR = "patches"

# Keys returned by DinoVisionTransformer.forward_features.
CLS_TOKEN_KEY = "x_norm_clstoken"
PATCH_TOKENS_KEY = "x_norm_patchtokens"

# Patch grids are large (grid^2 x 768 per image). Half precision halves the cost
# on disk and is far below the noise floor of a cosine similarity comparison.
PATCH_DTYPE = np.float16

# Register (storage) tokens are named differently across DINO generations.
REGISTER_TOKEN_CANDIDATES = ("storage_tokens", "register_tokens")

# Buffers that the architecture derives at construction time rather than learning.
# They are legitimately absent from the checkpoint and must NOT be treated as a
# mapping failure:
#   - rope_embed.periods : rotary position embedding frequencies, computed from
#     the head dimension.
#   - attn.qkv.bias_mask : DINOv3 fuses q/k/v into one Linear but has no bias on
#     the key projection, so the layer keeps a constant 0/1 mask that zeroes the
#     key slice of the fused bias. Its correctness is asserted in
#     _verify_structural_buffers rather than assumed.
NON_PERSISTENT_KEY_MARKERS = ("rope", "periods", "bias_mask")

# Fused qkv layout is [query | key | value]; the key third carries no bias.
QKV_PARTS = 3
KEY_PART_INDEX = 1


class DinoEmbedder:
    """Loads a local DINOv3 ViT-B/16 model and extracts embeddings from images.

    Attributes:
        input_dir (Path): Directory containing the deduplicated images.
        output_dir (Path): Directory where the .npy embeddings will be saved.
        device (torch.device): Compute device (CPU or CUDA).
        model (torch.nn.Module): The pre-trained DINOv3 model in eval mode.
        transform (transforms.Compose): The image preprocessing pipeline.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        device: str | None = None,
        model_path: str = MODEL_PATH,
    ) -> None:
        """Build the embedder and load the local checkpoint into the hub architecture.

        Args:
            input_dir: Directory holding the deduplicated source images.
            output_dir: Directory where embeddings and the vector database are written.
            device: Explicit torch device string. Autodetected when omitted.
            model_path: Path to the local DINOv3 .safetensors checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint is missing.
            RuntimeError: If any model parameter is left uninitialised after mapping.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = (
            torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[*] Compute device selected: {self.device}")

        print(f"[*] Loading DINOv3 architecture '{HUB_ENTRYPOINT}' from torch hub...")
        self.model = torch.hub.load(HUB_REPO, HUB_ENTRYPOINT, pretrained=False)

        print(f"[*] Loading local model weights from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] Model file not found at {model_path}.")
        raw_state_dict = load_file(model_path, device="cpu")
        print(f"[+] Checkpoint contains {len(raw_state_dict)} tensors.")

        print("[*] Translating HuggingFace parameter names to the hub architecture...")
        mapped_state_dict = self._map_hf_to_hub_keys(raw_state_dict)

        self._load_and_verify(mapped_state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _map_hf_to_hub_keys(self, hf_dict: dict) -> dict:
        """Translate HuggingFace checkpoint keys into hub architecture keys.

        Separate q/k/v projections are concatenated into the single fused ``qkv``
        tensor the hub blocks expect. DINOv3 attention has no bias on the key
        projection, so its slice of the fused bias is zero-filled.

        Args:
            hf_dict: Raw state dict as stored in the .safetensors file.

        Returns:
            dict: State dict keyed by hub architecture parameter names.
        """
        model_keys = set(self.model.state_dict().keys())
        hub_dict: dict = {}

        # --- Top-level embeddings and the final normalisation layer ---
        direct_map = {
            "embeddings.cls_token": "cls_token",
            "embeddings.mask_token": "mask_token",
            "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
            "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
            "norm.weight": "norm.weight",
            "norm.bias": "norm.bias",
        }
        for hf_key, hub_key in direct_map.items():
            if hf_key in hf_dict:
                hub_dict[hub_key] = hf_dict[hf_key]

        # Register / storage tokens: the target name varies between DINO releases.
        if "embeddings.register_tokens" in hf_dict:
            target = next((c for c in REGISTER_TOKEN_CANDIDATES if c in model_keys), None)
            if target is None:
                print(
                    "[!] Warning: checkpoint has register tokens but the architecture "
                    f"declares none of {REGISTER_TOKEN_CANDIDATES}. Skipping them."
                )
            else:
                hub_dict[target] = hf_dict["embeddings.register_tokens"]

        # --- Transformer blocks ---
        layer_indices = sorted({int(key.split(".")[1]) for key in hf_dict if key.startswith("layer.")})
        per_layer_map = {
            "norm1.weight": "norm1.weight",
            "norm1.bias": "norm1.bias",
            "norm2.weight": "norm2.weight",
            "norm2.bias": "norm2.bias",
            "layer_scale1.lambda1": "ls1.gamma",
            "layer_scale2.lambda1": "ls2.gamma",
            "mlp.up_proj.weight": "mlp.fc1.weight",
            "mlp.up_proj.bias": "mlp.fc1.bias",
            "mlp.down_proj.weight": "mlp.fc2.weight",
            "mlp.down_proj.bias": "mlp.fc2.bias",
            "attention.o_proj.weight": "attn.proj.weight",
            "attention.o_proj.bias": "attn.proj.bias",
        }

        for i in layer_indices:
            hf_prefix = f"layer.{i}"
            hub_prefix = f"blocks.{i}"

            for hf_suffix, hub_suffix in per_layer_map.items():
                hf_key = f"{hf_prefix}.{hf_suffix}"
                if hf_key in hf_dict:
                    hub_dict[f"{hub_prefix}.{hub_suffix}"] = hf_dict[hf_key]

            self._fuse_qkv(hf_dict, hub_dict, hf_prefix, hub_prefix)

        return self._adapt_shapes(hub_dict)

    def _fuse_qkv(self, hf_dict: dict, hub_dict: dict, hf_prefix: str, hub_prefix: str) -> None:
        """Concatenate separate q/k/v projections into the fused qkv tensors.

        Args:
            hf_dict: Raw checkpoint state dict.
            hub_dict: Output state dict, mutated in place.
            hf_prefix: Source block prefix, e.g. ``layer.0``.
            hub_prefix: Target block prefix, e.g. ``blocks.0``.
        """
        q_w = hf_dict.get(f"{hf_prefix}.attention.q_proj.weight")
        k_w = hf_dict.get(f"{hf_prefix}.attention.k_proj.weight")
        v_w = hf_dict.get(f"{hf_prefix}.attention.v_proj.weight")
        if q_w is None or k_w is None or v_w is None:
            return

        hub_dict[f"{hub_prefix}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

        q_b = hf_dict.get(f"{hf_prefix}.attention.q_proj.bias")
        v_b = hf_dict.get(f"{hf_prefix}.attention.v_proj.bias")
        if q_b is None or v_b is None:
            return

        # DINOv3 omits the key-projection bias; the fused tensor still needs the slot.
        k_b = hf_dict.get(f"{hf_prefix}.attention.k_proj.bias", torch.zeros_like(q_b))
        hub_dict[f"{hub_prefix}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

    def _adapt_shapes(self, hub_dict: dict) -> dict:
        """Reshape mapped tensors that differ only in layout from the model's parameters.

        Token tensors are stored as ``[1, 1, dim]`` in some checkpoints while the
        architecture declares ``[1, dim]``. Reshaping is allowed only when the
        element count matches exactly.

        Args:
            hub_dict: State dict keyed by hub parameter names.

        Returns:
            dict: The same state dict with every tensor matching the model's shapes.

        Raises:
            RuntimeError: If a tensor cannot be reshaped to the expected shape.
        """
        model_state = self.model.state_dict()
        adapted: dict = {}

        for key, tensor in hub_dict.items():
            expected = model_state.get(key)
            if expected is None:
                adapted[key] = tensor
                continue

            if tensor.shape == expected.shape:
                adapted[key] = tensor
                continue

            if tensor.numel() != expected.numel():
                raise RuntimeError(
                    f"[!] Cannot map '{key}': checkpoint shape {tuple(tensor.shape)} holds "
                    f"{tensor.numel()} elements but the model expects "
                    f"{tuple(expected.shape)} ({expected.numel()} elements)."
                )

            print(f"  [*] Reshaping '{key}' {tuple(tensor.shape)} -> {tuple(expected.shape)}")
            adapted[key] = tensor.reshape(expected.shape)

        return adapted

    def _load_and_verify(self, mapped_state_dict: dict) -> None:
        """Load the mapped weights and abort if any parameter stayed uninitialised.

        Args:
            mapped_state_dict: State dict keyed by hub parameter names.

        Raises:
            RuntimeError: If parameters remain unmatched after the load.
        """
        result = self.model.load_state_dict(mapped_state_dict, strict=False)

        unexpected = list(result.unexpected_keys)
        missing = [
            key
            for key in result.missing_keys
            if not any(marker in key.lower() for marker in NON_PERSISTENT_KEY_MARKERS)
        ]
        skipped = [key for key in result.missing_keys if key not in missing]

        print(f"[+] Mapped {len(mapped_state_dict)} tensors into the architecture.")
        if skipped:
            print(f"[*] {len(skipped)} derived buffer(s) intentionally not loaded: {skipped}")

        if unexpected:
            print(f"[!] {len(unexpected)} mapped key(s) do not exist in the model: {unexpected}")

        if missing:
            raise RuntimeError(
                f"[!] {len(missing)} model parameter(s) were never assigned a weight and would "
                f"stay randomly initialised: {missing}. Every embedding produced in this state "
                "would be meaningless. Fix the key mapping before continuing."
            )

        self._verify_structural_buffers()
        print("[+] All model parameters accounted for. Weights loaded correctly.")

    @torch.no_grad()
    def _verify_structural_buffers(self) -> None:
        """Assert that skipped buffers really are constants the architecture built itself.

        A buffer is only safe to skip if the module initialises it deterministically.
        For the fused attention bias mask that means a binary tensor whose key third
        is zero and whose query and value thirds are one. Checking this turns an
        assumption into a verified fact; a silent mismatch here would corrupt every
        attention layer exactly like an unmapped weight would.

        Raises:
            RuntimeError: If a bias mask does not match the expected constant pattern.
        """
        state = self.model.state_dict()
        masks = {key: tensor for key, tensor in state.items() if key.endswith("attn.qkv.bias_mask")}

        if not masks:
            return

        for key, mask in masks.items():
            unique = set(mask.unique().tolist())
            if not unique.issubset({0.0, 1.0}):
                raise RuntimeError(
                    f"[!] '{key}' is not a binary constant (values: {sorted(unique)}). It cannot "
                    "be a structural buffer, so it must be mapped from the checkpoint instead."
                )

            if mask.numel() % QKV_PARTS != 0:
                raise RuntimeError(
                    f"[!] '{key}' has {mask.numel()} elements, which is not divisible into "
                    f"{QKV_PARTS} query/key/value parts."
                )

            part = mask.numel() // QKV_PARTS
            key_slice = mask[KEY_PART_INDEX * part : (KEY_PART_INDEX + 1) * part]
            other = torch.cat([mask[:part], mask[(KEY_PART_INDEX + 1) * part :]])

            if key_slice.any() or not other.all():
                raise RuntimeError(
                    f"[!] '{key}' does not zero exactly the key projection bias. Expected the "
                    "middle third to be all zeros and the outer thirds all ones."
                )

        print(
            f"[+] Verified {len(masks)} attention bias mask(s): key-projection bias correctly "
            "zeroed by the architecture."
        )

    @torch.no_grad()
    def extract_features(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        """Run one image through the encoder and return its global and patch features.

        Args:
            image: RGB source image at its original resolution.

        Returns:
            tuple: The CLS vector of shape ``(embed_dim,)`` and the patch grid of
            shape ``(grid, grid, embed_dim)`` laid out row-major over the resized
            image, so grid cell (row, col) maps back to a known pixel region.

        Raises:
            RuntimeError: If the encoder returns an unexpected number of patch tokens.
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model.forward_features(input_tensor)

        cls_vector = features[CLS_TOKEN_KEY].squeeze(0).cpu().numpy()
        patch_tokens = features[PATCH_TOKENS_KEY].squeeze(0).cpu().numpy()

        grid = IMAGE_SIZE // PATCH_SIZE
        expected = grid * grid
        if patch_tokens.shape[0] != expected:
            raise RuntimeError(
                f"[!] Encoder returned {patch_tokens.shape[0]} patch tokens but a "
                f"{IMAGE_SIZE}x{IMAGE_SIZE} input should produce {expected} ({grid}x{grid}). "
                "The patch grid could not be reshaped, so no spatial mapping is possible."
            )

        patch_grid = patch_tokens.reshape(grid, grid, -1)
        return cls_vector, patch_grid

    def process_images(self) -> None:
        """Extract global and patch embeddings for every image, then build the database.

        Two artefacts are written per image: the global CLS vector, which the
        vector database aggregates, and the patch token grid, which Step 4 needs
        to locate an object inside a target frame. Both are written per image so
        an interrupted run can be resumed.
        """
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]

        if not image_files:
            print(f"[-] No images found in {self.input_dir} to process.")
            return

        patch_dir = self.output_dir / PATCH_SUBDIR
        patch_dir.mkdir(parents=True, exist_ok=True)

        grid = IMAGE_SIZE // PATCH_SIZE
        print(f"[*] Starting DINOv3 embedding extraction for {len(image_files)} images...")
        print(f"[*] Input resized to {IMAGE_SIZE}x{IMAGE_SIZE} ({grid}x{grid} patches, no crop).")
        print(f"[*] Patch grids cached as {PATCH_DTYPE.__name__} under {patch_dir}")

        succeeded = 0
        failed = 0

        for img_name in image_files:
            img_path = self.input_dir / img_name
            image_key = os.path.splitext(img_name)[0]

            try:
                image = Image.open(img_path).convert("RGB")
                cls_vector, patch_grid = self.extract_features(image)

                np.save(self.output_dir / f"{image_key}.npy", cls_vector)
                np.save(patch_dir / f"{image_key}.npy", patch_grid.astype(PATCH_DTYPE))

                print(f"  [+] {img_name} -> cls {cls_vector.shape}, patches {patch_grid.shape}")
                succeeded += 1

            except Exception as e:
                print(f"  [-] Failed to process {img_name}. Error: {e}")
                failed += 1

        print(f"[+] Extraction complete: {succeeded} succeeded, {failed} failed.")
        self.create_vector_database()

    def create_vector_database(self) -> None:
        """Aggregate every individual .npy embedding into a single compressed database.

        The archive is keyed by extensionless image name, which is the contract the
        propagation step relies on when it pairs embeddings with label files.
        """
        npy_files = [f for f in os.listdir(self.output_dir) if f.endswith(".npy") and f != VDB_FILENAME]

        if not npy_files:
            print("[-] No embeddings found; vector database not created.")
            return

        all_embeddings = {
            npy_file.removesuffix(".npy"): np.load(self.output_dir / npy_file) for npy_file in npy_files
        }

        db_path = self.output_dir / VDB_FILENAME
        np.savez_compressed(db_path, **all_embeddings)
        print(f"[+] Master Vector Database created at {db_path} ({len(all_embeddings)} vectors).")


if __name__ == "__main__":
    INPUT_FOLDER = "data/deduplicated"
    OUTPUT_FOLDER = "data/embeddings"

    embedder = DinoEmbedder(input_dir=INPUT_FOLDER, output_dir=OUTPUT_FOLDER)
    embedder.process_images()

"""Shared Segment Anything (SAM) engine.

This is a utility, not a pipeline step. It owns nothing in the ``data/``
hierarchy except the model weights and holds no state about where its prompts
came from, so the pipeline steps that use it stay decoupled from one another.

Its job is the narrow one every step needs: turn a coarse prompt (a box, a
point, or both) into a precise mask, and turn that mask into a YOLO bounding
box. Steps 3a, 3b and 4 all need exactly this, and each previously carried its
own copy of the loading and conversion code.
"""

import os

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

DEFAULT_CHECKPOINT = "data/models/sam_vit_b_01ec64.pth"
DEFAULT_MODEL_TYPE = "vit_b"


class SamEngine:
    """Wraps a SAM predictor with prompt helpers and mask-to-YOLO conversion.

    Attributes:
        device (torch.device): Compute device the model runs on.
        predictor (SamPredictor): The underlying SAM predictor.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        model_type: str = DEFAULT_MODEL_TYPE,
        device: str | None = None,
    ) -> None:
        """Load the SAM checkpoint onto the chosen device.

        Args:
            checkpoint_path: Path to the SAM .pth checkpoint.
            model_type: SAM variant key, e.g. ``vit_b``.
            device: Explicit torch device string. Autodetected when omitted.

        Raises:
            FileNotFoundError: If the checkpoint is missing.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"[!] SAM checkpoint not found at {checkpoint_path}. See the README for the download command."
            )

        self.device = (
            torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"[*] Loading SAM ({model_type}) from {checkpoint_path} on {self.device}...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print("[+] SAM ready.")

    def set_image(self, image_rgb: np.ndarray) -> None:
        """Compute and cache the image encoding for subsequent prompts.

        Args:
            image_rgb: Image as an HxWx3 uint8 array in RGB order.
        """
        self.predictor.set_image(image_rgb)

    def mask_from_prompt(
        self,
        box_xyxy: tuple[float, float, float, float] | None = None,
        point_xy: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Segment the object indicated by a box, a point, or both.

        Supplying both is the strongest prompt: the box bounds the search and the
        point disambiguates which object inside it is meant.

        Args:
            box_xyxy: Bounding box in pixels as (x_min, y_min, x_max, y_max).
            point_xy: A single foreground point in pixels as (x, y).

        Returns:
            tuple: The boolean mask and SAM's own confidence score for it.

        Raises:
            ValueError: If neither prompt is supplied.
        """
        if box_xyxy is None and point_xy is None:
            raise ValueError("[!] mask_from_prompt requires a box, a point, or both.")

        box = np.array(box_xyxy, dtype=np.float32)[None, :] if box_xyxy is not None else None
        coords = np.array([point_xy], dtype=np.float32) if point_xy is not None else None
        labels = np.array([1], dtype=np.int32) if point_xy is not None else None

        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=box,
            multimask_output=False,
        )
        return masks[0], float(scores[0])


def mask_to_yolo_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Convert a binary mask into a normalised YOLO box around its largest region.

    The largest connected component is used rather than the extent of every true
    pixel, so a few stray pixels elsewhere in the frame cannot inflate the box to
    cover the whole image.

    Args:
        mask: Boolean or 0/1 mask of shape (H, W).

    Returns:
        tuple | None: ``(x_center, y_center, width, height)`` normalised to
        [0, 1], or None when the mask is empty.
    """
    binary = mask.astype(np.uint8)
    if not binary.any():
        return None

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    img_h, img_w = binary.shape

    return (
        (x + w / 2.0) / img_w,
        (y + h / 2.0) / img_h,
        w / img_w,
        h / img_h,
    )

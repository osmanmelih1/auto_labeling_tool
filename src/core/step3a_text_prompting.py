"""Step 3a - Zero-Shot Seeding from a Text Prompt.

Creates seed labels without anyone drawing a box. Grounding DINO turns a text
prompt such as ``pallet.`` into candidate boxes, SAM turns the best of those into
a pixel-accurate mask, and the mask's extent becomes a YOLO label.

Seeds produced here feed Step 4, where every label file becomes a prototype
vector. A wrong box therefore does not just mislabel one image, it poisons the
prototype pool for the entire propagation run, so this step keeps only the
highest-scoring detection and refuses to guess when nothing clears the threshold.

Grounding DINO expects lowercase prompts whose phrases end with a period; the
prompt is normalised to that form rather than trusting the caller.

Input:  ``data/current_prompt.json`` (written by the GUI)
Output: ``data/labels/<image>.txt`` and ``data/masks/mask_<image>.png``
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from src.core.sam_engine import SamEngine, mask_to_yolo_box
from src.core.yolo_format import box_area, read_yolo_boxes

GROUNDING_DINO_ID = "IDEA-Research/grounding-dino-base"
SAM_MODEL_PATH = "data/models/sam_vit_b_01ec64.pth"
MASK_DIR = "data/masks"
LABEL_DIR = "data/labels"
PROMPT_FILE = "data/current_prompt.json"

# A seed box larger than this fraction of the frame is reported rather than
# trusted. Nothing in this dataset legitimately fills half the image, and the
# characteristic zero-shot failure is a box around the whole scene.
FRAME_SHARE_WARNING = 0.5

# A detection this close to the box threshold only just qualified. Printed with a
# [+] it reads as a success, and the difference between "found it" and "found
# almost nothing" is invisible: a run scoring 0.326 against a threshold of 0.30
# produced a plausible-looking box for a prompt the frame did not really contain.
MARGINAL_SCORE_MARGIN = 0.1


def normalise_prompt(text: str) -> str:
    """Put a text prompt into the form Grounding DINO expects.

    The model was trained on lowercase phrases terminated by a period. Passing
    ``Pallet`` instead of ``pallet.`` measurably degrades detection, and the GUI
    cannot be relied upon to enforce it.

    Args:
        text: Raw prompt as typed by the user.

    Returns:
        str: Lowercase prompt ending in a period.
    """
    cleaned = text.strip().lower()
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


class TextToMaskPipeline:
    """Detects an object from a text prompt and turns it into a YOLO label.

    Attributes:
        device (str): Compute device shared by both models.
        dino_processor: Grounding DINO tokeniser and image processor.
        dino_model: Grounding DINO zero-shot detector.
        sam (SamEngine): Shared SAM engine used for box-to-mask refinement.
    """

    def __init__(
        self,
        sam_model_path: str = SAM_MODEL_PATH,
        sam_type: str = "vit_b",
        device: str | None = None,
    ) -> None:
        """Load Grounding DINO and SAM onto the chosen device.

        Args:
            sam_model_path: Path to the SAM checkpoint.
            sam_type: SAM variant key, e.g. ``vit_b``.
            device: Explicit torch device string. Autodetected when omitted.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing pipeline on {self.device}...")

        print("[*] Loading Grounding DINO (Text-to-Box)...")
        self.dino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_ID)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_ID).to(
            self.device
        )

        self.sam = SamEngine(checkpoint_path=sam_model_path, model_type=sam_type, device=self.device)

    @torch.no_grad()
    def detect(
        self, image: Image.Image, text_prompt: str, box_threshold: float, text_threshold: float
    ) -> tuple[np.ndarray, float] | None:
        """Find the single best box matching a text prompt.

        Grounding DINO does not return its boxes in score order, so the best one
        is selected by argmax over the scores rather than by taking the first.

        Args:
            image: RGB source image.
            text_prompt: Normalised prompt.
            box_threshold: Minimum detection score to keep a box.
            text_threshold: Minimum token score used during post-processing.

        Returns:
            tuple | None: The best box as ``(x_min, y_min, x_max, y_max)`` in
            pixels with its score, or None when nothing clears the threshold.
        """
        width, height = image.size
        inputs = self.dino_processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        outputs = self.dino_model(**inputs)

        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=text_threshold,
            target_sizes=[(height, width)],
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]

        keep = scores > box_threshold
        boxes, scores = boxes[keep], scores[keep]

        if len(boxes) == 0:
            return None

        best = int(scores.argmax())
        print(f"[+] Found {len(boxes)} candidate(s). Best score: {float(scores[best]):.3f}")
        return boxes[best].cpu().numpy(), float(scores[best])

    def generate_mask_from_text(
        self,
        image_path: str,
        text_prompt: str,
        mask_output_path: str,
        label_output_path: str,
        class_id: int = 0,
        box_threshold: float = 0.3,
        text_threshold: float = 0.3,
    ) -> bool:
        """Detect, segment and write the seed label for one image.

        Args:
            image_path: Source image to annotate.
            text_prompt: What to look for, e.g. ``pallet``.
            mask_output_path: Where the mask image is written.
            label_output_path: Where the YOLO .txt label is written.
            class_id: YOLO class id to record.
            box_threshold: Minimum detection score to keep a box.
            text_threshold: Minimum token score used during post-processing.

        Returns:
            bool: True when a label was written, False when nothing was found.
        """
        prompt = normalise_prompt(text_prompt)
        print(f"\n[*] Processing image: {image_path}")
        print(f"[*] Searching for: '{prompt}'")

        image = Image.open(image_path).convert("RGB")

        detection = self.detect(image, prompt, box_threshold, text_threshold)
        if detection is None:
            print(f"[-] No '{prompt}' found. Use Step 3b to draw the box by hand instead.")
            return False

        best_box, detection_score = detection

        if detection_score < box_threshold + MARGINAL_SCORE_MARGIN:
            print(
                f"[!] {detection_score:.3f} only just clears the {box_threshold:.2f} threshold."
                f"\n    A score this low usually means '{prompt}' is not really in this frame and"
                "\n    the model matched the nearest thing to it. SAM will still return a confident"
                "\n    mask around whatever that was. Check the box on the canvas before Step 4."
            )

        print("[*] Refining the box into a pixel-accurate mask with SAM...")
        self.sam.set_image(np.array(image))
        mask, sam_score = self.sam.mask_from_prompt(box_xyxy=tuple(best_box))
        print(f"[+] Mask generated (SAM confidence {sam_score:.3f}).")

        Path(mask_output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_output_path)
        print(f"[+] Mask saved to: {mask_output_path}")

        yolo_box = mask_to_yolo_box(mask)
        if yolo_box is None:
            print("[-] The mask was empty, so no YOLO label could be derived.")
            return False

        # This step produces exactly one box, so it replaces the label file rather
        # than appending to it — otherwise repeated runs would stack duplicates on
        # the same object. But replacing is destructive, and a frame that already
        # carried hand-drawn boxes loses them here. Saying so is the difference
        # between a deliberate overwrite and a silent one.
        existing = read_yolo_boxes(label_output_path)
        if existing:
            print(
                f"[!] {os.path.basename(label_output_path)} already held {len(existing)} box(es)."
                "\n    This step writes a single box, so those have been replaced. If they were"
                "\n    drawn by hand, use 'Edit Labels' to put them back."
            )

        x_c, y_c, w, h = yolo_box
        Path(label_output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(label_output_path, "w") as f:
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        print(f"[+] Seed label saved to: {label_output_path}")

        # Both models have now reported a confidence, and neither of them is a
        # statement that the right object was found: Grounding DINO is sure the
        # phrase matches somewhere, SAM is sure of the mask it drew. A box that
        # covers most of the frame is the shape that failure takes, and no
        # confidence number shows it.
        share = box_area((x_c, y_c, w, h))
        print(f"[*] The seed box covers {share:.1%} of the frame.")
        if share > FRAME_SHARE_WARNING:
            print(
                f"[!] That is more than {FRAME_SHARE_WARNING:.0%} of the image. A box this large is"
                "\n    usually the floor or the whole pallet stack rather than the object asked for."
                "\n    Open the frame before running Step 4: this label becomes a prototype, so a"
                "\n    wrong box here propagates to every similar frame."
            )
        return True


def main() -> None:
    """Read the GUI's prompt file and seed the image it points at."""
    if not os.path.exists(PROMPT_FILE):
        print("[!] Error: Prompt file missing. Please enter a prompt in the GUI.")
        return

    with open(PROMPT_FILE) as f:
        prompt_data = json.load(f)

    image_path = prompt_data.get("image_path")
    class_id = prompt_data.get("class_id", 0)
    text_prompt = prompt_data.get("prompt", "")

    if not image_path or not os.path.exists(image_path):
        print(f"[!] Error: Target image not found at {image_path}")
        return

    if not text_prompt.strip():
        print("[!] Error: The prompt is empty.")
        return

    image_filename = os.path.basename(image_path)
    stem = os.path.splitext(image_filename)[0]

    # The mask is written as PNG rather than inheriting the source image's
    # extension. A mask is a two-value image and JPEG is lossy: saved as .jpg it
    # came back with grey pixels along every edge. Nothing reads these masks back
    # today — the label is derived before the file is written — but a lossy mask is
    # wrong on its own terms, and Step 3b already writes PNG.
    mask_path = os.path.join(MASK_DIR, f"mask_{stem}.png")
    label_path = os.path.join(LABEL_DIR, f"{stem}.txt")

    pipeline = TextToMaskPipeline()
    pipeline.generate_mask_from_text(
        image_path=image_path,
        text_prompt=text_prompt,
        mask_output_path=mask_path,
        label_output_path=label_path,
        class_id=class_id,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[!] An error occurred: {e}")

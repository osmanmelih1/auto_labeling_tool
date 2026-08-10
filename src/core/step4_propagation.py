"""
Module: step4_propagation.py
Description: Confidence-Tiered Similarity-based Label Propagation (Path B).
             Loads the Master VDB (embeddings_db.npz), reads the verified seed image
             from temp_seed.json (created by GUI), finds visually similar images,
             and propagates the YOLO label into THREE confidence tiers instead of a
             single blind threshold:

               - AUTO-ACCEPT  (score >= AUTO_ACCEPT_THRESHOLD):
                   Label is copied directly. Treated as production-ready ground truth.

               - REVIEW QUEUE (REVIEW_THRESHOLD <= score < AUTO_ACCEPT_THRESHOLD):
                   Label is copied as a DRAFT (so downstream steps still see a .txt file),
                   but the image is also registered in data/review_queue.json so a human
                   can confirm or reject it in the GUI before it reaches the final dataset.

               - REJECTED (score < REVIEW_THRESHOLD):
                   Nothing is copied, nothing is tracked. Too dissimilar to trust.

             Rationale: blindly trusting everything above 0.85 risks false positives near
             the boundary. Splitting into tiers keeps full automation for high-confidence
             matches while routing borderline cases to human review instead of silently
             poisoning the dataset.
"""

import os
import shutil
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# --- Confidence Tier Configuration ---
AUTO_ACCEPT_THRESHOLD = 0.92   # >= this score: direct automatic acceptance
REVIEW_THRESHOLD = 0.82        # >= this and < AUTO: copied as draft, awaits human review
                               # < REVIEW_THRESHOLD: rejected, ignored

IMAGE_DIR = "data/deduplicated"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def resolve_image_path(image_dir: str, image_key: str) -> Optional[str]:
    """Finds the actual image file path for a given extensionless image_key.
    Required by the Review GUI to display image thumbnails."""
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(image_dir, image_key + ext)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


class VDBPropagator:
    def __init__(
        self,
        vdb_path: str,
        review_queue_path: str = "data/review_queue.json",
        image_dir: str = IMAGE_DIR,
    ):
        self.vdb_path = vdb_path
        self.review_queue_path = Path(review_queue_path)
        self.image_dir = image_dir
        print(f"[*] Loading Master VDB from {vdb_path}...")

        if not os.path.exists(vdb_path):
            raise FileNotFoundError(f"[!] VDB not found at {vdb_path}. Run Step 2 first.")

        self.database = np.load(vdb_path)
        self.image_names = self.database.files
        print(f"[+] Loaded embeddings for {len(self.image_names)} images.")

    def cosine_similarity(self, vec_a, vec_b):
        """Calculates the cosine similarity between two vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _load_review_queue(self) -> dict:
        """Loads data/review_queue.json. Starts fresh if missing or corrupt."""
        if self.review_queue_path.exists():
            with open(self.review_queue_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print("[!] review_queue.json corrupt, re-initializing.")
        return {"pending": {}}

    def _save_review_queue(self, queue_data: dict) -> None:
        self.review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.review_queue_path, "w") as f:
            json.dump(queue_data, f, indent=2)

    def propagate_labels(
        self,
        query_image_name: str,
        label_dir: str,
        auto_threshold: float = AUTO_ACCEPT_THRESHOLD,
        review_threshold: float = REVIEW_THRESHOLD,
    ):
        """Finds similar images and routes the YOLO label into confidence tiers."""
        query_key = query_image_name.replace(".jpg", "").replace(".png", "").replace(".jpeg", "")
        seed_label_path = os.path.join(label_dir, f"{query_key}.txt")

        if query_key not in self.image_names:
            print(f"[-] Query image '{query_key}' not found in VDB.")
            return

        if not os.path.exists(seed_label_path):
            print(f"[-] Seed label not found: {seed_label_path}")
            print("[-] Please run Step 3 (Text or Manual) to generate the initial label first.")
            return

        print(f"\n[*] Starting confidence-tiered propagation for seed: {query_key}")
        print(
            f"[*] Thresholds -> AUTO-ACCEPT: >= {auto_threshold} | "
            f"REVIEW: >= {review_threshold} | REJECTED: < {review_threshold}"
        )

        query_embedding = self.database[query_key]
        review_queue = self._load_review_queue()

        auto_count = 0
        review_count = 0
        rejected_count = 0

        for img_key in self.image_names:
            if img_key == query_key:
                continue

            emb = self.database[img_key]
            score = self.cosine_similarity(query_embedding, emb)
            target_label_path = os.path.join(label_dir, f"{img_key}.txt")

            if score >= auto_threshold:
                shutil.copy2(seed_label_path, target_label_path)
                review_queue["pending"].pop(img_key, None)
                print(f"  [AUTO]   {img_key} -> Score: {score:.4f} | Directly accepted.")
                auto_count += 1

            elif score >= review_threshold:
                shutil.copy2(seed_label_path, target_label_path)
                review_queue["pending"][img_key] = {
                    "score": round(float(score), 4),
                    "seed_source": query_key,
                    "label_path": os.path.abspath(target_label_path),
                    "image_path": resolve_image_path(self.image_dir, img_key),
                    "image_key": img_key,
                    "flagged_at": datetime.now(timezone.utc).isoformat(),
                    "status": "pending_review",
                }
                print(f"  [REVIEW] {img_key} -> Score: {score:.4f} | Draft label copied, AWAITING APPROVAL.")
                review_count += 1

            else:
                rejected_count += 1

        self._save_review_queue(review_queue)

        print(f"\n[+] Propagation complete for seed '{query_key}':")
        print(f"    - Auto-accepted : {auto_count}")
        print(f"    - Review queue  : {review_count}  (see {self.review_queue_path})")
        print(f"    - Rejected      : {rejected_count}")
        if review_count > 0:
            print(
                f"[!] {review_count} image(s) awaiting human approval. "
                f"Open the 'Review Queue' screen in the GUI to accept/reject them."
            )


if __name__ == "__main__":
    VDB_PATH = "data/embeddings/embeddings_db.npz"
    LABEL_DIR = "data/labels"
    SEED_FILE = "data/temp_seed.json"

    try:
        if not os.path.exists(SEED_FILE):
            print(f"[!] Error: Seed file not found at {SEED_FILE}.")
            print("[!] Please use the GUI to draw a bounding box and confirm it first.")
        else:
            with open(SEED_FILE, "r") as f:
                seed_data = json.load(f)

            actual_seed_image = os.path.basename(seed_data["image_path"])
            propagator = VDBPropagator(vdb_path=VDB_PATH)
            propagator.propagate_labels(
                query_image_name=actual_seed_image,
                label_dir=LABEL_DIR,
            )

    except Exception as e:
        print(f"[!] An error occurred: {e}")
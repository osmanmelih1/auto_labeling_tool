"""Module: step4_propagation.py
Description: Multi-Seed Confidence-Tiered Label Propagation (Path B).
             Loads the Master VDB and scans the 'data/labels' directory to identify ALL
             existing labeled images (the Seed Pool). It then compares every unlabelled
             image against ALL seeds in the pool, finds the best match (highest cosine
             similarity), and propagates the YOLO label based on Confidence Tiers.

             This creates an "Avalanche Effect" for Active Learning:
             10 seeds -> prop -> 150 labels -> prop -> 1500 labels.

               - AUTO-ACCEPT  (score >= 0.92)
               - REVIEW QUEUE (0.82 <= score < 0.92)
               - REJECTED     (score < 0.82)
"""

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# --- Confidence Tier Configuration ---
AUTO_ACCEPT_THRESHOLD = 0.92  # >= this score: direct automatic acceptance
REVIEW_THRESHOLD = 0.82  # >= this and < AUTO: copied as draft, awaits human review
# < REVIEW_THRESHOLD: rejected, ignored

IMAGE_DIR = "data/deduplicated"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def resolve_image_path(image_dir: str, image_key: str) -> str | None:
    """Finds the actual image file path for a given extensionless image_key."""
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
        if self.review_queue_path.exists():
            with open(self.review_queue_path) as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    print("[!] review_queue.json corrupt, re-initializing.")
        return {"pending": {}}

    def _save_review_queue(self, queue_data: dict) -> None:
        self.review_queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.review_queue_path, "w") as f:
            json.dump(queue_data, f, indent=2)

    def propagate_all_seeds(self, label_dir: str):
        """Multi-Seed Propagation: Uses all existing labels as seeds to label the rest."""
        print(f"\n[*] Scanning for existing seeds in {label_dir}...")

        seed_pool: list[tuple[str, np.ndarray, str]] = []
        target_pool: list[tuple[str, np.ndarray, str]] = []

        # 1. Separate images into Seeds (has label) and Targets (no label)
        for img_key in self.image_names:
            label_path = os.path.join(label_dir, f"{img_key}.txt")
            emb = self.database[img_key]

            if os.path.exists(label_path):
                seed_pool.append((img_key, emb, label_path))
            else:
                target_pool.append((img_key, emb, label_path))

        if not seed_pool:
            print("[-] No seeds found! Please run Step 3a or 3b first to create at least one label.")
            return

        print(f"[+] Found {len(seed_pool)} active seeds in the pool.")
        print(f"[*] Attempting to propagate labels to {len(target_pool)} unlabelled images...")
        print(f"[*] Thresholds -> AUTO-ACCEPT: >= {AUTO_ACCEPT_THRESHOLD} | REVIEW: >= {REVIEW_THRESHOLD}")

        review_queue = self._load_review_queue()
        auto_count = 0
        review_count = 0
        rejected_count = 0

        # 2. Compare each Target against ALL Seeds to find the best match
        for target_key, target_emb, target_label_path in target_pool:
            best_score = -1.0
            best_seed_key = None
            best_seed_label_path = None

            for seed_key, seed_emb, seed_label_path in seed_pool:
                score = self.cosine_similarity(target_emb, seed_emb)
                if score > best_score:
                    best_score = score
                    best_seed_key = seed_key
                    best_seed_label_path = seed_label_path

            # 3. Apply Confidence Tier logic to the BEST match
            if best_score >= AUTO_ACCEPT_THRESHOLD:
                shutil.copy2(best_seed_label_path, target_label_path)
                review_queue["pending"].pop(target_key, None)
                print(f"  [AUTO]   {target_key} -> Matched with '{best_seed_key}' | Score: {best_score:.4f}")
                auto_count += 1

            elif best_score >= REVIEW_THRESHOLD:
                shutil.copy2(best_seed_label_path, target_label_path)
                review_queue["pending"][target_key] = {
                    "score": round(float(best_score), 4),
                    "seed_source": best_seed_key,
                    "label_path": os.path.abspath(target_label_path),
                    "image_path": resolve_image_path(self.image_dir, target_key),
                    "image_key": target_key,
                    "flagged_at": datetime.now(UTC).isoformat(),
                    "status": "pending_review",
                }
                print(
                    f"  [REVIEW] {target_key} -> Matched with '{best_seed_key}' | Score: {best_score:.4f} (AWAITING APPROVAL)"
                )
                review_count += 1

            else:
                rejected_count += 1

        self._save_review_queue(review_queue)

        print("\n[+] Multi-Seed Propagation Complete:")
        print(f"    - Base Seeds Used : {len(seed_pool)}")
        print(f"    - Auto-accepted   : {auto_count}")
        print(f"    - Review queue    : {review_count} (see {self.review_queue_path})")
        print(f"    - Rejected        : {rejected_count}")

        if review_count > 0:
            print(f"[!] {review_count} image(s) awaiting human approval in the GUI 'Review Queue'.")


if __name__ == "__main__":
    VDB_PATH = "data/embeddings/embeddings_db.npz"
    LABEL_DIR = "data/labels"

    try:
        propagator = VDBPropagator(vdb_path=VDB_PATH)
        propagator.propagate_all_seeds(label_dir=LABEL_DIR)
    except Exception as e:
        print(f"[!] An error occurred: {e}")

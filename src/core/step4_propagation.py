"""
Module: step4_propagation.py
Description: Similarity-based Label Propagation (Path B).
             Loads the Master VDB (embeddings_db.npz), reads the verified seed image 
             from temp_seed.json (created by GUI), finds visually similar images,
             and automatically propagates the YOLO label to them based on a strict threshold.
"""

import os
import shutil
import json
import numpy as np

class VDBPropagator:
    def __init__(self, vdb_path: str):
        self.vdb_path = vdb_path
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

    def propagate_labels(self, query_image_name: str, label_dir: str, threshold: float = 0.85):
        """Finds similar images and automatically applies the YOLO label to them."""
        # Strip extension to match VDB keys
        query_key = query_image_name.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        seed_label_path = os.path.join(label_dir, f"{query_key}.txt")
        
        if query_key not in self.image_names:
            print(f"[-] Query image '{query_key}' not found in VDB.")
            return
            
        if not os.path.exists(seed_label_path):
            print(f"[-] Seed label not found: {seed_label_path}")
            print("[-] Please run Step 3 (Text or Manual) to generate the initial label first.")
            return

        print(f"\n[*] Starting label propagation for seed: {query_key}")
        query_embedding = self.database[query_key]
        
        propagated_count = 0

        # Compare the seed against all other images in the database
        for img_key in self.image_names:
            if img_key == query_key:
                continue  # Skip comparing with itself
            
            emb = self.database[img_key]
            score = self.cosine_similarity(query_embedding, emb)
            
            print(f"  [*] Checking {img_key}... Score: {score:.4f}")
            
            # If similarity is above realistic threshold, propagate the label
            if score >= threshold:
                target_label_path = os.path.join(label_dir, f"{img_key}.txt")
                shutil.copy2(seed_label_path, target_label_path)
                print(f"    [+] MATCH! Propagated label to: {img_key}.txt")
                propagated_count += 1
                
        print(f"\n[+] Propagation complete! Successfully labeled {propagated_count} similar images automatically.")


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
            
            # Get the actual image name from JSON
            actual_seed_image = os.path.basename(seed_data["image_path"])
            
            propagator = VDBPropagator(vdb_path=VDB_PATH)
            
            # Restored the strict industrial threshold to 0.85
            propagator.propagate_labels(
                query_image_name=actual_seed_image, 
                label_dir=LABEL_DIR, 
                threshold=0.85
            )
            
    except Exception as e:
        print(f"[!] An error occurred: {e}")
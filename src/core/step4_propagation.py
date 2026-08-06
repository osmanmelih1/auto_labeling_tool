"""
Module: step4_propagation
Description: Similarity-based Propagation (Path B).
             Loads the Master VDB (embeddings_db.npz), takes a seed image,
             finds the most visually similar images using Cosine Similarity,
             and prepares them for the next labeling phase.
"""

import os
import shutil
from pathlib import Path
import numpy as np

class VDBPropagator:
    def __init__(self, vdb_path: str):
        self.vdb_path = vdb_path
        print(f"[*] Loading Master VDB from {vdb_path}...")
        
        if not os.path.exists(vdb_path):
            raise FileNotFoundError(f"[!] VDB not found at {vdb_path}. Run Step 2 first.")
        
        # Load the .npz file into memory
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

    def find_similar_images(self, query_image_name: str, top_k: int = 5):
        """Finds the top_k most similar images to the query image."""
        # Strip extension to match VDB keys
        query_key = query_image_name.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        
        if query_key not in self.image_names:
            print(f"[-] Query image '{query_key}' not found in VDB.")
            return []

        print(f"\n[*] Starting similarity search for seed: {query_key}")
        query_embedding = self.database[query_key]

        results = {}
        # Compare the seed against all other images in the database
        for img_key in self.image_names:
            if img_key == query_key:
                continue  # Skip comparing with itself
            
            emb = self.database[img_key]
            score = self.cosine_similarity(query_embedding, emb)
            results[img_key] = score

        # Sort by highest similarity
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
        top_results = sorted_results[:top_k]

        print(f"\n[+] Top {len(top_results)} most similar images found:")
        for rank, (img_name, score) in enumerate(top_results, 1):
            print(f"    {rank}. {img_name} (Similarity: {score:.4f})")
            
        return top_results


if __name__ == "__main__":
    VDB_PATH = "data/embeddings/embeddings_db.npz"
    INPUT_DIR = "data/deduplicated"
    OUTPUT_DIR = "data/similar_results"
    
    try:
        # Get the first image as our test seed
        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"[-] No images found in {INPUT_DIR}.")
        else:
            test_seed = image_files[0]
            
            propagator = VDBPropagator(vdb_path=VDB_PATH)
            # Find the top 3 most similar images (since we only have 7 images total in test, 3 is a good number)
            similar_items = propagator.find_similar_images(query_image_name=test_seed, top_k=3)
            
            # Copy these similar images to a new folder so we can visually inspect them
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"\n[*] Copying top results to {OUTPUT_DIR} for visual inspection...")
            
            for img_key, score in similar_items:
                for ext in ['.jpg', '.jpeg', '.png']:
                    src_path = os.path.join(INPUT_DIR, img_key + ext)
                    if os.path.exists(src_path):
                        dst_path = os.path.join(OUTPUT_DIR, f"{score:.2f}_{img_key}{ext}")
                        shutil.copy2(src_path, dst_path)
                        break
            
            print(f"[+] Done! Check the '{OUTPUT_DIR}' folder to see if AI found visually similar scenes.")
            
    except Exception as e:
        print(f"[!] An error occurred: {e}")
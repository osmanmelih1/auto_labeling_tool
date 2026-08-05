"""
Module: step1_deduplication
Description: A module to identify and remove exact or near-duplicate images 
             from a dataset using CNN-based feature extraction. This ensures 
             a clean dataset before generating embeddings with DinoV3.
Author: [Senin Adın/Ekibin]
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set

from imagededup.methods import CNN


class ImageDeduplicator:
    """
    Identifies and filters out duplicate images in a given directory.

    This class utilizes a Convolutional Neural Network (CNN) to extract features
    from images and compares them to find duplicates based on a similarity threshold.

    Attributes:
        input_dir (Path): The directory containing the raw images.
        output_dir (Path): The directory where unique images will be saved.
        cnn (CNN): The CNN model instance from imagededup.
    """

    def __init__(self, input_dir: str, output_dir: str) -> None:
        """
        Initializes the ImageDeduplicator and ensures the output directory exists.

        Args:
            input_dir (str): Path to the source directory containing raw images.
            output_dir (str): Path to the target directory for unique images.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.cnn = CNN()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_duplicates(self) -> Dict[str, List[str]]:
        """
        Encodes images and finds duplicates based on structural similarity.

        Returns:
            Dict[str, List[str]]: A dictionary where the key is the image filename,
            and the value is a list of its duplicate filenames.
        """
        print(f"[*] Analyzing images in {self.input_dir}...")
        encodings = self.cnn.encode_images(image_dir=str(self.input_dir))
        
        # Threshold can be adjusted; 0.95 is generally safe for near-duplicates
        duplicates = self.cnn.find_duplicates(
            encoding_map=encodings, min_similarity_threshold=0.95
        )
        
        return duplicates

    def process_and_save_uniques(self) -> None:
        """
        Filters out duplicate images and copies only the unique ones 
        to the designated output directory.
        """
        duplicates_dict = self.find_duplicates()
        images_to_ignore: Set[str] = set()

        for image_name, duplicate_list in duplicates_dict.items():
            if image_name not in images_to_ignore:
                images_to_ignore.update(duplicate_list)

        try:
            all_images = set(os.listdir(self.input_dir))
        except FileNotFoundError:
            print(f"[!] Error: The directory {self.input_dir} was not found.")
            return

        unique_images = all_images - images_to_ignore

        print(f"[*] Total images processed: {len(all_images)}")
        print(f"[*] Duplicates found: {len(images_to_ignore)}")
        print(f"[*] Unique images to be saved: {len(unique_images)}")

        for img in unique_images:
            src_path = self.input_dir / img
            dst_path = self.output_dir / img
            shutil.copy2(src_path, dst_path)
            
        print(f"[+] Process complete. Unique images saved to {self.output_dir}.")


if __name__ == "__main__":
    # Example usage for testing
    INPUT_FOLDER = "../data/raw"
    OUTPUT_FOLDER = "../data/deduplicated"
    
    deduplicator = ImageDeduplicator(input_dir=INPUT_FOLDER, output_dir=OUTPUT_FOLDER)
    deduplicator.process_and_save_uniques()
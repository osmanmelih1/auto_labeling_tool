"""Step 5 - Dataset Export.

Packages the labelled images into the directory layout YOLO expects and writes
the ``data.yaml`` that points at it.

Only images that have a matching, non-empty label file are exported. An image
without a label is not an unlabelled example to YOLO, it is an explicit negative
sample, so silently including everything would teach the model that the object is
absent from frames nobody ever reviewed.

The split is stratified over the set of classes present in each image rather than
being a plain shuffle. With a few hundred images a uniform random split can put
every instance of a rare class in train, leaving validation unable to measure it
at all.

Class names come from ``data/classes.json`` and are never hardcoded: the same
program has to export a pallet dataset and a medical dataset unchanged.

Input:  ``data/deduplicated/``, ``data/labels/``, ``data/classes.json``
Output: ``datasets/{train,val}/{images,labels}/`` and ``datasets/data.yaml``
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

from src.core.class_config import class_name, load_classes

IMAGE_DIR = "data/deduplicated"
LABEL_DIR = "data/labels"
OUTPUT_DIR = "datasets"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

VAL_RATIO = 0.2
# Fixed so that re-exporting the same labels reproduces the same split. Without
# it, a model retrained after adding labels would be scored against a validation
# set that had partly leaked into its previous training run.
RANDOM_SEED = 42


def read_class_ids(label_path: Path) -> set[int]:
    """Collect the class ids referenced by a label file.

    Args:
        label_path: Path to a YOLO .txt label file.

    Returns:
        set[int]: Every class id appearing on a well-formed line.
    """
    ids: set[int] = set()
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    ids.add(int(parts[0]))
                except ValueError:
                    continue
    except OSError as e:
        print(f"  [-] Could not read {label_path.name}: {e}")
    return ids


class DatasetExporter:
    """Pairs images with labels and writes a YOLO-ready dataset directory."""

    def __init__(
        self,
        image_dir: str = IMAGE_DIR,
        label_dir: str = LABEL_DIR,
        output_dir: str = OUTPUT_DIR,
        val_ratio: float = VAL_RATIO,
        seed: int = RANDOM_SEED,
    ) -> None:
        """Record the source and destination locations.

        Args:
            image_dir: Directory holding the deduplicated images.
            label_dir: Directory holding the YOLO label files.
            output_dir: Directory to build the dataset in.
            val_ratio: Fraction of each stratum held out for validation.
            seed: Seed making the split reproducible.
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.val_ratio = val_ratio
        self.seed = seed

    def find_pairs(self) -> list[tuple[Path, Path, set[int]]]:
        """Match every image against its label file.

        Returns:
            list: One ``(image_path, label_path, class_ids)`` per usable pair.
        """
        pairs: list[tuple[Path, Path, set[int]]] = []
        unlabelled = 0
        empty = 0

        if not self.image_dir.exists():
            raise FileNotFoundError(f"[!] Image directory not found: {self.image_dir}")

        images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

        for image_path in images:
            label_path = self.label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                unlabelled += 1
                continue

            class_ids = read_class_ids(label_path)
            if not class_ids:
                empty += 1
                continue

            pairs.append((image_path, label_path, class_ids))

        print(f"[+] {len(pairs)} image/label pair(s) ready for export.")
        if unlabelled:
            print(f"[*] {unlabelled} image(s) have no label and were skipped.")
        if empty:
            print(f"[*] {empty} label file(s) were empty or malformed and were skipped.")

        return pairs

    def split(
        self, pairs: list[tuple[Path, Path, set[int]]]
    ) -> tuple[list[tuple[Path, Path, set[int]]], list[tuple[Path, Path, set[int]]]]:
        """Divide the pairs into training and validation sets.

        Images are grouped by the exact set of classes they contain and each
        group is split independently, so every class combination is represented
        on both sides whenever it has more than one image.

        Args:
            pairs: All usable image/label pairs.

        Returns:
            tuple: The training pairs and the validation pairs.
        """
        rng = random.Random(self.seed)
        strata: dict[frozenset[int], list] = defaultdict(list)
        for pair in pairs:
            strata[frozenset(pair[2])].append(pair)

        train: list = []
        val: list = []

        for stratum in sorted(strata, key=lambda s: sorted(s)):
            group = strata[stratum]
            rng.shuffle(group)

            # Round up so a stratum of two images still yields one validation
            # image, but never take the whole stratum.
            n_val = min(len(group) - 1, round(len(group) * self.val_ratio))
            n_val = max(n_val, 1) if len(group) > 1 else 0

            val.extend(group[:n_val])
            train.extend(group[n_val:])

        rng.shuffle(train)
        rng.shuffle(val)
        return train, val

    def write_split(self, name: str, pairs: list[tuple[Path, Path, set[int]]]) -> None:
        """Copy one split's images and labels into the output directory.

        Args:
            name: Split name, ``train`` or ``val``.
            pairs: Pairs belonging to this split.
        """
        image_out = self.output_dir / name / "images"
        label_out = self.output_dir / name / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for image_path, label_path, _ in pairs:
            shutil.copy2(image_path, image_out / image_path.name)
            shutil.copy2(label_path, label_out / label_path.name)

        print(f"    - {name:5}: {len(pairs)} image(s) -> {image_out}")

    def write_data_yaml(self, names: list[str], used_ids: set[int]) -> Path:
        """Write the data.yaml describing the exported dataset.

        Args:
            names: Class names in class-id order.
            used_ids: Class ids actually present in the exported labels.

        Returns:
            Path: The written data.yaml.
        """
        highest = max(used_ids) if used_ids else -1
        count = max(len(names), highest + 1)

        lines = [
            "# Generated by src/core/step5_export.py - do not edit by hand.",
            f"path: {self.output_dir.resolve().as_posix()}",
            "train: train/images",
            "val: val/images",
            "",
            "names:",
        ]
        lines.extend(f"  {class_id}: {class_name(names, class_id)}" for class_id in range(count))

        yaml_path = self.output_dir / "data.yaml"
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return yaml_path

    def report_distribution(
        self, split_name: str, pairs: list[tuple[Path, Path, set[int]]], names: list[str]
    ) -> None:
        """Print how many images of each class ended up in a split.

        Args:
            split_name: Split name to label the output with.
            pairs: Pairs belonging to this split.
            names: Class names in class-id order.
        """
        counts: dict[int, int] = defaultdict(int)
        for _, _, class_ids in pairs:
            for class_id in class_ids:
                counts[class_id] += 1

        summary = ", ".join(f"{class_name(names, cid)}={counts[cid]}" for cid in sorted(counts))
        print(f"    - {split_name:5}: {summary or 'no classes'}")

    def export(self) -> None:
        """Build the complete YOLO dataset directory and its data.yaml."""
        names = load_classes()
        if not names:
            print(
                "[!] No classes are defined in data/classes.json. Define them in the GUI "
                "before exporting, otherwise data.yaml will carry placeholder names."
            )

        pairs = self.find_pairs()
        if not pairs:
            print("[-] Nothing to export. Run the seeding and propagation steps first.")
            return

        if self.output_dir.exists():
            print(f"[*] Removing the previous export at {self.output_dir}...")
            shutil.rmtree(self.output_dir)

        train, val = self.split(pairs)

        print(f"\n[*] Writing dataset to {self.output_dir.resolve()}")
        self.write_split("train", train)
        self.write_split("val", val)

        used_ids = {class_id for _, _, ids in pairs for class_id in ids}
        yaml_path = self.write_data_yaml(names, used_ids)

        print("\n[+] Class distribution (images containing each class):")
        self.report_distribution("train", train, names)
        self.report_distribution("val", val, names)

        undefined = sorted(cid for cid in used_ids if cid >= len(names))
        if undefined:
            print(
                f"[!] Label files reference class id(s) {undefined} that have no name in "
                "data/classes.json. Placeholder names were written to data.yaml."
            )

        print("\n[+] Export complete:")
        print(f"    - Total exported : {len(pairs)}")
        print(f"    - Train / Val    : {len(train)} / {len(val)}")
        print(f"    - data.yaml      : {yaml_path.resolve()}")
        print(f"[*] Train with: yolo detect train data={yaml_path.resolve().as_posix()} model=yolov8n.pt")


if __name__ == "__main__":
    try:
        DatasetExporter().export()
    except Exception as e:
        print(f"[!] An error occurred: {e}")

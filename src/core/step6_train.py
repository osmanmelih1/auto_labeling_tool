"""Step 6 - YOLO Training.

Trains a YOLO detector on the dataset produced by Step 5, closing the loop the
whole pipeline exists to serve: a handful of hand-drawn boxes go in, a trained
object detector comes out.

This step deliberately does very little of its own. Ultralytics already handles
augmentation, scheduling and checkpointing well, so the value added here is the
checks around it: confirming the dataset is actually trainable, sizing the batch
to the data and the GPU, and reporting the metrics that say whether the labels
were any good.

That last point is the real purpose of training early. Validation mAP is the only
end-to-end measurement of label quality available: propagation can report high
confidence scores and still have produced boxes a detector cannot learn from.

Input:  ``datasets/data.yaml`` (written by Step 5)
Output: ``runs/<name>/weights/best.pt`` and the usual Ultralytics artefacts
"""

import os
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

DATA_YAML = "datasets/data.yaml"
OUTPUT_DIR = "runs"
RUN_NAME = "train"

# Nano is the right default here: the datasets this tool produces start small, and
# a larger backbone would overfit them while taking far longer to tell us whether
# the labels are usable.
#
# The path matters as much as the name. Ultralytics downloads a checkpoint it
# cannot find into the current working directory, which is how stray .pt files
# accumulate at the repository root; every other weight this project uses lives
# in data/models, and so should this one.
BASE_MODEL = "data/models/yolov8n.pt"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16

# Ultralytics spawns worker processes for data loading. This step is launched as a
# subprocess of the GUI, and on Windows process spawning inside a spawned process
# is a reliable source of hangs, so loading stays on the main process. With the
# dataset sizes involved it is not the bottleneck.
DATALOADER_WORKERS = 0

# Below this many training images, a run tells you the pipeline is wired up
# correctly and nothing at all about detection quality.
MIN_USEFUL_TRAIN_IMAGES = 50

# Below this many validation images, a high mAP means the model memorised them.
# Worth saying out loud next to the metrics, because a printed 0.99 is otherwise
# very easy to mistake for a result.
MIN_USEFUL_VAL_IMAGES = 20


def count_images(split_dir: Path) -> int:
    """Count image files in one split directory.

    Args:
        split_dir: Directory such as ``datasets/train/images``.

    Returns:
        int: Number of image files present.
    """
    if not split_dir.exists():
        return 0
    return sum(1 for p in split_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


class YoloTrainer:
    """Validates the exported dataset and trains a YOLO detector on it."""

    def __init__(
        self,
        data_yaml: str = DATA_YAML,
        base_model: str = BASE_MODEL,
        output_dir: str = OUTPUT_DIR,
        run_name: str = RUN_NAME,
        epochs: int = EPOCHS,
        image_size: int = IMAGE_SIZE,
        batch_size: int = BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        """Record the training configuration.

        Args:
            data_yaml: Dataset descriptor written by Step 5.
            base_model: Pre-trained checkpoint to fine-tune from.
            output_dir: Directory Ultralytics writes runs into.
            run_name: Subdirectory name for this run.
            epochs: Number of passes over the training set.
            image_size: Square size images are letterboxed to.
            batch_size: Requested batch size, clamped to the dataset size.
            device: Explicit torch device string. Autodetected when omitted.
        """
        self.data_yaml = Path(data_yaml)
        self.base_model = base_model
        self.output_dir = output_dir
        self.run_name = run_name
        self.epochs = epochs
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")

    def validate_dataset(self) -> tuple[int, int]:
        """Check the exported dataset is present and trainable.

        Returns:
            tuple: Number of training and validation images.

        Raises:
            FileNotFoundError: If Step 5 has not been run.
            ValueError: If either split is empty.
        """
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"[!] {self.data_yaml} not found. Run Step 5 (Export Dataset) first.")

        root = self.data_yaml.parent
        n_train = count_images(root / "train" / "images")
        n_val = count_images(root / "val" / "images")

        print(f"[+] Dataset found at {root.resolve()}")
        print(f"    - train: {n_train} image(s)")
        print(f"    - val  : {n_val} image(s)")

        if n_train == 0:
            raise ValueError("[!] The training split is empty. Nothing to learn from.")
        if n_val == 0:
            raise ValueError(
                "[!] The validation split is empty, so training could not be measured. "
                "Label more images and export again."
            )

        if n_train < MIN_USEFUL_TRAIN_IMAGES:
            print(
                f"\n[!] Only {n_train} training image(s). This run will confirm the pipeline "
                "is wired up correctly, but its metrics say nothing about detection quality. "
                f"Expect roughly {MIN_USEFUL_TRAIN_IMAGES}+ images per class before the "
                "numbers mean anything."
            )

        return n_train, n_val

    def effective_batch_size(self, n_train: int) -> int:
        """Clamp the batch size so it never exceeds the training set.

        Args:
            n_train: Number of training images.

        Returns:
            int: Batch size to actually use.
        """
        batch = min(self.batch_size, n_train)
        if batch != self.batch_size:
            print(f"[*] Batch size reduced from {self.batch_size} to {batch} to fit the dataset.")
        return max(batch, 1)

    def resolve_base_model(self) -> str:
        """Return the checkpoint to fine-tune from, keeping it under ``data/models``.

        Ultralytics fetches a checkpoint it cannot find, and it fetches it into
        the current working directory rather than to the path it was asked for.
        Left alone that scatters .pt files across the repository root. Here the
        download is allowed to happen and the file is then moved to where every
        other weight in this project lives, so the second run finds it locally.

        Returns:
            str: Path to a checkpoint that exists on disk, or the bare asset name
            on the first run, when only Ultralytics can supply it.
        """
        target = Path(self.base_model)
        if target.exists():
            return str(target)

        stray = Path(target.name)
        if stray.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stray), str(target))
            print(f"[*] Moved {stray} into {target.parent} where the other weights live.")
            return str(target)

        print(f"[*] {target} not found. Ultralytics will download {target.name} and it will be kept there.")
        return target.name

    def tidy_downloaded_checkpoints(self) -> None:
        """File away every checkpoint Ultralytics left in the working directory.

        Two arrive uninvited, and only the first was anticipated. The base model,
        when it was not already on disk. And a second, unrelated model that the
        automatic mixed precision check downloads to confirm the GPU gives the
        same answer with AMP on as with it off — which is where the stray
        yolo26n.pt at the project root came from, twice, after the first attempt
        at this only handled the base model.

        Both land in the current working directory regardless of the path they
        were asked for, both are checkpoints, and every other weight in this
        project lives in data/models.
        """
        store = Path(self.base_model).parent
        store.mkdir(parents=True, exist_ok=True)

        for stray in sorted(Path().glob("*.pt")):
            target = store / stray.name
            if target.exists():
                # A duplicate of a checkpoint already filed away. Keeping the
                # stored one and dropping this is the only outcome that leaves
                # the root clean without overwriting something in use.
                stray.unlink()
                print(f"[*] Discarded a duplicate {stray.name} from the project root.")
                continue

            shutil.move(str(stray), str(target))
            print(f"[+] Filed {stray.name} into {store}.")

    def report_device(self) -> None:
        """Print which device will be used and how much memory it has."""
        if self.device == "cpu":
            print("[!] Training on the CPU. This is slow; see the README for the CUDA setup.")
            return

        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[+] Training on GPU: {name} ({total:.1f} GB VRAM)")

    def train(self) -> None:
        """Validate the dataset, run training, and report the resulting metrics."""
        n_train, _ = self.validate_dataset()
        self.report_device()

        batch = self.effective_batch_size(n_train)

        print(f"\n[*] Fine-tuning {self.base_model} for {self.epochs} epoch(s)")
        print(f"[*] image size {self.image_size}, batch {batch}, workers {DATALOADER_WORKERS}\n")

        # An absolute project path is required. Ultralytics resolves a relative
        # one against its own settings directory and task name, so passing "runs"
        # lands the output in runs/detect/runs/<name> rather than runs/<name>.
        project = Path(self.output_dir).resolve()
        project.mkdir(parents=True, exist_ok=True)

        model = YOLO(self.resolve_base_model())
        results = model.train(
            data=str(self.data_yaml.resolve()),
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=batch,
            device=self.device,
            workers=DATALOADER_WORKERS,
            project=str(project),
            name=self.run_name,
            exist_ok=False,
            plots=True,
        )

        self.tidy_downloaded_checkpoints()
        self.report_results(results)

    def report_results(self, results) -> None:
        """Summarise the finished run and point at the weights.

        Args:
            results: The object returned by Ultralytics' train call.
        """
        save_dir = Path(getattr(results, "save_dir", self.output_dir))
        best = save_dir / "weights" / "best.pt"

        print("\n[+] Training complete.")

        metrics = getattr(results, "results_dict", None) or {}
        map50 = metrics.get("metrics/mAP50(B)")
        map5095 = metrics.get("metrics/mAP50-95(B)")
        precision = metrics.get("metrics/precision(B)")
        recall = metrics.get("metrics/recall(B)")

        if map50 is not None:
            print(f"    - mAP@50     : {map50:.4f}")
            print(f"    - mAP@50-95  : {map5095:.4f}")
            print(f"    - precision  : {precision:.4f}")
            print(f"    - recall     : {recall:.4f}")

            n_val = count_images(self.data_yaml.parent / "val" / "images")
            if n_val < MIN_USEFUL_VAL_IMAGES:
                print(
                    f"\n[!] These metrics come from {n_val} validation image(s). A near-perfect "
                    "score on a handful of images means the model memorised them, not that it "
                    "detects anything. Do not report these as a result."
                )
            else:
                print(
                    "\n[*] These numbers measure the labels as much as the model: propagation "
                    "can report high confidence and still produce boxes a detector cannot "
                    "learn from."
                )
        else:
            print("[!] Ultralytics returned no metrics dictionary; see the run directory.")

        print(f"\n[*] Run directory : {save_dir.resolve()}")
        if best.exists():
            print(f"[*] Best weights  : {best.resolve()}")
            print(f"[*] Try it with   : yolo detect predict model={best.as_posix()} source=<image>")
        else:
            print("[!] best.pt was not written; check the log above for a failed run.")


if __name__ == "__main__":
    # Ultralytics writes its own settings file; keep it out of the user's home so
    # the project stays self-contained.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(OUTPUT_DIR).resolve()))

    try:
        YoloTrainer().train()
    except Exception as e:
        print(f"[!] An error occurred: {e}")

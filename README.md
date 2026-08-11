# Sophtrun — Auto Labeling Tool

A PyQt6 desktop application that turns a folder of raw images into a training-ready
YOLO dataset with minimal human effort. The user annotates a handful of seed images;
the tool propagates those labels across the rest of the dataset using DINOv3 features
and SAM, and routes only the uncertain cases back to a human for review.

---

## Architecture

The pipeline is **decoupled**: every step is a standalone module that reads its input
from and writes its output to the `data/` hierarchy. No module imports another module's
internals — they communicate exclusively through files (JSON, TXT, images, NPZ).

```
data/raw/ ──[1]──> data/deduplicated/ ──[2]──> data/embeddings/embeddings_db.npz  (global CLS)
                            │                  data/embeddings/patches/*.npy      (28x28 grid)
                            │                                  │
                            │  [3a] text prompt                │
                            │  [3b] manual bounding box        │
                            ▼                                  ▼
                      data/labels/*.txt  ─── seeds ───> [4] patch prototype
                                                             → similarity heatmap
                                                             → SAM → localize
                                                                  │
                              data/labels/, data/masks/, data/debug/ <┘
                                                                  │
                                              data/review_queue.json ──[5] human review
                                                                  │
                                                                  ▼
                                                             datasets/ ──[6] YOLO training
```

| Step | Module | Purpose |
| --- | --- | --- |
| 1 | `src/core/step1_deduplication.py` | Removes near-duplicate images (`imagededup` CNN, threshold 0.95). |
| 2 | `src/core/step2_embedding.py` | Extracts DINOv3 ViT-B/16 features: one global CLS vector per image plus a cached 28×28 patch token grid. |
| 3a | `src/core/step3a_text_prompting.py` | Zero-shot seeding: Grounding DINO (text→box) → SAM (box→mask) → YOLO label. |
| 3b | `src/core/step3b_manual_seeding.py` | Manual seeding: GUI bounding box → SAM (box→mask) → YOLO label. |
| 4 | `src/core/step4_propagation.py` | Patch-level propagation with localisation. |
| 5 | `src/gui/app.py` (Review Queue) | Human-in-the-loop accept / reject for borderline matches. |
| 6 | `src/core/step5_export.py` | Packages the labels into a YOLO dataset and writes `data.yaml`. |
| 7 | `src/core/step6_train.py` | Fine-tunes a YOLO detector on the exported dataset. |

### Training measures the labels, not just the model

Validation mAP is the only end-to-end check on label quality the pipeline has.
Propagation can report high confidence scores and still have produced boxes a
detector cannot learn from, and no amount of staring at heatmaps will prove
otherwise. Training early, even on a small dataset, is how that gets caught.

Step 7 refuses to run on an empty split, clamps the batch size to the dataset,
and says plainly when the dataset is too small (under ~50 training images) for
its metrics to mean anything. Weights land in `runs/<name>/weights/best.pt`.

Three shared utilities sit outside the numbered steps:

- `src/core/sam_engine.py` loads SAM once and converts prompts to masks and masks
  to YOLO boxes for whichever step needs it.
- `src/core/class_config.py` reads and writes the project's class definitions.
- `src/core/review_queue.py` defines the structure of `data/review_queue.json`,
  which both the propagation step and the GUI read and write.

### Rejections are remembered

Propagation is run repeatedly as the seed pool grows. Without a record of what a
human already turned down, every rejected image would be proposed again on every
run — on a few thousand images that means rejecting the same wrong box over and
over.

Rejecting an entry therefore deletes its label file *and* records the score it
was rejected at. Later runs skip it while its score stays within
`REPROPOSE_MARGIN` (0.05) of that value. If a run scores it clearly higher, the
seed pool has learned something new and the image is worth a second look, so it
comes back.

The Review Queue screen shows how many images are currently suppressed and has a
**Clear Rejection History** button for when the seed pool has changed enough that
old rejections say more about the prototypes of the time than about the images.

### Defining classes

The tool is dataset-agnostic, so no class name appears anywhere in the source.
Classes live in `data/classes.json`, are edited through the GUI's **Manage
Classes** button, and are read by both the annotation canvas and the exporter:

```json
{ "classes": ["pallet", "forklift"] }
```

A class's position in the list is its YOLO class id, so existing labels would
change meaning if entries were reordered or removed from the middle. The class
manager therefore only appends, renames, and removes the last entry. Box colours
are generated from the class id rather than stored, so any number of classes gets
a readable palette without a per-project colour table.

Labelling a different domain means editing this file, not the code.

### How Step 4 locates an object

The seed's coordinates are never copied to the target. For each target image:

1. Every seed box is reduced to a **prototype vector** by mean-pooling the DINOv3
   patch tokens inside it.
2. The target's patch grid is compared against every prototype, giving a cosine
   **similarity heatmap**. The best score over all prototypes is the detection
   confidence, and the winning seed is recorded as provenance.
3. **Every** connected region above an absolute similarity level gives a **coarse
   box** in the target's own pixel space, not just the strongest one.
4. **SAM** turns each coarse box plus its peak point into a precise mask, whose
   bounding box becomes a YOLO label line.

A frame holding three pallets must be labelled with three boxes. Labelling only
the strongest tells the detector that the other two are background, which is
worse than leaving the frame out of the dataset entirely. The detection level is
absolute rather than relative to the frame's own peak, because a relative level
defines "hot" in terms of the single best match and hides every dimmer instance.

Region size is not used as evidence: a distant object can occupy a single patch
cell, so the similarity score alone decides. Detections overlapping by more than
55% are merged, which is usually one object split in two by an occluder.

An image is auto-accepted only when **every** box in it clears the threshold. One
uncertain instance is reason enough for a human to see the frame.

Every decision also writes a heatmap overlay to `data/debug/`, named
`tier_score_image.jpg`, so results can be checked by eye rather than trusted.

### Confidence tiers (Step 4)

| Patch similarity | Decision |
| --- | --- |
| `>= 0.86` | **AUTO-ACCEPT** — label written directly to `data/labels/`. |
| `0.78 – 0.86` | **REVIEW QUEUE** — written as a draft, queued in `data/review_queue.json`. |
| `< 0.78` | **REJECT** — ignored. |

These are calibrated on a small validation run and sit deliberately on the
cautious side, so most true matches reach the review queue rather than being
accepted outright. Tighten them once the score distribution over the full dataset
is known. Thresholds live in `src/core/step4_propagation.py`
(`AUTO_ACCEPT_THRESHOLD`, `REVIEW_THRESHOLD`) and are imported by the GUI so both
stay in sync.

Seed quality matters more than seed quantity: every label file in `data/labels/`
becomes a prototype, so one wrong box poisons the pool for the whole run.

---

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/osmanmelih1/auto_labeling_tool.git
cd auto_labeling_tool
uv sync
```

### Data directory skeleton

`data/` is git-ignored (it holds multi-GB weights and thousands of images), so create
it after cloning:

```bash
mkdir -p data/raw data/deduplicated data/embeddings data/labels data/masks data/models
```

`data/embeddings/patches/`, `data/debug/` and `data/review_queue.json` are created
by the steps that write them.

Drop your source images into `data/raw/`.

### Model weights

Both files go into `data/models/`. Paths are resolved relative to the project root.

**SAM (ViT-B) — `sam_vit_b_01ec64.pth`**

```bash
curl -L -o data/models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

**DINOv3 (ViT-B/16) — `dinov3_vitb16.safetensors`**

Download `model.safetensors` from
[`facebook/dinov3-vitb16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)
and rename it to `dinov3_vitb16.safetensors`. The repository is **gated** — log in to
Hugging Face and accept the DINOv3 license first.

**Grounding DINO** (`IDEA-Research/grounding-dino-base`) is fetched automatically by
`transformers` on first run of Step 3a and cached in the Hugging Face cache directory.

---

## Running

```bash
uv run main.py
```

`main.py` anchors the working directory to the project root, because every step module
addresses the data hierarchy with relative `data/...` paths by design.

Individual steps run as modules, not as file paths, so that they can import shared
helpers such as `src.core.sam_engine`:

```bash
uv run python -m src.core.step2_embedding
uv run python -m src.core.step4_propagation
```

### Typical workflow

1. **Load Image** — pick an image from `data/deduplicated/`.
2. **1. Deduplication** — populate `data/deduplicated/` from `data/raw/`.
3. **2. Embedding (VDB)** — build `data/embeddings/embeddings_db.npz`.
4. Seed a few images per class:
   - **3a. Text Prompting** — type a prompt (e.g. `pallet.`) and pick a class, or
   - **3b. Manual Seeding** — drag a bounding box on the canvas, press `Enter` to
     confirm, then run the step.
5. **4. Propagation** — spread the seed labels across the unlabelled images.
6. **5. Review Queue** — accept or reject the borderline results.
7. **6. Export Dataset** — build `datasets/` and `data.yaml`, ready for training.
8. **7. Train YOLO** — fine-tune a detector and report validation metrics.

Define your classes with **Manage Classes** before drawing the first box; the
canvas and the exporter both read them from `data/classes.json`.

**Canvas controls:** scroll to zoom, right-click drag to pan, left-click drag to draw a
box, `Enter` to confirm, `Esc` to cancel.

---

## Development

```bash
uv run ruff check .          # lint
uv run ruff check . --fix    # autofix
uv run ruff format .         # format
```

### Conventions

- **English only** — all code, comments, docstrings and log messages.
- **Docstrings required** — every module, class and function states what it does.
  Enforced by ruff's `D` ruleset (Google convention).
- **Log prefixes** — `[*]` for start / progress, `[+]` for success, `[-]` for failure or
  rejection, `[!]` for errors and warnings.
- **Module isolation** — a module must never import another step's internals. Steps talk
  to each other through files only.
- **No hardcoded absolute paths** — always resolve through the `data/` hierarchy.
- **Never commit** `data/`, `.venv/`, `__pycache__/`, model weights or embeddings.
  `.gitignore` and `.gitattributes` enforce this; line endings are normalized to LF.

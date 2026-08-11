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
data/raw/  ──[1]──> data/deduplicated/  ──[2]──> data/embeddings/embeddings_db.npz
                              │                              │
                              │  [3a] text prompt            │
                              │  [3b] manual bounding box    │
                              ▼                              ▼
                        data/labels/*.txt  <──[4] propagation (cosine similarity)
                        data/masks/*.png              │
                                                      ▼
                                          data/review_queue.json ──[5] human review
                                                      │
                                                      ▼
                                                 datasets/ ──[6] YOLO training
```

| Step | Module | Purpose |
| --- | --- | --- |
| 1 | `src/core/step1_deduplication.py` | Removes near-duplicate images (`imagededup` CNN, threshold 0.95). |
| 2 | `src/core/step2_embedding.py` | Extracts DINOv3 ViT-B/16 features and builds the master vector database. |
| 3a | `src/core/step3a_text_prompting.py` | Zero-shot seeding: Grounding DINO (text→box) → SAM (box→mask) → YOLO label. |
| 3b | `src/core/step3b_manual_seeding.py` | Manual seeding: GUI bounding box → SAM (box→mask) → YOLO label. |
| 4 | `src/core/step4_propagation.py` | Multi-seed propagation with confidence tiers. |
| 5 | `src/gui/app.py` (Review Queue) | Human-in-the-loop accept / reject for borderline matches. |
| 6 | _planned_ | Dataset export (`step5_export.py`) and classifier / YOLO training. |

### Confidence tiers (Step 4)

| Cosine score | Decision |
| --- | --- |
| `>= 0.92` | **AUTO-ACCEPT** — label written directly to `data/labels/`. |
| `0.82 – 0.92` | **REVIEW QUEUE** — written as a draft, queued in `data/review_queue.json`. |
| `< 0.82` | **REJECT** — ignored. |

Thresholds live in `src/core/step4_propagation.py` (`AUTO_ACCEPT_THRESHOLD`,
`REVIEW_THRESHOLD`) and are imported by the GUI so both stay in sync.

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

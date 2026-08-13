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
- `src/gui/label_editor.py` is the review canvas: it edits the boxes of one image
  in place and writes them back as a plain YOLO label file.

### Rejections are remembered

Propagation is run repeatedly as the seed pool grows. Without a record of what a
human already turned down, every rejected image would be proposed again on every
run — on a few thousand images that means rejecting the same wrong box over and
over.

Rejecting an entry therefore deletes its label file and its mask *and* records
the score it was rejected at. Later runs skip it while its score stays within
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

Each class can also carry the rule that decides what belongs in it:

```json
{ "classes": [
    { "name": "palet_3lu", "description": "Three stacked rows of egg trays" },
    { "name": "koli",      "description": "Cardboard boxes rather than trays" }
] }
```

Write those rules. A class boundary that lives only in someone's head is the main
reason datasets end up labelled inconsistently, and the person labelling next
month may not be the person who decided. The rule is shown in the class editor
and next to the class dropdown, where it is needed: while a box is being drawn.

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

The winning prototype is chosen per heatmap cell, not once per image, so a frame
holding two different kinds of object is labelled with both classes rather than
with whichever one matched strongest somewhere.

### Correcting labels in review

Accept-or-reject is the wrong pair of choices to offer, because propagation is
usually *almost* right. A box ten pixels short, one spurious box on the floor,
or the right box under the wrong class all forced a rejection — and rejecting
deletes the label, throwing away the accurate work SAM did on the rest of the
frame.

The review screen therefore edits, not just judges. The right-hand pane is a
real canvas: drag a box to move it, drag its handles to resize, drag on empty
background to add one, `Delete` to remove one, and `1`–`9` to set the selected
box's class. Every edit rewrites the label file immediately, so there is no
unsaved state to lose. Rejection is left for the case it was meant for: the
frame holds nothing worth labelling.

Boxes carry the confidence they were propagated with. Those that would have been
auto-accepted are drawn solid; those below the threshold are dashed and prefixed
with `?`. A reviewer's attention goes to the dashed ones, which are the only
part of the frame the machine was unsure about.

Class correction matters more than it sounds. Similarity matching tells
materials apart well and tells apart classes that differ only by *how many* of
something is stacked poorly: the patches of a two-row stack and a three-row
stack are identical, only the extent differs. Box geometry does not rescue this
either — measured on this dataset, the same class varies by 35% in aspect ratio
between the left and right of the frame, an order of magnitude more than the
difference between the classes. So the number keys exist: placing the box is the
expensive part and the machine has already done it.

`A` accepts and `R` rejects, and both move straight to the next frame, so a
queue can be worked without returning to the list between images.

Closing the screen reports the **median** seconds per frame and what the rest of
the queue would cost at that pace. Median rather than mean, because a review
session is not continuous — the reviewer answers the door, or thinks hard about
one difficult frame — and a mean over those gaps measures the interruptions
instead of the work. This number is the one that decides whether the pipeline
needs a trained model in the loop or merely a faster screen, and it is not worth
guessing at.

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
accepted outright. Thresholds live in `src/core/step4_propagation.py`
(`AUTO_ACCEPT_THRESHOLD`, `REVIEW_THRESHOLD`) and are imported by the GUI so both
stay in sync.

Do not adjust them by eye. Run the calibrator instead:

```bash
uv run python -m src.tools.calibrate_thresholds
```

It holds each seed out in turn and scores it against the prototypes built from
the other seeds, which yields a real sample of the positive distribution without
anyone annotating anything extra. It then scores every unlabelled image, prints
both distributions as histograms, tabulates what each candidate threshold admits
and costs, and suggests values. It changes nothing on disk.

Two readings matter. Whether the known positives sit above the review threshold —
if they do not, true matches never reach a human. And whether the unlabelled
scores are bimodal — a valley between two humps is the natural place to cut,
while one continuous mass means the features are not separating the object from
the background and no threshold will fix that.

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

**YOLOv8n** (`yolov8n.pt`) is downloaded by Ultralytics on the first training run,
along with a second, unrelated checkpoint its mixed-precision check uses. Both land
in the current working directory whatever path they were asked for, so Step 7 files
every checkpoint it finds at the root into `data/models/` when training ends.
Nothing is left beside `main.py`.

### Where things live

Everything the pipeline reads and writes is under `data/`, with two deliberate
exceptions: `datasets/` and `runs/`. Those are Ultralytics' own conventions, and a
`data.yaml` that points at the standard layout is worth more than consistency with
our own directory rule — the exported dataset should be usable by anyone who knows
YOLO and has never seen this tool. Both are git-ignored.

Inside `data/`, three kinds of file are mixed together and it is worth knowing
which is which before deleting anything:

| File | Kind | If you delete it |
| --- | --- | --- |
| `classes.json` | project configuration | the class scheme is gone; existing labels become meaningless |
| `review_queue.json` | pipeline state | pending decisions and the rejection history are lost |
| `temp_seed.json`, `current_prompt.json` | GUI-to-step messages | nothing; they are rewritten on the next action |

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
6. **5. Review Queue** — correct the borderline results, then accept or reject.
7. **6. Export Dataset** — build `datasets/` and `data.yaml`, ready for training.
8. **7. Train YOLO** — fine-tune a detector and report validation metrics.

Define your classes with **Manage Classes** before drawing the first box; the
canvas and the exporter both read them from `data/classes.json`.

**Seeding canvas:** scroll to zoom, right-click drag to pan, left-click drag to draw a
box, `Enter` to confirm, `Esc` to cancel, `Backspace` to delete the last box.

**Review editor:** scroll to zoom, right-click drag to pan, click a box to select it,
drag it to move, drag a handle to resize, drag on empty background to add a box,
`1`–`9` to set the selected box's class, `Delete` to remove it, `A` to accept the
frame and `R` to reject it — both move on to the next.

Both canvases zoom about the cursor, and zooming out stops once the whole image
fits. The scene is deliberately padded around the image: a `QGraphicsView` that
has nothing to scroll centres itself and ignores any attempt to shift it, which
is what makes an unpadded view zoom about its middle no matter where the pointer
is. Boxes are clamped to the image, so the padding is only somewhere to scroll.

---

## Development

```bash
uv run pytest                # tests
uv run ruff check .          # lint
uv run ruff check . --fix    # autofix
uv run ruff format .         # format
```

### Tests

68 tests, about a second, no GPU and no dataset. They run against a throwaway
project directory, so they never read the operator's classes or delete the
operator's labels.

| File | What it protects |
| --- | --- |
| `test_yolo_format.py` | The label format every step reads and writes: round-trip precision, and that one malformed line does not discard the rest of the file. |
| `test_review_queue.py` | Rejection memory. Too weak and a rejected frame returns on every run; too strong and one the tool has learned to recognise never gets a second chance. |
| `test_label_editor.py` | The review canvas, asserted against the label file on disk rather than the widget's own state, because saving on every edit is the whole contract. |
| `test_zoom.py` | That the point under the cursor stays under the cursor, on both canvases. |
| `test_review_dialog.py` | That every card carries a decision, that a class change rewrites one box and not its neighbour, and that rejecting removes the mask as well as the label. |
| `test_step6_paths.py` | That the YOLO checkpoint ends up in `data/models` and never at the project root. Skipped when the training dependencies are absent. |

Qt runs on the `offscreen` platform, so there is nothing to look at while they
run and nothing to install beyond `pytest`.

Two of these exist because of bugs that reached the working tree. Review cards
were once built with their Accept and Reject buttons stranded inside another
method, past its `return`; nothing about the code looked wrong. And zooming
followed the cursor only from whichever point Qt had last recorded, which no
amount of reading the wheel handler would have revealed. Anything that can only
be caught by running it belongs here.

### Conventions

- **English only** — all code, comments, docstrings and log messages.
- **Tests for anything only running can catch** — Qt wiring, geometry, and the
  side effects a decision has on disk. Pure formatting or a print statement does
  not need one.
- **Docstrings required** — every module, class and function states what it does.
  Enforced by ruff's `D` ruleset (Google convention).
- **Log prefixes** — `[*]` for start / progress, `[+]` for success, `[-]` for failure or
  rejection, `[!]` for errors and warnings.
- **Module isolation** — a module must never import another step's internals. Steps talk
  to each other through files only.
- **No hardcoded absolute paths** — always resolve through the `data/` hierarchy.
- **Never commit** `data/`, `.venv/`, `__pycache__/`, model weights or embeddings.
  `.gitignore` and `.gitattributes` enforce this; line endings are normalized to LF.

# Sophtrun / auto_labeling_tool

A PyQt6 desktop tool that labels an image dataset for YOLO object detection with
as little human attention as possible, then trains on what it produced. The
operator is a computer-vision engineer working on Windows with an RTX 3060
(6 GB). Conversation is in Turkish; everything written to disk is in English.

## Hard rules

- **No Turkish anywhere in the repository** — not in code, comments, docstrings,
  log messages, commit messages or documentation. Turkish belongs in the chat.
- **Give complete, runnable code.** Never abbreviate an edit with "the rest stays
  the same" or `...`. A file shown is a file that can be pasted over the old one.
- **Log prefixes:** `[*]` starting or informational, `[+]` success, `[-]` failure
  or rejection, `[!]` warning or refusal.
- **Keep the modules isolated.** A step knows only the files it reads and writes
  under `data/`, never another step's internals. Shared arithmetic lives in
  `src/core/yolo_format.py`; shared thresholds in `src/core/tiers.py`.
- **No hardcoded absolute paths.** Everything resolves against the `data/`
  hierarchy, relative to the project root.
- **Docstrings on every module, class and function** (ruff `D`, Google
  convention). Line length 110. Run `uv run ruff format .` then
  `uv run ruff check .` before committing.
- **The operator runs the commits.** Prepare the change and the message; hand him
  the commands. Long messages go in `commit_msg.txt` and are used with
  `git commit -F commit_msg.txt`.
- **`data/`, `.venv/`, `datasets/`, `runs/`, `__pycache__/` are gitignored** and
  must stay that way. Model weights and thousands of images once caused a 639 MB
  push timeout; only source belongs in the repository.

## Commands

```
uv run main.py                                   # the GUI
uv run pytest                                    # 160 tests
uv run ruff format . && uv run ruff check .
uv run python -m src.core.step5_export           # any step, by module path
uv run python -m src.tools.audit_labels          # any tool, likewise
```

## The pipeline

Each step leaves files where the next one looks for them; none imports another.

| Step | Does | Writes |
|---|---|---|
| 1 deduplication | drops near-duplicate frames | `data/deduplicated/` |
| 2 embedding | DINOv3 ViT-B/16 feature vectors | `data/embeddings/embeddings_db.npz` |
| 3a text prompting | Grounding DINO text → box, SAM box → mask | `data/labels/`, `data/masks/` |
| 3b manual seeding | operator's drag → SAM → tight box | same |
| 4 propagation | cosine similarity from seeds to unlabelled | `data/labels/`, `data/review_queue.json` |
| 5 export | 80/20 split, background images, `data.yaml` | `datasets/` |
| 6 train | fine-tunes YOLOv8n on the GPU | `runs/train-N/` |
| 7 predict | trained detector pre-labels the rest | `data/labels/`, `data/review_queue.json` |

`src/tools/` holds things that inspect rather than produce: `audit_labels`,
`preview_labels`, `find_class_examples`, `remap_classes`, `calibrate_thresholds`,
`gpu_check`. **None of them writes to `data/labels/`** — proposing is not
labelling, and a tool that quietly relabelled would be worse than no tool.

Step 4 is the cold start; step 7 replaces it once a model exists. Both feed the
same review queue, and the review screen does not know which produced a box.

## What this project learned the hard way

- **Read the review queue least-confident first.** A frame scored 0.95 teaches
  nobody anything. Correcting the hard ones is what makes the next model better.
- **Deduplication is why a rare class stays rare.** Searched against `data/raw`,
  every candidate for the scarce classes that had survived deduplication was
  already labelled; every genuinely new one had been discarded as a near
  duplicate. Deduplication exists to stop the same thing being labelled twice,
  and on a rare class that is exactly the wrong instinct. Recovering one is a
  copy back into `data/deduplicated/`.
- **A class that gets worse as it gains data is dirty, not starved.**
  `duzensiz_istif` fell 0.557 → 0.495 while growing 27 → 31 boxes; a fifth of its
  labels were wrong. Correcting six boxes took it to 0.740 without adding a
  single new example. Compare `palet_2li`, which was genuinely starved: 26 → 62
  training images took it 0.495 → 0.951. The two need opposite treatments, and
  the direction of the score as data grows is what tells them apart.
- **One wrong label damages two classes.** A neatly stacked single pallet
  labelled `duzensiz_istif` was teaching the model both that irregular stacks
  look neat and that `palet_1li` does not exist there. Moving that one box took
  `palet_1li` from 0.566 to 0.857 on unchanged training data. Look for a
  mislabel's beneficiary, not just its victim.
- **`audit_labels` is blind to consistent mistakes.** The model was trained on
  these labels, so a mistake made the same way every time is one the model has
  learned and will agree with. A clean audit means "no new contradictions",
  never "the labels are right".
- **Trust the audit in proportion to what the class taught the model.** It was
  mostly right about `duzensiz_istif` at 31 boxes and wrong three times in four
  about `palet_1li` at 11 — with eight training images it cannot recognise a
  neat single pallet, so it called them irregular. Both classes produced
  disagreements that all pointed one way, and that pattern does not separate
  dirty labels from an under-taught class. Only opening the frames does.
- **Dispute rate tracks class size, not class quality.** Across 766 boxes:
  `palet_3lu` 0.5%, `koli` 2%, `palet_2li` 2.6%, `duzensiz_istif` 7%,
  `palet_1li` 36%. Read that as a map of what the model knows, not of where the
  errors are.
- **Classes may sit on different axes and that is fine.** `palet_1li/2li/3lu`
  count rows; `duzensiz_istif` is what an irregular or incomplete stack is called
  when the rows cannot be counted. A different axis is not a broken one.
- **Confirmed-empty frames are worth labelling.** They export as background
  images (~10% of the training set) and cost about half a second each to confirm.
  They raise precision at a small cost in recall, which is the right trade here.
- **Subprocess output must be read as UTF-8.** Steps run as subprocesses piped
  into the GUI console. Read with the locale code page, a Turkish Windows decodes
  cp1254 and Ultralytics' progress bars kill the run — but only from the GUI,
  never from a terminal, where there is no pipe.
- **Measure pace with the median, not the mean.** Review sessions get
  interrupted; one coffee break should not move the number.

## Testing

`tests/` mirrors the modules. Qt tests use `QT_QPA_PLATFORM=offscreen` via an
autouse fixture in `conftest.py`; heavy models are skipped with `importorskip`.
Test names are sentences describing the behaviour, and each docstring says why
the case is worth testing rather than restating the assertion.

Two bugs have escaped through the test suite by being written on Linux for a
Windows machine: a path comparison of `\` against `/`, and the cp1254 crash
above. Prefer `pathlib` comparisons over string ones, and be suspicious of
anything involving encodings or path separators.

## Current state

834 frames labelled (104 confirmed empty), 766 boxes, six classes. Six training
rounds; mAP50-95 0.685 → 0.856, review load 68% of frames → 7.4%. Roughly 2000 of
5240 raw images are not yet ingested, but deduplication keeps only about a
quarter of new frames, so the pool is saturating.

Every class now clears 0.74 mAP50-95. The weakest is `duzensiz_istif` at 0.740
with precision 0.484 — it over-fires, and it holds 25 boxes after cleaning.

**Read the per-class table, not the headline.** Several classes are measured on
three to nine validation instances, so their numbers move on one box. The
direction across rounds is evidence; the third decimal place is not.

Next: `audit_labels` with no `--class` filter, judged by `runs/train-8`. The
cleaner model can see contradictions the dirty one agreed with.

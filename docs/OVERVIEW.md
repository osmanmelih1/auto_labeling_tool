# Sophtrun auto-labeling tool — what it produced, and what to distrust

A desktop tool that labels an image dataset for YOLO object detection with as
little human attention as possible, then trains on what it produced. One
operator, one machine, an RTX 3060 with 6 GB.

This page is the result, not the architecture. The pipeline is documented in
`CLAUDE.md`; the point here is what the numbers are and how much weight they can
carry.

## The headline

| | |
|---|---|
| Raw frames ingested | 5242 |
| Frames kept after deduplication | 1159 (22%) |
| Frames labelled | 1159 — all of them |
| Boxes | 1046, plus 162 frames confirmed to hold nothing |
| Classes | 6 |
| Training rounds | 9 |
| Best mAP50-95 | **0.858** (round 8), 0.807 on the current, larger validation set |
| Human review load | 68% of frames → **3.5%** |

That last row is what the tool is for. The first pass needed a decision on two
frames in three. The current pass, with a trained detector doing the
pre-labelling, needs one in twenty-eight: the last run auto-accepted 250 frames,
confirmed 57 as empty, and sent 11 to a human.

## Round by round

Each round is: export an 80/20 split, fine-tune YOLOv8n for 100 epochs, let the
new detector pre-label the unlabelled frames, correct only what it was unsure
about, repeat.

| Round | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| 5 | 0.776 | 0.684 | 0.731 | 0.672 |
| 7 | 0.862 | 0.780 | 0.912 | 0.853 |
| 8 | 0.932 | **0.858** | 0.831 | 0.938 |
| 9 | 0.887 | 0.807 | 0.911 | 0.782 |

**Round 9 is not worse than round 8.** The validation set is regenerated on every
export, so those two numbers are measured on different frames — round 9's
validation set is 36% larger and shares few images with round 8's. The one
comparison that survives is per class on a bigger test set: `palet_3lu` went from
0.94 to 0.957 while its validation instances grew 42%.

## Per class, current round

222 validation images, 212 instances.

| Class | mAP50-95 | Val instances | Training boxes | Share of dataset |
|---|---|---|---|---|
| palet_3lu | 0.957 | 168 | 826 | 79.0% |
| koli | 0.941 | 11 | 57 | 5.4% |
| palet_2li | 0.906 | 18 | 89 | 8.5% |
| elde_tasinan | 0.704 | 3 | 16 | 1.5% |
| duzensiz_istif | 0.686 | 9 | 44 | 4.2% |
| palet_1li | 0.645 | 3 | 14 | 1.3% |

**Read this table, not the headline.** Four of six classes are measured on three
to eighteen instances. At three instances a single box moves the score by a third.
The direction across rounds is evidence; the third decimal place is not.

## Known weaknesses

**A 59:1 class imbalance.** `palet_3lu` holds 826 boxes and `palet_1li` holds 14.
Every score below 0.75 in the table above belongs to a class with fewer than 50
examples. This is the dominant limit on the model and it is not fixable by more
labelling of the same footage: the frames simply do not contain single pallets in
any quantity. It needs either targeted collection or training-side oversampling.

**Deduplication works against the rare classes.** Searching the raw set for more
`palet_1li` found that every surviving candidate was already labelled and every
genuinely new one had been discarded as a near-duplicate. Deduplication exists so
the same object is not labelled twice, which on a scarce class is exactly the
wrong instinct. Recovering one is a copy back into `data/deduplicated/`.

**The audit cannot see a consistent mistake.** `src/tools/audit_labels` flags
boxes the trained detector disagrees with. But the detector was trained on these
labels, so an error made the same way every time is one it has learned and will
agree with. A clean audit means "no new contradictions", never "the labels are
right".

**The raw pool is saturating, not exhausted.** Around 2000 of the 5242 raw images
have not been ingested, but deduplication keeps only about a quarter of new
frames, so the reachable ceiling is a few hundred more frames rather than a few
thousand.

## Two findings worth more than the scores

**A class that gets worse as it gains data is dirty, not starved — and the two
need opposite treatments.** `duzensiz_istif` fell 0.557 → 0.495 while growing from
27 to 31 boxes. A fifth of its labels were wrong. Correcting six boxes took it to
0.740 without adding a single new example. Compare `palet_2li`, which was
genuinely starved: 26 → 62 training images took it 0.495 → 0.951. The direction of
the score as data grows is what separates the two cases.

**One wrong label damages two classes.** A neatly stacked single pallet labelled
`duzensiz_istif` was teaching the model both that irregular stacks look neat and
that `palet_1li` does not appear in that situation. Moving that one box took
`palet_1li` from 0.566 to 0.857 on otherwise unchanged training data. When
hunting a mislabel, look for its beneficiary and not only its victim.

## What it costs to run

- Deduplication and embedding: minutes, once per batch of new footage.
- Pre-labelling 318 frames: minutes on the GPU.
- Human review at the current rate: 11 frames.
- Training 100 epochs on 885 images: roughly 50 minutes on the 3060.

A confirmed-empty frame costs about half a second and is worth labelling: 162 of
them export as background images, which raises precision at a small cost in
recall.

## Reproducing any of it

```
uv run main.py                                   # the GUI, all 8 steps and the tools
uv run pytest                                    # 206 tests
uv run python -m src.core.step6_train            # or any step, by module path
uv run python -m src.tools.audit_labels          # or any tool, likewise
```

Nothing in the source is tied to this dataset: classes live in
`data/classes.json` and are edited from the GUI, colours are generated for any
number of them, every path resolves under `data/`. Pointing the tool at another
project is `uv run python -m src.tools.new_project --apply`, which clears the
data and keeps the ~700 MB of model weights.

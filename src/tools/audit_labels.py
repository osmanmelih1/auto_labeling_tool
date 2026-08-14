"""Find labels the trained detector confidently disagrees with.

A class that gets *worse* as it gains examples is not starved, it is dirty. The
new examples are teaching the model something the old ones contradict, and no
amount of further labelling fixes that — it makes it worse.

Finding the contradictions by eye means opening every frame of the class and
remembering what the class is supposed to mean. This does the first pass
instead. The detector has already seen the whole dataset; where it looks at a
box and confidently names a different class than the label does, one of the two
is wrong, and it is worth a human deciding which.

That is not a claim the model is right. On a class with thirty examples the
model is often the one that is wrong, and its disagreement then says the class
is genuinely hard rather than mislabelled. Either answer is worth knowing, and
both are cheaper to reach from a shortlist of ten frames than from all of them.

A second kind of suspect is reported too: a labelled box the detector sees
nothing at all in. That is what a junk box looks like from the model's side.

**Nothing is written to ``data/labels/``.** The output is a shortlist and a
folder of previews; the correction is a human decision made in the review
editor.

Input:  ``runs/*/weights/best.pt``, ``data/labels/``, ``data/deduplicated/``
Output: ``data/audit/`` (previews only)
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from src.core.class_config import load_classes
from src.core.yolo_format import iou, read_yolo_boxes, yolo_box_to_pixels

IMAGE_DIR = "data/deduplicated"
LABEL_DIR = "data/labels"
OUTPUT_DIR = "data/audit"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# A prediction has to land on the same object before its class is worth
# comparing. Below this the model is talking about something else in the frame.
MATCH_IOU = 0.5

# The detector is asked for everything it can see, however faint, so that an
# unmatched label means "nothing here at all" rather than "nothing above a
# threshold I happened to pick".
SEARCH_CONFIDENCE = 0.05

# Only a confident disagreement is worth a human's attention. A model that is
# unsure is not evidence of anything.
DISAGREE_CONFIDENCE = 0.50


def predict_boxes(model, image_path: Path) -> list[tuple[int, float, tuple]]:
    """Run the detector over one image.

    Args:
        model: A loaded Ultralytics model.
        image_path: Image to predict on.

    Returns:
        list: ``(class_id, confidence, (x_center, y_center, width, height))``,
        normalised to [0, 1].
    """
    result = model.predict(str(image_path), conf=SEARCH_CONFIDENCE, verbose=False)[0]
    return [(int(b.cls.item()), float(b.conf.item()), tuple(b.xywhn[0].tolist())) for b in result.boxes]


def judge(
    label: tuple[int, tuple],
    predictions: list[tuple[int, float, tuple]],
    match_iou: float,
    disagree_confidence: float,
) -> tuple[str, int | None, float]:
    """Compare one labelled box against everything the detector saw.

    Args:
        label: ``(class_id, box)`` for the labelled object.
        predictions: The detector's output for the same frame.
        match_iou: Overlap at which a prediction is about the same object.
        disagree_confidence: Confidence below which a disagreement is ignored.

    Returns:
        tuple: A verdict of ``"agrees"``, ``"disagrees"`` or ``"unseen"``, the
        class the model preferred (None when it saw nothing), and its confidence.
    """
    class_id, box = label

    overlapping = [(c, conf, p) for c, conf, p in predictions if iou(box, p) >= match_iou]
    if not overlapping:
        return "unseen", None, 0.0

    best_class, best_confidence, _ = max(overlapping, key=lambda item: item[1])

    if best_class == class_id:
        return "agrees", best_class, best_confidence
    if best_confidence < disagree_confidence:
        return "agrees", best_class, best_confidence

    return "disagrees", best_class, best_confidence


def save_preview(
    image_path: Path,
    label: tuple[int, tuple],
    verdict: str,
    proposed: int | None,
    confidence: float,
    output_dir: Path,
    names: list[str],
) -> None:
    """Draw the disputed box with both opinions on it.

    Args:
        image_path: Source image.
        label: ``(class_id, box)`` for the labelled object.
        verdict: The verdict from :func:`judge`.
        proposed: The class the model preferred, if any.
        confidence: The model's confidence in that class.
        output_dir: Directory receiving the preview.
        names: Class names in id order.
    """
    class_id, box = label

    try:
        image = Image.open(image_path).convert("RGB")
    except OSError as e:
        print(f"[-] Could not open {image_path.name}: {e}")
        return

    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = yolo_box_to_pixels(box, image.width, image.height)

    draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)
    label_text = f"label: {names[class_id] if class_id < len(names) else class_id}"
    if proposed is not None:
        label_text += f"   model: {names[proposed] if proposed < len(names) else proposed} {confidence:.2f}"
    else:
        label_text += "   model: nothing here"
    draw.text((x1 + 4, max(0, y1 - 14)), label_text, fill=(255, 80, 80))

    stem = f"{verdict}_{confidence:.2f}_{image_path.stem}.jpg"
    image.save(output_dir / stem, quality=88)


def main(argv: list[str] | None = None) -> int:
    """Shortlist the labels a trained detector contradicts.

    Args:
        argv: Command line arguments. ``sys.argv[1:]`` when omitted.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--class", dest="class_name", help="audit only labels of this class")
    parser.add_argument("--weights", help="checkpoint to judge with; the latest run when omitted")
    parser.add_argument("--images", default=IMAGE_DIR, help="directory of source images")
    parser.add_argument("--labels", default=LABEL_DIR, help="directory of label files")
    parser.add_argument("--out", default=OUTPUT_DIR, help="directory to write previews into")
    parser.add_argument("--iou", type=float, default=MATCH_IOU, help="overlap counting as the same object")
    parser.add_argument(
        "--confidence",
        type=float,
        default=DISAGREE_CONFIDENCE,
        help="confidence below which a disagreement is ignored",
    )
    args = parser.parse_args(argv)

    names = load_classes()

    wanted_class = None
    if args.class_name:
        if args.class_name not in names:
            print(f"[!] No class named '{args.class_name}'. Known: {', '.join(names)}")
            return 1
        wanted_class = names.index(args.class_name)

    # Imported here rather than at the top: the prediction step pulls in
    # ultralytics and torch, and the verdict logic below is arithmetic that
    # should stay testable without either.
    from src.core.step7_predict import find_latest_weights

    weights = args.weights or find_latest_weights()
    if not weights or not Path(weights).exists():
        print("[!] No trained detector found. Train one before auditing labels.")
        return 1

    from ultralytics import YOLO

    print(f"[*] Judging with {weights}...")
    model = YOLO(weights)

    images = {p.stem: p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.jpg"):
        stale.unlink()

    suspects: list[tuple[str, str, str, int, int | None, float]] = []
    examined = frames = 0

    for label_path in sorted(Path(args.labels).glob("*.txt")):
        boxes = read_yolo_boxes(str(label_path))
        of_interest = [(class_id, tuple(box)) for class_id, *box in boxes if wanted_class in (None, class_id)]
        if not of_interest:
            continue

        image_path = images.get(label_path.stem)
        if image_path is None:
            continue

        frames += 1
        predictions = predict_boxes(model, image_path)

        for class_id, box in of_interest:
            examined += 1
            verdict, proposed, confidence = judge((class_id, box), predictions, args.iou, args.confidence)
            if verdict == "agrees":
                continue

            suspects.append((verdict, label_path.stem, image_path.name, class_id, proposed, confidence))
            save_preview(image_path, (class_id, box), verdict, proposed, confidence, output_dir, names)

    scope = f"'{args.class_name}'" if args.class_name else "every class"
    print(f"[+] Examined {examined} box(es) of {scope} across {frames} frame(s).\n")

    if not suspects:
        print("[+] The detector agrees with every label it could see. Nothing to review.")
        return 0

    # Loudest disagreements first: the model's confidence is the closest thing
    # available to a ranking of how likely the label is to be wrong.
    suspects.sort(key=lambda s: (s[0] != "disagrees", -s[5]))

    print(f"    {'verdict':<10} {'labelled':<16} {'model says':<16} {'conf':>5}  frame")
    for verdict, key, _, class_id, proposed, confidence in suspects:
        says = "nothing here" if proposed is None else names[proposed]
        print(f"    {verdict:<10} {names[class_id]:<16} {says:<16} {confidence:>5.2f}  {key}")

    disagreements = sum(1 for s in suspects if s[0] == "disagrees")
    print(f"\n[+] {disagreements} confident disagreement(s), {len(suspects) - disagreements} unseen box(es).")
    print(f"[+] Previews in {output_dir}. Open the frame in Edit Labels to correct one.")
    print("[*] A disagreement is a question, not a verdict: on a small class the model is often")
    print("    the one that is wrong, and that is worth knowing too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

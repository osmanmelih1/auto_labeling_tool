"""YOLO label format helpers.

This is a utility, not a pipeline step, and deliberately depends on nothing
heavier than the standard library. Reading a label file and converting a box
between normalised and pixel coordinates is arithmetic; putting it here means the
GUI can do it without importing torch, and means the parsing rules are written
once rather than in each step that happens to need them.

The format is the YOLO convention: one object per line, as
``class_id x_center y_center width height``, with the four coordinates
normalised to [0, 1] against the image size.
"""


def read_yolo_boxes(label_path: str) -> list[tuple[int, float, float, float, float]]:
    """Parse a YOLO label file into class ids and normalised boxes.

    Malformed lines are skipped rather than raising: one bad label should not
    stop a run across thousands of images, and the caller usually cannot do
    anything more useful with the exception than ignore that line anyway.

    Args:
        label_path: Path to a YOLO .txt label file.

    Returns:
        list: One ``(class_id, x_center, y_center, width, height)`` per valid line.
    """
    boxes: list[tuple[int, float, float, float, float]] = []

    try:
        with open(label_path) as f:
            lines = f.readlines()
    except OSError as e:
        print(f"[-] Could not read {label_path}: {e}")
        return boxes

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            xc, yc, w, h = (float(p) for p in parts[1:])
        except ValueError:
            continue
        boxes.append((class_id, xc, yc, w, h))

    return boxes


def write_yolo_boxes(label_path: str, boxes: list[tuple[int, float, float, float, float]]) -> None:
    """Write class ids and normalised boxes as a YOLO label file.

    Args:
        label_path: Destination .txt path.
        boxes: One ``(class_id, x_center, y_center, width, height)`` per object.
    """
    with open(label_path, "w") as f:
        for class_id, xc, yc, w, h in boxes:
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def yolo_box_to_pixels(
    box: tuple[float, float, float, float], img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Convert a normalised YOLO box into pixel corner coordinates.

    Args:
        box: ``(x_center, y_center, width, height)`` normalised to [0, 1].
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        tuple: ``(x_min, y_min, x_max, y_max)`` clamped to the image bounds.
    """
    xc, yc, w, h = box
    x_min = int(max(0, (xc - w / 2) * img_w))
    y_min = int(max(0, (yc - h / 2) * img_h))
    x_max = int(min(img_w, (xc + w / 2) * img_w))
    y_max = int(min(img_h, (yc + h / 2) * img_h))
    return x_min, y_min, x_max, y_max


def box_area(box: tuple[float, float, float, float]) -> float:
    """Return the normalised area of a YOLO box.

    Args:
        box: ``(x_center, y_center, width, height)`` normalised to [0, 1].

    Returns:
        float: Width multiplied by height.
    """
    return box[2] * box[3]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute intersection over union for two normalised YOLO boxes.

    Args:
        a: First box as ``(x_center, y_center, width, height)``.
        b: Second box in the same format.

    Returns:
        float: Overlap ratio between 0 and 1.
    """
    ax0, ay0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax1, ay1 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx1, by1 = b[0] + b[2] / 2, b[1] + b[3] / 2

    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h

    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0

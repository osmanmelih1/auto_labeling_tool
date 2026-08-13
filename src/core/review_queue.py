"""Shared review queue store.

This is a utility, not a pipeline step. Both the propagation step and the GUI
read and write ``data/review_queue.json``, so the file's structure is defined
once here instead of being re-implemented on each side.

The queue has two sections:

``pending``
    Borderline matches waiting for a human decision.

``rejected``
    Matches a human has turned down, together with the score they were rejected
    at. Propagation is run repeatedly as the seed pool grows, and without this
    record every rejected image would be proposed again on every single run. On
    a few thousand images that means rejecting the same wrong box over and over.

Rejection is not permanent. A rejected image is only suppressed while the new
score stays close to the one it was rejected at; if a later run scores it
clearly higher, the seed pool has learned something new and the image is worth
a second look. That threshold is ``REPROPOSE_MARGIN``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

REVIEW_QUEUE_PATH = "data/review_queue.json"

# One line per review sitting. What review costs per frame is the number that
# decides whether this pipeline needs a model in the loop, and the first time it
# was measured the answer was printed to a console panel that died with the
# window. A record that only exists while the application is open is not a
# measurement.
SESSION_LOG_PATH = "data/review_sessions.jsonl"

# A rejected image is proposed again only once its score exceeds the score it was
# rejected at by this much. Small enough that a genuinely improved match returns,
# large enough that run-to-run noise does not resurrect it.
REPROPOSE_MARGIN = 0.05


def load_queue(path: str = REVIEW_QUEUE_PATH) -> dict:
    """Read the review queue, tolerating a missing or corrupt file.

    Args:
        path: Location of the queue file.

    Returns:
        dict: The queue, always containing ``pending`` and ``rejected`` mappings.
    """
    queue: dict = {"pending": {}, "rejected": {}}

    file_path = Path(path)
    if not file_path.exists():
        return queue

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[!] {path} could not be read ({e}). Re-initialising it.")
        return queue

    if isinstance(data, dict):
        for section in ("pending", "rejected"):
            if isinstance(data.get(section), dict):
                queue[section] = data[section]

    return queue


def save_queue(queue: dict, path: str = REVIEW_QUEUE_PATH) -> None:
    """Write the review queue to disk.

    Args:
        queue: The queue structure to persist.
        path: Location of the queue file.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)


def is_suppressed(queue: dict, image_key: str, score: float) -> tuple[bool, float]:
    """Decide whether a previously rejected image should be skipped again.

    Args:
        queue: The review queue.
        image_key: Image being considered.
        score: Score the current run produced for it.

    Returns:
        tuple: Whether to skip it, and the score it was rejected at (0.0 when it
        was never rejected).
    """
    entry = queue.get("rejected", {}).get(image_key)
    if entry is None:
        return False, 0.0

    previous = float(entry.get("score", 0.0))
    return score <= previous + REPROPOSE_MARGIN, previous


def add_pending(queue: dict, image_key: str, entry: dict) -> None:
    """Queue a match for human review, clearing any earlier rejection.

    Args:
        queue: The review queue, mutated in place.
        image_key: Image being queued.
        entry: Record describing the match.
    """
    queue.setdefault("pending", {})[image_key] = entry
    queue.setdefault("rejected", {}).pop(image_key, None)


def accept(queue: dict, image_key: str) -> dict | None:
    """Remove an entry from the pending list, keeping its label on disk.

    Args:
        queue: The review queue, mutated in place.
        image_key: Image being accepted.

    Returns:
        dict | None: The entry that was accepted, or None if it was not pending.
    """
    return queue.setdefault("pending", {}).pop(image_key, None)


def reject(queue: dict, image_key: str) -> dict | None:
    """Move an entry from pending to rejected, recording when and at what score.

    Args:
        queue: The review queue, mutated in place.
        image_key: Image being rejected.

    Returns:
        dict | None: The entry that was rejected, or None if it was not pending.
    """
    entry = queue.setdefault("pending", {}).pop(image_key, None)
    if entry is None:
        return None

    queue.setdefault("rejected", {})[image_key] = {
        "score": entry.get("score", 0.0),
        "seed_source": entry.get("seed_source"),
        "class_id": entry.get("class_id"),
        "rejected_at": datetime.now(UTC).isoformat(),
    }
    return entry


def append_session_record(record: dict, path: str = SESSION_LOG_PATH) -> None:
    """Append one review sitting to the session log.

    Written as JSON lines so sittings accumulate across days without any
    read-modify-write, and so the file survives a crash mid-session with every
    earlier line intact.

    Args:
        record: Facts about the sitting: counts, pace and timestamps.
        path: Location of the session log.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_rejections(queue: dict) -> int:
    """Forget every rejection so those images can be proposed again.

    Useful after the seed pool has changed substantially, when past rejections
    say more about the old prototypes than about the images themselves.

    Args:
        queue: The review queue, mutated in place.

    Returns:
        int: How many rejections were cleared.
    """
    count = len(queue.get("rejected", {}))
    queue["rejected"] = {}
    return count

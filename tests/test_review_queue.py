"""Rejection memory, which decides what a human is asked twice.

Propagation runs repeatedly as the seed pool grows. Without a record of what was
already turned down, every rejected image returns on every run; with too strong
a record, an image the tool has genuinely learned to recognise never gets a
second chance. The margin between those two failures is what this module tests.
"""

from src.core.review_queue import (
    REPROPOSE_MARGIN,
    accept,
    add_pending,
    clear_rejections,
    is_suppressed,
    load_queue,
    reject,
    save_queue,
)


def entry(key: str, score: float) -> dict:
    """Build a minimal queue record.

    Args:
        key: Image key.
        score: Confidence propagation gave the frame.

    Returns:
        dict: A record with the fields the queue itself reads.
    """
    return {"image_key": key, "score": score, "seed_source": "seed_a", "class_id": 0}


def test_missing_queue_file_loads_as_empty(tmp_path):
    """A first run has no queue file, which is not an error."""
    queue = load_queue(str(tmp_path / "review_queue.json"))
    assert queue == {"pending": {}, "rejected": {}}


def test_corrupt_queue_file_is_reinitialised(tmp_path):
    """A truncated write must not stop the tool from starting."""
    path = tmp_path / "review_queue.json"
    path.write_text('{"pending": {"a":', encoding="utf-8")

    assert load_queue(str(path)) == {"pending": {}, "rejected": {}}


def test_round_trip_through_disk(tmp_path):
    """What is saved is what is loaded."""
    path = str(tmp_path / "review_queue.json")
    queue = {"pending": {}, "rejected": {}}
    add_pending(queue, "a", entry("a", 0.83))
    save_queue(queue, path)

    assert load_queue(path)["pending"]["a"]["score"] == 0.83


def test_accept_removes_the_entry_and_returns_it():
    """Accepting clears the queue entry; the label file stays on disk."""
    queue = {"pending": {}, "rejected": {}}
    add_pending(queue, "a", entry("a", 0.83))

    returned = accept(queue, "a")

    assert returned["image_key"] == "a"
    assert queue["pending"] == {}
    assert queue["rejected"] == {}


def test_accepting_an_unknown_key_is_harmless():
    """A double click on accept must not raise."""
    assert accept({"pending": {}, "rejected": {}}, "ghost") is None


def test_reject_records_the_score_it_was_rejected_at():
    """The score is the whole point of the record; without it nothing can change."""
    queue = {"pending": {}, "rejected": {}}
    add_pending(queue, "a", entry("a", 0.83))

    reject(queue, "a")

    assert queue["pending"] == {}
    assert queue["rejected"]["a"]["score"] == 0.83
    assert "rejected_at" in queue["rejected"]["a"]


def test_a_rejected_image_is_suppressed_at_a_similar_score():
    """Run-to-run noise must not resurrect a rejected frame."""
    queue = {"pending": {}, "rejected": {}}
    add_pending(queue, "a", entry("a", 0.83))
    reject(queue, "a")

    skip, previous = is_suppressed(queue, "a", 0.83 + REPROPOSE_MARGIN / 2)

    assert skip is True
    assert previous == 0.83


def test_a_rejected_image_returns_once_it_scores_clearly_higher():
    """A clearly better score means the seed pool learned something new."""
    queue = {"pending": {}, "rejected": {}}
    add_pending(queue, "a", entry("a", 0.83))
    reject(queue, "a")

    skip, _ = is_suppressed(queue, "a", 0.83 + REPROPOSE_MARGIN + 0.01)

    assert skip is False


def test_an_image_that_was_never_rejected_is_never_suppressed():
    """Suppression must not leak to frames nobody has judged."""
    assert is_suppressed({"pending": {}, "rejected": {}}, "fresh", 0.99) == (False, 0.0)


def test_requeuing_clears_an_earlier_rejection():
    """A frame cannot be pending and rejected at once."""
    queue = {"pending": {}, "rejected": {}}
    add_pending(queue, "a", entry("a", 0.83))
    reject(queue, "a")

    add_pending(queue, "a", entry("a", 0.91))

    assert "a" in queue["pending"]
    assert "a" not in queue["rejected"]


def test_clear_rejections_reports_how_many_it_forgot():
    """The GUI shows this count, so it has to be the real one."""
    queue = {"pending": {}, "rejected": {}}
    for key in ("a", "b", "c"):
        add_pending(queue, key, entry(key, 0.8))
        reject(queue, key)

    assert clear_rejections(queue) == 3
    assert queue["rejected"] == {}

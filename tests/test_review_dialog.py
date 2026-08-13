"""The Review Queue screen: the list, the editor pane and the decisions.

The bug this file exists to prevent has already happened once. An edit moved the
Accept and Reject buttons inside another method, past its return, so review
cards would have been built with no way to decide anything. Nothing about the
code looked wrong; only running it would have shown it.
"""

import json
import os

import pytest
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QMessageBox, QPushButton

from src.core.yolo_format import read_yolo_boxes, write_yolo_boxes
from src.gui.app import ReviewQueueDialog


@pytest.fixture
def queue(project_sandbox):
    """Write a three-entry queue with images, labels and masks on disk.

    The first frame deliberately holds two different classes, which is the case
    a single per-frame dropdown could never express.

    Args:
        project_sandbox: The sandboxed project root.

    Returns:
        tuple: The queue file path and the entries keyed by image key.
    """
    entries = {}
    for key, boxes, scores in (
        ("a", [(2, 0.3, 0.5, 0.2, 0.3), (4, 0.7, 0.5, 0.1, 0.2)], [0.89, 0.79]),
        ("b", [(0, 0.5, 0.5, 0.3, 0.3)], [0.84]),
        ("c", [(1, 0.5, 0.5, 0.3, 0.3)], [0.81]),
    ):
        image_path = project_sandbox / f"{key}.png"
        QImage(320, 240, QImage.Format.Format_RGB32).save(str(image_path))
        label_path = project_sandbox / f"{key}.txt"
        write_yolo_boxes(str(label_path), boxes)
        mask_path = project_sandbox / f"{key}_mask.png"
        QImage(320, 240, QImage.Format.Format_RGB32).save(str(mask_path))

        entries[key] = {
            "image_key": key,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "mask_path": str(mask_path),
            "score": scores[0],
            "weakest_score": min(scores),
            "box_scores": scores,
            "object_count": len(boxes),
            "seed_source": "seed_x",
            "class_id": boxes[0][0],
            "flagged_at": "2026-08-12T00:00:00+00:00",
            "status": "pending_review",
        }

    path = project_sandbox / "review_queue.json"
    path.write_text(json.dumps({"pending": entries, "rejected": {}}), encoding="utf-8")
    return str(path), entries


@pytest.fixture
def dialog(qapp, queue):
    """Open the review dialog on the prepared queue.

    Args:
        qapp: The shared QApplication.
        queue: The queue fixture.

    Returns:
        tuple: The dialog and the entries it was built from.
    """
    path, entries = queue
    return ReviewQueueDialog(parent=None, review_queue_path=path), entries


def test_every_card_carries_a_decision(dialog):
    """Cards without Accept and Reject buttons is a bug that has happened before."""
    dlg, _ = dialog
    assert len(dlg.cards) == 3

    for key, card in dlg.cards.items():
        labels = [b.text() for b in card.findChildren(QPushButton)]
        assert any("Accept" in text for text in labels), key
        assert any("Reject" in text for text in labels), key


def test_a_card_summarises_the_label_file_per_class(dialog):
    """The list has to be scannable without opening each frame."""
    dlg, _ = dialog
    text = dlg.cards["a"].summary_label.text()

    assert "pallet_3" in text
    assert "irregular" in text


def test_a_mixed_frame_opens_with_both_classes_intact(dialog):
    """The old per-frame dropdown rewrote every box; the editor must not."""
    dlg, entries = dialog

    dlg._show_preview(entries["a"])

    assert [box["class_id"] for box in dlg.editor.boxes] == [2, 4]


def test_per_box_scores_reach_the_editor(dialog):
    """Dashed boxes are how a reviewer finds the uncertain part of a frame."""
    dlg, entries = dialog

    dlg._show_preview(entries["a"])

    assert [box["score"] for box in dlg.editor.boxes] == [0.89, 0.79]


def test_the_toolbar_is_inert_until_a_box_is_selected(dialog):
    """A class dropdown pointing at nothing would be a way to corrupt a label."""
    dlg, entries = dialog
    dlg._show_preview(entries["a"])

    assert dlg.editor_class_combo.isEnabled() is False
    assert dlg.delete_box_btn.isEnabled() is False

    dlg.editor.select(1)

    assert dlg.editor_class_combo.isEnabled() is True
    assert dlg.delete_box_btn.isEnabled() is True
    assert dlg.editor_class_combo.currentIndex() == 4


def test_the_dropdown_rewrites_one_box_and_not_its_neighbour(dialog):
    """This is the whole reason the per-frame control was removed."""
    dlg, entries = dialog
    dlg._show_preview(entries["a"])
    dlg.editor.select(1)

    dlg.editor_class_combo.setCurrentIndex(3)

    assert [c for c, *_ in read_yolo_boxes(entries["a"]["label_path"])] == [2, 3]


def test_the_card_summary_follows_an_edit(dialog):
    """The list and the file must not disagree about what was labelled."""
    dlg, entries = dialog
    dlg._show_preview(entries["a"])
    dlg.editor.select(1)

    dlg.editor_class_combo.setCurrentIndex(3)

    assert "carton" in dlg.cards["a"].summary_label.text()


def test_deleting_the_last_box_warns_rather_than_looking_empty(dialog):
    """A frame with no boxes is a frame that should probably be rejected."""
    dlg, entries = dialog
    dlg._show_preview(entries["b"])
    dlg.editor.select(0)

    dlg.delete_box_btn.click()

    assert read_yolo_boxes(entries["b"]["label_path"]) == []
    assert "confirm empty" in dlg.cards["b"].summary_label.text().lower()


def test_accepting_keeps_the_label_and_advances(dialog):
    """After a decision the next frame should already be open."""
    dlg, _ = dialog
    dlg._populate_cards()
    first = dlg._order[0]
    dlg._show_preview(dlg.queue_data["pending"][first])

    dlg.accept_next_btn.click()

    assert first not in dlg.queue_data["pending"]
    assert dlg._selected_key == dlg._order[1]
    assert dlg.editor.boxes


def test_rejecting_deletes_the_label_and_the_mask(dialog):
    """A label left behind returns as a seed; a mask left behind is an orphan."""
    dlg, entries = dialog
    dlg._populate_cards()
    key = dlg._order[0]
    dlg._show_preview(dlg.queue_data["pending"][key])

    dlg.reject_next_btn.click()

    assert key in dlg.queue_data["rejected"]
    assert not os.path.exists(entries[key]["label_path"])
    assert not os.path.exists(entries[key]["mask_path"])


def test_emptying_the_queue_clears_the_pane(dialog):
    """The last decision must not leave a stale frame on screen."""
    dlg, _ = dialog
    dlg._show_preview(dlg.queue_data["pending"][dlg._order[0]])
    for _ in range(3):
        dlg.accept_next_btn.click()

    assert dlg.queue_data["pending"] == {}
    assert dlg._selected_key is None
    assert dlg.accept_next_btn.isEnabled() is False


def test_pace_is_not_reported_before_it_means_anything(dialog):
    """Two or three decisions cannot support a median; saying so is better than lying."""
    dlg, _ = dialog
    dlg.decision_times = [0.0, 3.0, 6.0]

    assert dlg._pace_summary() is None


def test_pace_reports_the_median_not_the_mean(dialog):
    """One long interruption must not become the reported cost of a frame.

    Five decisions four seconds apart with a five-minute break in the middle: the
    mean is over a minute, the median is four seconds, and four seconds is what
    reviewing actually costs.
    """
    dlg, _ = dialog
    dlg.decision_times = [0.0, 4.0, 8.0, 308.0, 312.0, 316.0]

    summary = dlg._pace_summary()

    assert "Median 4.0 s per frame" in summary
    assert "5 decision(s)" in summary


def test_pace_projects_the_rest_of_the_queue(dialog):
    """The projection is the number that decides whether the loop needs a model in it."""
    dlg, _ = dialog
    dlg.decision_times = [0.0, 6.0, 12.0, 18.0, 24.0]

    # Three frames are still pending in the fixture.
    assert "remaining 3" in dlg._pace_summary()


def test_pace_says_nothing_about_a_queue_that_is_finished(dialog):
    """There is nothing left to project once the queue is empty."""
    dlg, _ = dialog
    dlg.queue_data["pending"] = {}
    dlg.decision_times = [0.0, 5.0, 10.0, 15.0, 20.0]

    assert "remaining" not in dlg._pace_summary()


def test_a_decision_is_timed(dialog):
    """The measurement has to come from the decisions, not from a stopwatch."""
    dlg, _ = dialog
    dlg._show_preview(dlg.queue_data["pending"][dlg._order[0]])

    dlg.accept_next_btn.click()
    dlg.reject_next_btn.click()

    assert len(dlg.decision_times) == 2


def test_closing_writes_the_session_to_disk(dialog, project_sandbox):
    """The console panel dies with the window; the measurement must not.

    Args:
        dialog: The review dialog and its entries.
        project_sandbox: The sandboxed project root.
    """
    dlg, _ = dialog
    dlg._show_preview(dlg.queue_data["pending"][dlg._order[0]])
    dlg.accept_next_btn.click()
    dlg.reject_next_btn.click()

    dlg.close()

    lines = (project_sandbox / "data" / "review_sessions.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["accepted"] == 1
    assert record["rejected"] == 1
    assert record["still_pending"] == 1


def test_sessions_accumulate_rather_than_overwrite(dialog, project_sandbox):
    """Two sittings in one day are two records, not one.

    Args:
        dialog: The review dialog and its entries.
        project_sandbox: The sandboxed project root.
    """
    dlg, _ = dialog
    dlg._show_preview(dlg.queue_data["pending"][dlg._order[0]])
    dlg.accept_next_btn.click()
    dlg.close()
    dlg.session_accepted = 0
    dlg.accept_next_btn.click()
    dlg.close()

    lines = (project_sandbox / "data" / "review_sessions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_a_sitting_with_no_decisions_writes_nothing(dialog, project_sandbox):
    """Opening the screen and closing it again is not a review session.

    Args:
        dialog: The review dialog and its entries.
        project_sandbox: The sandboxed project root.
    """
    dlg, _ = dialog

    dlg.close()

    assert not (project_sandbox / "data" / "review_sessions.jsonl").exists()


def test_bulk_reject_removes_every_label(dialog, monkeypatch):
    """Bulk actions take the same path as single ones, including the masks.

    Args:
        dialog: The review dialog and its entries.
        monkeypatch: Used to answer the confirmation box automatically.
    """
    dlg, entries = dialog
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    dlg._bulk_reject()

    assert dlg.queue_data["pending"] == {}
    for entry in entries.values():
        assert not os.path.exists(entry["label_path"])
        assert not os.path.exists(entry["mask_path"])

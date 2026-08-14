"""Shortlisting the labels a trained detector contradicts.

The judgement is the whole tool, and it is the part that can be wrong quietly:
a verdict that is too eager sends a human to look at correct labels, and one
that is too shy leaves the dirty class dirty. So the verdicts are tested
directly, without a model.
"""

import json

import pytest

from src.tools.audit_labels import judge, main

BOX = (0.5, 0.5, 0.2, 0.2)


def test_a_prediction_of_the_same_class_is_agreement():
    """The common case, and the one that must stay silent."""
    verdict, proposed, confidence = judge((2, BOX), [(2, 0.9, BOX)], 0.5, 0.5)

    assert verdict == "agrees"
    assert proposed == 2


def test_a_confident_prediction_of_another_class_is_a_disagreement():
    """This is the signal: the model looked at the object and named it differently."""
    verdict, proposed, confidence = judge((2, BOX), [(4, 0.88, BOX)], 0.5, 0.5)

    assert verdict == "disagrees"
    assert proposed == 4
    assert confidence == pytest.approx(0.88)


def test_an_unsure_prediction_of_another_class_is_not_a_disagreement():
    """A model that is guessing is not evidence, and sending a human to look is a cost."""
    verdict, _, _ = judge((2, BOX), [(4, 0.31, BOX)], 0.5, 0.5)

    assert verdict == "agrees"


def test_a_prediction_somewhere_else_in_the_frame_is_ignored():
    """Two objects in one frame must not be compared to each other."""
    elsewhere = (0.1, 0.1, 0.1, 0.1)

    verdict, proposed, _ = judge((2, BOX), [(4, 0.99, elsewhere)], 0.5, 0.5)

    assert verdict == "unseen"
    assert proposed is None


def test_a_box_the_detector_sees_nothing_in_is_reported():
    """A junk box looks like this from the model's side: an object nobody else finds."""
    verdict, proposed, confidence = judge((2, BOX), [], 0.5, 0.5)

    assert verdict == "unseen"
    assert proposed is None
    assert confidence == 0.0


def test_the_loudest_of_several_overlapping_predictions_wins():
    """Overlapping detections are common; the confident one is the one that matters.

    A quiet prediction of the labelled class must not silence a loud
    disagreement, or the tool reports nothing on exactly the frames worth
    seeing.
    """
    verdict, proposed, _ = judge((2, BOX), [(2, 0.20, BOX), (4, 0.95, BOX)], 0.5, 0.5)

    assert verdict == "disagrees"
    assert proposed == 4


def test_an_unknown_class_name_is_refused_rather_than_audited_as_nothing(project_sandbox, capsys):
    """A typo would otherwise audit every class and look like a clean result.

    Args:
        project_sandbox: The sandboxed project root.
        capsys: Captured output.
    """
    (project_sandbox / "data" / "classes.json").write_text(
        json.dumps({"classes": [{"name": "pallet_1", "description": "one row"}]}),
        encoding="utf-8",
    )

    assert main(["--class", "no_such_class"]) == 1
    assert "No class named" in capsys.readouterr().out


def test_a_missing_detector_is_reported_rather_than_crashing(project_sandbox, capsys):
    """The audit depends on a trained model; saying so beats a traceback.

    Args:
        project_sandbox: The sandboxed project root.
        capsys: Captured output.
    """
    assert main([]) == 1
    assert "No trained detector" in capsys.readouterr().out

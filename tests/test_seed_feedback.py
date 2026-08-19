"""Telling the operator what the seeding steps actually produced.

Steps 3a and 3b write a label and print two confidences, and neither confidence
answers the question being asked. Grounding DINO's score says the phrase matches
somewhere in the image; SAM's says it is sure of the mask it drew. A mask around
the floor the pallet stands on scores 0.98 on both.

These seeds become prototypes in Step 4 and propagate to every similar frame, so
a mistake here is the most expensive one in the pipeline. What is tested is the
arithmetic behind the two checks that a confidence cannot make — how much of the
frame the box covers, and how far it sits from the box that was asked for.
"""

import pytest

# Both steps import torch and transformers at module level, which is what the
# GUI is careful never to do. Skipping rather than failing keeps the suite
# runnable on a machine without the model stack installed.
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("cv2")

from src.core.step3a_text_prompting import (  # noqa: E402
    FRAME_SHARE_WARNING,
    MARGINAL_SCORE_MARGIN,
    normalise_prompt,
)
from src.core.step3b_manual_seeding import BOX_AGREEMENT_WARNING  # noqa: E402
from src.core.yolo_format import box_area, iou  # noqa: E402


def test_a_box_around_the_whole_scene_is_over_the_warning_line():
    """The characteristic zero-shot failure: a box on the floor, not the object.

    Grounding DINO reports a high score for it because the phrase does match the
    image somewhere, so nothing else in the step catches it.
    """
    whole_frame = (0.5, 0.5, 0.95, 0.9)

    assert box_area(whole_frame) > FRAME_SHARE_WARNING


def test_a_plausible_pallet_box_stays_under_it():
    """The check has to be quiet on the ordinary case or it will be ignored."""
    pallet = (0.4, 0.6, 0.3, 0.35)

    assert box_area(pallet) < FRAME_SHARE_WARNING


def test_refining_a_rough_box_is_not_treated_as_disagreement():
    """SAM is meant to move the edges: tightening is the point of the step.

    A rough rectangle around a pallet legitimately loses its padding, and warning
    about that would train the operator to ignore the warning.
    """
    drawn = (0.5, 0.5, 0.40, 0.40)
    tightened = (0.5, 0.5, 0.32, 0.34)

    assert iou(drawn, tightened) > BOX_AGREEMENT_WARNING


def test_a_mask_of_something_else_falls_below_the_line():
    """This is the case worth a warning: SAM segmented a different object.

    It reports high confidence either way, because it is confident about the mask
    and not about whether the mask is what was wanted.
    """
    drawn = (0.30, 0.30, 0.20, 0.20)
    elsewhere = (0.75, 0.80, 0.25, 0.20)

    assert iou(drawn, elsewhere) < BOX_AGREEMENT_WARNING


def test_the_observed_marginal_detection_would_now_be_flagged():
    """The run that prompted this: 0.326 against a 0.30 threshold, reported as [+].

    "package" was not really in that frame. Grounding DINO matched the nearest
    thing to it, SAM returned a mask at 0.948, and the size check passed at 5.7%
    of the frame — every number on screen looked healthy. What separates that run
    from a real detection is how little room it had above the threshold.
    """
    observed = 0.326
    flagged_below = 0.3 + MARGINAL_SCORE_MARGIN

    assert observed < flagged_below


def test_a_detection_with_room_above_the_threshold_is_not_flagged():
    """Warning on every run would make the warning worth nothing."""
    confident = 0.72
    flagged_below = 0.3 + MARGINAL_SCORE_MARGIN

    assert confident >= flagged_below


def test_the_thresholds_sit_where_they_can_still_be_read():
    """All three are judgements, not measurements, so they are asserted deliberately.

    If one moves, it should be because someone decided to move it.
    """
    assert FRAME_SHARE_WARNING == 0.5
    assert BOX_AGREEMENT_WARNING == 0.35
    assert MARGINAL_SCORE_MARGIN == 0.1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Pallet", "pallet."), ("  BOX  ", "box."), ("pallet.", "pallet."), ("", "")],
)
def test_the_prompt_is_normalised_before_the_model_sees_it(raw, expected):
    """Grounding DINO measurably degrades on anything but lowercase-with-period.

    Args:
        raw: Prompt as the operator might type it.
        expected: The form the model expects.
    """
    assert normalise_prompt(raw) == expected

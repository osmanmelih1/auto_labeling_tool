"""Confidence tier configuration for propagation.

These live in their own module, with no dependencies, because both the
propagation step and the GUI need them. Reading them from the step would drag
torch and SAM into the GUI process, which never runs a model of its own: it
launches the steps as subprocesses. Two floats are not worth five seconds of
start-up and half a gigabyte of memory.

Keeping them in one place also means the numbers shown to the user cannot drift
from the numbers actually applied.

Do not tune these by eye. ``uv run python -m src.tools.calibrate_thresholds``
measures the score distribution and reports what each candidate value would
admit and cost.
"""

# Calibrated against the first validation run, where images that genuinely
# contained the seeded object scored 0.82-0.91 and images that did not scored
# 0.69-0.76. They sit deliberately on the cautious side of that gap: with only a
# few seeds most true matches land in the review queue rather than being accepted
# outright.
AUTO_ACCEPT_THRESHOLD = 0.86
REVIEW_THRESHOLD = 0.78

# A heatmap cell belongs to an object region when its similarity reaches this
# absolute level. It is tied to the review threshold on purpose: a region is worth
# proposing exactly when its best patch would pass human review. A level relative
# to the frame's own peak would instead define "hot" in terms of the single
# strongest match, hiding every dimmer instance of the same object.
DETECTION_LEVEL = REVIEW_THRESHOLD

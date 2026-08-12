"""Measure the propagation score distribution and propose confidence thresholds.

The thresholds in Step 4 were fitted on seven images. Before running the full
dataset they need checking against real numbers, because accepted labels become
seeds for the next round: a threshold that is too loose does not produce a few
bad labels, it produces bad prototypes that then produce more bad labels.

Choosing thresholds needs to know which images genuinely contain the object, and
nobody has annotated that. This tool gets the answer without extra annotation by
holding each seed out in turn: score a seed image against the prototypes built
from all the *other* seeds. Those images are known to contain the object, so
their scores are a real sample of the positive distribution. Every unlabelled
image is then scored the same way, giving the mixed distribution the thresholds
have to cut through.

Two things are then worth reading:

- Where the held-out positives sit. The review threshold should be below almost
  all of them, or true matches never reach a human at all.
- Whether the unlabelled scores are bimodal. A clear valley between two humps is
  the natural place to cut; no valley means the features are not separating the
  object from the background and no threshold will rescue that.

Nothing is written to the dataset. This only reads and reports.

Usage:
    uv run python -m src.tools.calibrate_thresholds
"""

import numpy as np

from src.core.step4_propagation import (
    AUTO_ACCEPT_THRESHOLD,
    REVIEW_THRESHOLD,
    PatchPropagator,
    SeedPrototype,
    unit,
)

HISTOGRAM_BINS = 28
HISTOGRAM_WIDTH = 46

# Candidate thresholds are swept over this range when tabulating the trade-off.
SWEEP_START = 0.70
SWEEP_STOP = 0.96
SWEEP_STEP = 0.02

# A review threshold is only worth suggesting if it admits at least this share of
# the known positives. Anything stricter is discarding real matches to save
# review effort, which defeats the point of the pipeline.
TARGET_POSITIVE_RECALL = 0.95

# Below this many unlabelled images there is no distribution to speak of, so the
# absence of a valley says nothing about the data. Reporting "no valley found"
# in that case would read as evidence of a problem when it is only a lack of
# measurement.
MIN_VALLEY_SAMPLES = 20


def histogram(values: list[float], title: str, low: float, high: float) -> str:
    """Render values as a text histogram.

    Text rather than a plot so the output reads the same in a terminal, in the
    GUI console and pasted into a message, with no plotting dependency.

    Args:
        values: Scores to bin.
        title: Heading printed above the bars.
        low: Lower edge of the first bin.
        high: Upper edge of the last bin.

    Returns:
        str: The rendered histogram.
    """
    if not values:
        return f"{title}\n  (no values)"

    counts, edges = np.histogram(values, bins=HISTOGRAM_BINS, range=(low, high))
    peak = max(int(counts.max()), 1)

    lines = [title]
    for count, edge in zip(counts, edges[:-1], strict=False):
        bar = "#" * int(round(count / peak * HISTOGRAM_WIDTH))
        lines.append(f"  {edge:5.3f} | {bar:<{HISTOGRAM_WIDTH}} {count}")
    return "\n".join(lines)


def find_valley(values: list[float], low: float, high: float) -> float | None:
    """Locate the emptiest bin between the two busiest ones, if there are two.

    Args:
        values: Scores to analyse.
        low: Lower edge of the first bin.
        high: Upper edge of the last bin.

    Returns:
        float | None: Score at the valley, or None when the data is not bimodal.
    """
    if len(values) < MIN_VALLEY_SAMPLES:
        return None

    counts, edges = np.histogram(values, bins=HISTOGRAM_BINS, range=(low, high))
    peaks = np.argsort(counts)[::-1]

    first = int(peaks[0])
    second = next((int(p) for p in peaks[1:] if abs(int(p) - first) >= 3), None)
    if second is None:
        return None

    left, right = sorted((first, second))
    between = counts[left + 1 : right]
    if len(between) == 0:
        return None

    # A valley only means something if it is genuinely emptier than the humps.
    if between.min() > 0.35 * counts[[left, right]].min():
        return None

    return float(edges[left + 1 + int(between.argmin())])


class ThresholdCalibrator:
    """Scores held-out seeds and unlabelled images to inform threshold choice."""

    def __init__(self, propagator: PatchPropagator | None = None) -> None:
        """Reuse the propagator's data loading without loading SAM.

        Args:
            propagator: An existing propagator, or None to build a SAM-free one.
        """
        self.propagator = propagator or PatchPropagator.__new__(PatchPropagator)
        if propagator is None:
            self._configure_without_sam()

    def _configure_without_sam(self) -> None:
        """Set up just enough of the propagator to read grids and labels.

        SAM is only needed to turn regions into masks, which this tool never
        does, and loading it would cost seconds and VRAM for nothing.
        """
        from pathlib import Path

        from src.core.step4_propagation import IMAGE_DIR, LABEL_DIR, PATCH_DIR

        self.propagator.patch_dir = Path(PATCH_DIR)
        self.propagator.label_dir = Path(LABEL_DIR)
        self.propagator.image_dir = IMAGE_DIR

        if not self.propagator.patch_dir.exists():
            raise FileNotFoundError(f"[!] No patch grids at {self.propagator.patch_dir}. Run Step 2 first.")

        self.propagator.image_keys = sorted(p.stem for p in self.propagator.patch_dir.glob("*.npy"))
        if not self.propagator.image_keys:
            raise FileNotFoundError(f"[!] No patch grids inside {self.propagator.patch_dir}.")

    def best_score(self, image_key: str, prototypes: list[SeedPrototype]) -> float:
        """Score one image against a prototype set.

        Args:
            image_key: Image to score.
            prototypes: Prototypes to compare against.

        Returns:
            float: Highest patch similarity found.
        """
        grid = unit(self.propagator.load_patch_grid(image_key))
        return max(float((grid @ p.vector).max()) for p in prototypes)

    def leave_one_out(self, prototypes: list[SeedPrototype]) -> list[tuple[str, float]]:
        """Score each seed image against the prototypes of the other seeds.

        These images are known to contain the object, so the resulting scores are
        a genuine sample of the positive distribution.

        Args:
            prototypes: All seed prototypes.

        Returns:
            list: ``(image_key, score)`` per seed image with at least one other
            seed to compare against.
        """
        seed_keys = sorted({p.image_key for p in prototypes})
        results: list[tuple[str, float]] = []

        for held_out in seed_keys:
            others = [p for p in prototypes if p.image_key != held_out]
            if not others:
                continue
            results.append((held_out, self.best_score(held_out, others)))

        return results

    def target_scores(self, prototypes: list[SeedPrototype]) -> list[tuple[str, float]]:
        """Score every unlabelled image against the full prototype set.

        Args:
            prototypes: All seed prototypes.

        Returns:
            list: ``(image_key, score)`` per unlabelled image.
        """
        seed_keys = {p.image_key for p in prototypes}
        return [
            (key, self.best_score(key, prototypes))
            for key in self.propagator.image_keys
            if key not in seed_keys
        ]

    def sweep(self, positives: list[float], targets: list[float]) -> None:
        """Tabulate, for each candidate threshold, what it admits and what it costs.

        Args:
            positives: Scores of the held-out seeds.
            targets: Scores of the unlabelled images.
        """
        print("\n  threshold  known positives kept   unlabelled images at or above")
        print("  " + "-" * 66)

        steps = int(round((SWEEP_STOP - SWEEP_START) / SWEEP_STEP)) + 1
        for i in range(steps):
            level = SWEEP_START + i * SWEEP_STEP
            kept = sum(1 for s in positives if s >= level)
            above = sum(1 for s in targets if s >= level)

            recall = kept / len(positives) if positives else 0.0
            flag = "  <- loses known positives" if positives and recall < TARGET_POSITIVE_RECALL else ""
            share = f"{above / len(targets):.0%}" if targets else "n/a"

            print(
                f"     {level:.2f}     {kept:3}/{len(positives):<3} ({recall:5.0%})"
                f"        {above:5}  ({share}){flag}"
            )

    def recommend(self, positives: list[float], targets: list[float], valley: float | None) -> None:
        """Print threshold suggestions with the reasoning behind each.

        Args:
            positives: Scores of the held-out seeds.
            targets: Scores of the unlabelled images.
            valley: Score at the distribution valley, when one was found.
        """
        print("\n[+] Suggestions")

        if len(positives) < 3:
            print(
                f"    Only {len(positives)} known positive(s). That is too few to place a "
                "threshold on; label a few more seeds by hand and run this again."
            )
            return

        floor = float(np.percentile(positives, 5))
        print(f"    - 5th percentile of known positives : {floor:.4f}")
        print(f"    - lowest known positive             : {min(positives):.4f}")

        if valley is not None:
            print(f"    - valley in the unlabelled scores   : {valley:.4f}")
            review = min(floor, valley)
            print(
                f"\n    REVIEW >= {review:.2f} keeps essentially every known positive and sits "
                "at or below the gap between the two humps."
            )
        elif len(targets) < MIN_VALLEY_SAMPLES:
            review = floor
            print(f"    - only {len(targets)} unlabelled image(s), too few to look for a valley")
            print(
                f"\n    REVIEW >= {review:.2f} from the positives alone. This says nothing yet "
                "about how well the scores separate objects from background; that needs at "
                f"least {MIN_VALLEY_SAMPLES} unlabelled images to be visible at all."
            )
        else:
            review = floor
            print("    - no valley: the unlabelled scores form one continuous mass")
            print(
                f"\n    REVIEW >= {review:.2f} from the positives alone, but the missing valley "
                "matters more than the number. It means the features are not separating this "
                "object from the background, and no threshold fixes that. Check the debug "
                "overlays before trusting any result, and consider more varied seeds."
            )

        auto = float(np.percentile(positives, 50))
        print(
            f"    AUTO   >= {auto:.2f} is the median known positive: half of the matches this "
            "confident are ones a human already vouched for."
        )

        print(f"\n    Currently configured: AUTO {AUTO_ACCEPT_THRESHOLD}, REVIEW {REVIEW_THRESHOLD}")
        current_kept = sum(1 for s in positives if s >= AUTO_ACCEPT_THRESHOLD)
        if current_kept < len(positives) * 0.5:
            print(
                f"    Note: only {current_kept}/{len(positives)} known positives clear the "
                f"configured AUTO threshold, so most true matches are going to review."
            )
        print("    Nothing has been changed. Edit them in src/core/step4_propagation.py.")

    def run(self) -> None:
        """Compute both distributions and print the full report."""
        prototypes = self.propagator.build_prototypes()
        if not prototypes:
            print("[-] No seeds found. Annotate a few images with Step 3a or 3b first.")
            return

        seed_count = len({p.image_key for p in prototypes})
        print(f"[+] {len(prototypes)} prototype(s) from {seed_count} seed image(s).")
        print(f"[*] Scoring {len(self.propagator.image_keys)} cached image(s). SAM is not loaded.\n")

        loo = self.leave_one_out(prototypes)
        targets = self.target_scores(prototypes)

        positive_scores = [s for _, s in loo]
        target_values = [s for _, s in targets]

        if not positive_scores:
            print("[-] Only one seed image, so nothing can be held out. Add another seed.")
            return

        low = min([*positive_scores, *target_values, 0.5]) - 0.02
        high = max([*positive_scores, *target_values, 1.0]) + 0.02

        print("[+] Held-out seeds (known to contain the object)")
        for key, score in sorted(loo, key=lambda kv: -kv[1]):
            print(f"    {score:.4f}  {key}")

        print()
        print(histogram(positive_scores, "[+] Known positives", low, high))
        print()
        print(histogram(target_values, "[+] Unlabelled images", low, high))

        valley = find_valley(target_values, low, high)
        self.sweep(positive_scores, target_values)
        self.recommend(positive_scores, target_values, valley)


if __name__ == "__main__":
    try:
        ThresholdCalibrator().run()
    except Exception as e:
        print(f"[!] An error occurred: {e}")

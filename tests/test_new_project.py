"""Emptying one project's data without taking the next one's tools with it.

This is the most destructive thing in the repository, and the two ways it can be
wrong are opposites: leaving a project's data behind silently corrupts the next
one, and clearing too much throws away 700 MB of downloaded model weights. Both
are tested, and so is the dry run that stands between them.
"""

import json

from src.tools.new_project import human, main, measure


def build(root, *, with_models=True):
    """Populate a sandbox that looks like a finished project.

    Args:
        root: The sandboxed project root.
        with_models: Whether to place model weights alongside the project data.

    Returns:
        dict: The paths the tests assert on.
    """
    paths = {
        "raw": root / "data" / "raw" / "frame.jpg",
        "labels": root / "data" / "labels" / "frame.txt",
        "dedup": root / "data" / "deduplicated" / "frame.jpg",
        "weights": root / "runs" / "train-1" / "weights" / "best.pt",
        "dataset": root / "datasets" / "train" / "images" / "frame.jpg",
        "queue": root / "data" / "review_queue.json",
        "classes": root / "data" / "classes.json",
    }
    if with_models:
        paths["model"] = root / "data" / "models" / "sam_vit_b.pth"

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    paths["classes"].write_text(json.dumps({"classes": ["pallet"]}), encoding="utf-8")
    return paths


def test_a_dry_run_deletes_nothing(project_sandbox, capsys):
    """The default has to be safe: this is the one tool that destroys work.

    Args:
        project_sandbox: The sandboxed project root.
        capsys: Captured output.
    """
    paths = build(project_sandbox)

    assert main([]) == 0

    assert all(path.exists() for path in paths.values())
    assert "Nothing was deleted" in capsys.readouterr().out


def test_applying_clears_every_kind_of_project_state(project_sandbox):
    """Anything left behind is mixed into the next project without a word.

    Old frames survive deduplication because Step 1 adds rather than replaces,
    and an old checkpoint is what Step 7 will pre-label the new project with.

    Args:
        project_sandbox: The sandboxed project root.
    """
    paths = build(project_sandbox)

    assert main(["--apply"]) == 0

    for name in ("raw", "labels", "dedup", "weights", "dataset", "queue", "classes"):
        assert not paths[name].exists(), name


def test_the_model_weights_are_kept(project_sandbox):
    """DINOv3, SAM and the YOLO seed belong to no project and are slow to fetch.

    Args:
        project_sandbox: The sandboxed project root.
    """
    paths = build(project_sandbox)

    main(["--apply"])

    assert paths["model"].exists()


def test_the_class_scheme_can_be_kept_for_a_second_project_on_the_same_domain(project_sandbox):
    """A new dataset of the same objects should not need the classes retyped.

    Args:
        project_sandbox: The sandboxed project root.
    """
    paths = build(project_sandbox)

    main(["--apply", "--keep-classes"])

    assert paths["classes"].exists()
    assert not paths["labels"].exists()


def test_the_directories_survive_so_no_step_has_to_create_them(project_sandbox):
    """Steps write into these paths assuming they are there.

    Args:
        project_sandbox: The sandboxed project root.
    """
    build(project_sandbox)

    main(["--apply"])

    assert (project_sandbox / "data" / "labels").is_dir()
    assert (project_sandbox / "data" / "raw").is_dir()


def test_a_trained_detector_is_called_out_before_it_is_deleted(project_sandbox, capsys):
    """The checkpoint is the point of the whole exercise and is not in Git.

    Args:
        project_sandbox: The sandboxed project root.
        capsys: Captured output.
    """
    build(project_sandbox)

    main([])

    out = capsys.readouterr().out
    assert "trained detector" in out
    assert "best.pt" in out


def test_a_second_run_reports_an_empty_project_rather_than_a_second_success(project_sandbox, capsys):
    """Run twice, the second pass has nothing to do and should say so.

    The first version of this test asserted the message on a sandbox it assumed
    was empty. The fixture writes a classes.json, so the tool correctly found
    something to clear and the test failed — the test was wrong about its own
    starting conditions, not the tool about its job. Clearing first and then
    checking is the honest way to reach the empty case.

    Args:
        project_sandbox: The sandboxed project root.
        capsys: Captured output.
    """
    build(project_sandbox)
    assert main(["--apply"]) == 0
    capsys.readouterr()

    assert main(["--apply"]) == 0

    assert "already a fresh project" in capsys.readouterr().out


def test_a_missing_path_is_measured_as_nothing(project_sandbox):
    """Not every project writes every directory, and absence is not an error.

    Args:
        project_sandbox: The sandboxed project root.
    """
    assert measure(project_sandbox / "never_existed") == (0, 0)


def test_sizes_are_reported_at_a_readable_scale():
    """A report in bytes is a report nobody reads."""
    assert human(512) == "512 B"
    assert human(2048) == "2.0 KB"
    assert human(5 * 1024 * 1024) == "5.0 MB"
    assert human(3 * 1024**3) == "3.0 GB"

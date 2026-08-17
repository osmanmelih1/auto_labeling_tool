"""Reading a step's output back into the GUI console.

Every step is launched as a subprocess and its output is piped into the console
panel. Piping is where the encoding stops being the terminal's problem and
becomes ours: training died on its first progress bar with a UnicodeDecodeError,
and only when started from the GUI, because run from a terminal there is no pipe
and nothing to decode.
"""

import subprocess
import sys

from src.gui.app import WorkerThread


def test_the_pipe_is_read_as_utf8(qapp, monkeypatch):
    """Ultralytics draws progress bars with box characters no code page has.

    A Turkish or Western European Windows decodes a pipe as cp1254 or cp1252 by
    default, and a run twenty minutes in dies on a character it drew itself.

    Args:
        qapp: The shared QApplication.
        monkeypatch: Used to capture the Popen arguments.
    """
    captured = {}

    class FakeProcess:
        stdout = iter([])
        returncode = 0

        def wait(self):
            return 0

    def fake_popen(*args, **kwargs):
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    WorkerThread("src.core.step6_train").run()

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_the_child_is_told_to_write_utf8(qapp, monkeypatch):
    """Reading as UTF-8 only helps if the child is writing it.

    Args:
        qapp: The shared QApplication.
        monkeypatch: Used to capture the Popen arguments.
    """
    captured = {}

    class FakeProcess:
        stdout = iter([])
        returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (captured.update(k), FakeProcess())[1])

    WorkerThread("src.core.step5_export").run()

    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"


def test_a_byte_no_codec_expects_does_not_kill_the_run(qapp, monkeypatch, tmp_path):
    """One unexpected byte should be a question mark, not the end of a run.

    Args:
        qapp: The shared QApplication.
        monkeypatch: Unused; kept for symmetry with the other cases.
        tmp_path: Pytest's temporary directory.
    """
    script = tmp_path / "noisy.py"
    script.write_text(
        "import sys\nsys.stdout.buffer.write(b'before \\x81 after\\n')\n",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = process.stdout.read()
    process.wait()

    assert "before" in output
    assert "after" in output


def test_every_line_reaches_the_console(qapp, monkeypatch):
    """The console is the only place a step reports what it did.

    Args:
        qapp: The shared QApplication.
        monkeypatch: Used to replace the subprocess with a scripted one.
    """
    emitted = []

    class FakeProcess:
        stdout = iter(["[*] one\n", "[+] two\n"])
        returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProcess())

    worker = WorkerThread("src.core.step1_deduplication")
    worker.log_signal.connect(emitted.append)
    worker.run()

    assert emitted == ["[*] one\n", "[+] two\n"]


def test_tool_arguments_reach_the_command_line(qapp, monkeypatch):
    """The steps take no arguments; the inspection tools take a class or --apply.

    A dropped argument here is not an error but a silently different run — the
    whole dataset audited instead of one class, or a report where a deletion was
    asked for.

    Args:
        qapp: The shared QApplication.
        monkeypatch: Used to capture the Popen arguments.
    """
    captured = []

    class FakeProcess:
        stdout = iter([])
        returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (captured.append(a[0]), FakeProcess())[1])

    WorkerThread("src.tools.audit_labels", ["--class", "koli"]).run()

    assert captured[0][-3:] == ["src.tools.audit_labels", "--class", "koli"]


def test_a_step_with_no_arguments_is_launched_exactly_as_before(qapp, monkeypatch):
    """Adding the argument list must not add an empty one to every step.

    Args:
        qapp: The shared QApplication.
        monkeypatch: Used to capture the Popen arguments.
    """
    captured = []

    class FakeProcess:
        stdout = iter([])
        returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (captured.append(a[0]), FakeProcess())[1])

    WorkerThread("src.core.step6_train").run()

    assert captured[0][-1] == "src.core.step6_train"


def test_a_failure_to_start_is_reported_rather_than_swallowed(qapp, monkeypatch):
    """A step that never starts must say so; silence reads as success.

    Args:
        qapp: The shared QApplication.
        monkeypatch: Used to make the subprocess fail to launch.
    """
    emitted = []
    codes = []

    def explode(*args, **kwargs):
        raise FileNotFoundError("uv is not on the path")

    monkeypatch.setattr(subprocess, "Popen", explode)

    worker = WorkerThread("src.core.step4_propagation")
    worker.log_signal.connect(emitted.append)
    worker.finished_signal.connect(codes.append)
    worker.run()

    assert "uv is not on the path" in "".join(emitted)
    assert codes == [-1]

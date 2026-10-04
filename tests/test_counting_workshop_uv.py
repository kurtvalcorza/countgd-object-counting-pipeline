"""The detection and counting workshop runs in a uv isolated environment, never in the kernel.

Section 2 of the notebook builds a hash-locked CPython 3.12 venv and starts one long-lived process
with the venv interpreter (the carried cell runner); an IPython input transformer forwards every
later titled code cell to it. These tests read the notebook JSON and drive the carried cell runner
with this interpreter as a stand-in for the venv one. No package is installed and no model runs.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_Open_Vocabulary_Detection_and_Counting_Workshop.ipynb"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {c["id"]: c for c in NB["cells"]}
CODE = [c for c in NB["cells"] if c["cell_type"] == "code"]
LOCK = ROOT / "tools" / "counting-workshop-requirements.lock"
RUNNER = ROOT / "tools" / "counting_workshop_env_worker.py"

sys.path.insert(0, str(ROOT / "tools"))
import build_counting_workshop_carrier as carrier  # noqa: E402


def source(cell_id: str) -> str:
    return "".join(CELLS[cell_id]["source"])


def carried() -> dict[str, str]:
    tree = ast.parse(source("cnt-uv-carrier"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "CARRIED_FILES":
            return ast.literal_eval(node.value)
    raise AssertionError("CARRIED_FILES")


def transformer():
    """The route_cells_to_environment function of section 2, without IPython."""
    tree = ast.parse(source("ef509873"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "route_cells_to_environment"
    )
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "ef509873", "exec"), ns)
    return ns["route_cells_to_environment"]


def test_no_kernel_install_and_no_restart_guard():
    for cell in CODE:
        text = "".join(cell["source"])
        if cell["id"] == "cnt-uv-carrier":
            continue  # carries the lock text, whose comments name the uv install command
        assert not re.search(
            r"\bpip\s+install\b|-m\s*\"?pip", text.replace('"pip", "install"', "")
        ), cell["id"]
        for token in (
            "Restart session",
            "NUMPY_PRELOADED",
            "DIMER_NOTEBOOK_CI_PREINSTALLED",
            "if stale:",
        ):
            assert token not in text, (cell["id"], token)
    bootstrap = source("ef509873")
    assert '"pip", "install", "--python", str(PYTHON)' in bootstrap  # into the venv only


def test_carrier_cell_is_current_and_carries_the_lock_and_runner():
    assert carrier.main(["--check"]) == 0
    files = carried()
    assert files["requirements.lock.txt"] == LOCK.read_text(encoding="utf-8")
    assert files["cell_runner.py"] == RUNNER.read_text(encoding="utf-8")
    assert NB["metadata"]["dimer"]["carried_files"]["requirements.lock.txt"]["source"] == (
        "tools/counting-workshop-requirements.lock"
    )


def test_lock_pins_every_package_with_hashes_and_keeps_the_former_pins():
    text = LOCK.read_text(encoding="utf-8")
    requirements = re.findall(
        r"^([a-z0-9][a-z0-9._-]*)==([^\s\\]+) \\\n((?:    --hash=sha256:[0-9a-f]{64}.*\n)+)",
        text,
        re.M,
    )
    names = {name for name, _version, _hashes in requirements}
    assert len(requirements) == len(re.findall(r"^[a-z0-9][a-z0-9._-]*==", text, re.M)) > 40
    pins = {name: version for name, version, _hashes in requirements}
    for name, version in {
        "torch": "2.14.0",
        "torchvision": "0.29.0",
        "torchaudio": "2.11.0",
        "transformers": "4.57.6",
        "huggingface-hub": "0.36.2",
        "safetensors": "0.8.0",
        "numpy": "2.5.3",
        "pillow": "11.3.0",
        "scipy": "1.18.1",
        "matplotlib": "3.10.8",
        "pandas": "2.2.3",
    }.items():
        assert pins[name] == version, name
    assert "pip" not in names
    assert "--python-platform x86_64-manylinux_2_28" in text and "--generate-hashes" in text
    assert "Direct pins" in text and "former" in text


def test_install_is_hash_locked_wheels_only_with_a_pinned_uv_and_python():
    bootstrap = source("ef509873")
    for token in (
        '"--require-hashes", "--only-binary", ":all:"',
        '"--managed-python", "--python", "3.12.12"',
        "UV_SHA256 =",
        "uv-0.12.15-",
        'raise RuntimeError("uv wheel size/hash mismatch")',
        'platform.system() != "Linux" or platform.machine() != "x86_64"',
    ):
        assert token in bootstrap, token


def test_child_environment_drops_kernel_settings():
    bootstrap = source("ef509873")
    assert 'ENV["MPLBACKEND"] = "Agg"' in bootstrap
    assert (
        '("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")'
        in bootstrap
    )


def test_cells_run_with_the_venv_interpreter():
    bootstrap = source("ef509873")
    assert 'PYTHON = ENV_ROOT / "bin" / "python"' in bootstrap
    assert "DIMER_ENV = IsolatedEnvironment(PYTHON, CELL_RUNNER, ENV)" in bootstrap
    assert '[str(python), "-u", str(runner)]' in bootstrap
    assert "input_transformers_cleanup" in bootstrap
    route = transformer()
    kernel_only = {"cnt-uv-carrier", "ef509873"}
    for cell in CODE:
        lines = "".join(cell["source"]).splitlines(keepends=True)
        routed = route(lines)
        if cell["id"] in kernel_only:
            assert routed == lines, cell["id"]
            continue
        assert len(routed) == 1 and routed[0].startswith("DIMER_ENV.run("), cell["id"]
        call = ast.parse(routed[0]).body[0].value
        assert ast.literal_eval(call.args[0]) == "".join(lines)  # the cell's own code, unchanged
        both = ast.literal_eval(call.keywords[0].value)
        assert both is (cell["id"] == "daa99d90"), cell["id"]
    assert route(["x = 1\n"]) == ["x = 1\n"]  # untitled code (the kernel's own) is left alone


def test_routed_cells_reach_the_environment_byte_for_byte():
    """IPython's own clean-up transforms rewrite whitespace-only lines; routing must come first."""
    assert "input_transformers_cleanup[:] = [route_cells_to_environment] + [" in source("ef509873")
    inputtransformer2 = pytest.importorskip("IPython.core.inputtransformer2")
    manager = inputtransformer2.TransformerManager()
    manager.cleanup_transforms.insert(0, transformer())
    for cell in CODE:
        if cell["id"] in {"cnt-uv-carrier", "ef509873"}:
            continue
        text = "".join(cell["source"])
        call = ast.parse(manager.transform_cell(text)).body[0].value
        assert ast.literal_eval(call.args[0]).rstrip("\n") == text.rstrip("\n"), cell["id"]


def test_no_cell_line_over_2000_characters():
    for cell in NB["cells"]:
        for line in "".join(cell["source"]).split("\n"):
            assert len(line) <= 2000, (cell["id"], len(line))


def test_kernel_cells_import_only_the_standard_library_and_ipython():
    allowed = set(sys.stdlib_module_names) | {"IPython"}
    for cell_id in ("daa99d90", "cnt-uv-carrier", "ef509873"):
        for node in ast.walk(ast.parse(source(cell_id))):
            if isinstance(node, ast.Import):
                roots = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            assert roots <= allowed | {"google"}, (cell_id, roots)
    assert "import torch" in source("cnt-env-imports") and "DEVICE =" in source("cnt-env-imports")


def test_revision_recorded():
    assert NB["metadata"]["workshop_revision"] == "0.3.0-candidate"
    entry = NB["metadata"]["dimer"]["review_revisions"][-1]
    assert entry["change"] == "uv isolated environment" and entry["date"] == "2026-10-03"


# -------------------- the carried cell runner, driven with this interpreter as a stand-in


class Runner:
    def __init__(self, tmp_path: Path):
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(RUNNER)],
            cwd=tmp_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self.ready = self.event()

    def event(self) -> dict:
        line = self.process.stdout.readline()
        assert line, self.process.stderr.read()
        return json.loads(line)

    def send(self, request: dict) -> None:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

    def run(self, source: str, reply_to_calls=None) -> tuple[list[dict], dict]:
        self.send({"op": "exec", "name": "test", "source": source})
        events = []
        while True:
            event = self.event()
            if event["t"] == "call":
                self.send(reply_to_calls(event))
            if event["t"] == "done":
                return events, event
            events.append(event)

    def close(self) -> int:
        self.process.stdin.close()
        return self.process.wait(timeout=30)


@pytest.fixture
def runner(tmp_path):
    process = Runner(tmp_path)
    yield process
    process.close()


def test_runner_keeps_one_namespace_and_returns_prints_and_displays(runner):
    assert runner.ready["t"] == "ready" and runner.ready["protocol"].startswith("dimer.")
    events, done = runner.run("x = 41\nprint('hello')")
    assert done["ok"] and [e["text"] for e in events if e["t"] == "out"] == ["hello\n"]
    events, done = runner.run(
        "class Table:\n    def _repr_html_(self):\n        return '<b>t</b>'\n"
        "display(Table())\nx + 1"
    )
    shown = [e["data"] for e in events if e["t"] == "display"]
    assert shown[0]["text/html"] == "<b>t</b>" and shown[1]["text/plain"] == "42"
    events, done = runner.run(
        "import sys, __main__\nprint(__main__.x, globals() is vars(__main__))"
    )
    assert "".join(e["text"] for e in events if e["t"] == "out").startswith("41 True")


def test_runner_reports_errors_and_keeps_serving(runner):
    _events, done = runner.run("def f():\n    raise ValueError('bad row 3')\nf()")
    assert not done["ok"]
    assert done["error"]["type"] == "ValueError" and done["error"]["message"] == "bad row 3"
    assert "raise ValueError('bad row 3')" in done["error"]["traceback"]  # the cell's source line
    _events, done = runner.run("print('still here')")
    assert done["ok"]


def test_runner_receives_the_controls_with_tuples_and_paths(runner):
    sys.path.insert(0, str(RUNNER.parent))
    import counting_workshop_env_worker as worker

    values = {"GRID": (0.1, 0.2), "OUTPUT_ROOT": Path("outputs"), "FLAG": False, "N": None}
    runner.send({"op": "set", "values": worker.encode_value(values)})
    assert runner.event() == {"t": "done", "ok": True, "error": None}
    events, done = runner.run(
        "print(type(GRID).__name__, GRID, type(OUTPUT_ROOT).__name__.endswith('Path'), FLAG, N)"
    )
    assert (
        "".join(e["text"] for e in events if e["t"] == "out")
        == "tuple (0.1, 0.2) True False None\n"
    )


def test_runner_asks_the_kernel_for_the_upload_dialog(runner, tmp_path):
    calls = []

    def reply(event):
        calls.append(event)
        return {"files": None}

    events, done = runner.run("answer = host_upload('work/byod_upload.zip')\nprint(answer)", reply)
    assert done["ok"] and calls[0]["name"] == "upload"
    assert calls[0]["destination"] == "work/byod_upload.zip"
    assert "".join(e["text"] for e in events if e["t"] == "out") == "{'files': None}\n"


def test_runner_shows_figures_like_an_inline_backend(runner):
    pytest.importorskip("matplotlib")
    events, done = runner.run(
        "import matplotlib.pyplot as plt\nplt.plot([0, 1])\nplt.show()\n"
        "plt.figure()\nplt.plot([1, 0])"
    )
    assert done["ok"]
    images = [e for e in events if e["t"] == "display" and "image/png" in e["data"]]
    assert len(images) == 2  # plt.show() and the figure left open at the end of the cell

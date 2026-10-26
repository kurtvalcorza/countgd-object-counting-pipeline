"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import importlib
import io
import json
import os
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/countgd_object_counting_colab.ipynb"
BYOD_CELL = "countgd_object_counting-28"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


SOURCE = _cell(BYOD_CELL)
STEM = "countgd_object_counting"
MODULES = [importlib.import_module(f"countgd_pipeline.{name}") for name in ("config", "metrics", "synthetic", "samples", "pipeline")]
EARLIER = {"splits": "old", "train_records": "old", "test_records": "old", "data_source": "old", "adapted_syn": "old"}


def _png(value: int = 120) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (value, value, value)).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _labels(n: int, *, points: bool = True, files=None) -> dict:
    rows = ["id,file,label,count,points"]
    entries = {}
    for i in range(n):
        name = (files or {}).get(i, f"img{i}.png")
        rows.append(f'r{i},{name},apple,1,"[[4, 4]]"' if points else f"r{i},{name},apple,1,")
        entries[f"img{i}.png"] = _png(10 + i)
    return {"labels.csv": "\n".join(rows) + "\n", **entries}


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    (out / f"{STEM}_adapter").mkdir(parents=True, exist_ok=True)
    for name in (f"{STEM}_train.csv", f"{STEM}_result.json", f"{STEM}_evaluation_report.json", "provenance.json"):
        (out / name).write_text("{}", encoding="utf-8")


def _namespace() -> dict:
    ns: dict = {"os": os, "Path": Path}
    for module in MODULES:
        ns.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    ns.update(EARLIER, pipe="kept")
    return ns


def _run(monkeypatch, tmp_path, path: str, *, use_byod=True, colab=None, source=None, ns=None) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    _seed_outputs(tmp_path)
    source = (source or SOURCE).replace("USE_BYOD = False", f"USE_BYOD = {use_byod}", 1).replace("BYOD_ZIP_PATH = ''", f"BYOD_ZIP_PATH = {path!r}", 1)
    ns = ns if ns is not None else _namespace()
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime-host\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    (tmp_path / "zip.txt").write_bytes(_zip(_labels(20)))
    (tmp_path / "nolabels.zip").write_bytes(_zip({"img0.png": _png()}))
    (tmp_path / "columns.zip").write_bytes(_zip({"labels.csv": "id,file\nr0,img0.png\n", "img0.png": _png()}))
    (tmp_path / "evil.zip").write_bytes(_zip({**_labels(20), "../evil.png": _png()}))
    (tmp_path / "missing_image.zip").write_bytes(_zip({k: v for k, v in _labels(20).items() if k != "img3.png"}))
    (tmp_path / "not_image.zip").write_bytes(_zip({**_labels(20), "img2.png": b"text"}))
    (tmp_path / "no_points.zip").write_bytes(_zip(_labels(20, points=False)))
    (tmp_path / "few.zip").write_bytes(_zip(_labels(3)))
    return {
        "text file (the scripted /etc/hostname run)": ("hostname", "its contents are not a zip archive"),
        "folder": ("folder", "does not exist or is not a file"),
        "missing path": ("nope.zip", "does not exist or is not a file"),
        "zip without .zip suffix": ("zip.txt", "BYOD needs a .zip file"),
        "no labels.csv": ("nolabels.zip", "exactly one labels.csv"),
        "missing columns": ("columns.zip", "labels.csv needs the columns"),
        "unsafe member": ("evil.zip", "absolute or parent-directory"),
        "named image missing": ("missing_image.zip", "is not in the zip"),
        "not an image": ("not_image.zip", "is not an image Pillow can decode"),
        "count without points": ("no_points.zip", "neither points nor boxes"),
        "too few to split": ("few.zip", "too few to split"),
    }


CASES = ["text file (the scripted /etc/hostname run)", "folder", "missing path", "zip without .zip suffix", "no labels.csv", "missing columns", "unsafe member", "named image missing", "not an image", "count without points", "too few to split"]


@pytest.mark.parametrize("case", CASES)
def test_byod_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    file_name, rule = _cases(tmp_path)[case]
    ns = _namespace()
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, str(tmp_path / file_name), ns=ns)
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == []
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert ns["pipe"] == "kept"


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "", colab=None)
    assert _refusal_line(caught.value).startswith("Refused: (no file): USE_BYOD is on but BYOD_ZIP_PATH is empty, and this runtime has no upload dialog")


def test_cancelled_upload_is_one_refused_line(monkeypatch, tmp_path):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {})
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "", colab=colab)
    assert _refusal_line(caught.value).startswith("Refused: (no file): No file was uploaded")


def test_uploaded_file_is_named_in_the_refusal(monkeypatch, tmp_path):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {"mine.zip": b"not a zip"})
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "", colab=colab)
    assert _refusal_line(caught.value).startswith("Refused: mine.zip: ")


def test_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "good.zip").write_bytes(_zip(_labels(20)))
    ns = _namespace()

    def broken(path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run(monkeypatch, tmp_path, str(tmp_path / "good.zip"), ns=ns)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing is removed for an internal error


def test_sample_path_errors_are_not_refusals(monkeypatch, tmp_path):
    ns = _namespace()

    def broken(split):
        raise ValueError("sample generator broke")

    ns["build_synthetic_dataset"] = broken
    with pytest.raises(ValueError, match="sample generator broke") as caught:
        _run(monkeypatch, tmp_path, "", use_byod=False, ns=ns)
    assert not getattr(caught.value, "dimer_refusal", False)


def test_good_byod_zip_reaches_the_split(monkeypatch, tmp_path):
    (tmp_path / "good.zip").write_bytes(_zip(_labels(20)))
    ns = _namespace()
    seen = {}

    def stop_after_split(records, seed):
        seen["n"] = len(records)
        raise RuntimeError("stop after the BYOD branch")

    ns_source = SOURCE.replace("        splits = split_byod(records, SPLIT_SEED)\n", "        splits = stop_after_split(records, SPLIT_SEED)\n", 1)
    assert ns_source != SOURCE
    ns["stop_after_split"] = stop_after_split
    with pytest.raises(RuntimeError, match="stop after the BYOD branch"):
        _run(monkeypatch, tmp_path, str(tmp_path / "good.zip"), ns=ns, source=ns_source)
    assert seen["n"] == 20


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_ZIP_PATH='/etc/hostname': a file present in every fresh runtime."""
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, "/etc/hostname")
    assert _refusal_line(caught.value).startswith("Refused: hostname: BYOD needs a .zip file; its contents are not a zip archive")

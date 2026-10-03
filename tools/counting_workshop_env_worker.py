"""Cell runner of the counting workshop notebook's isolated environment.

Notebook: tutorials/DIMER_Open_Vocabulary_Detection_and_Counting_Workshop.ipynb.

The workshop notebook carries this file byte-for-byte and starts it once, in section 2, with the
interpreter of a uv-built CPython 3.12 virtual environment whose packages come from a hash-locked,
wheels-only install. From then on the notebook kernel forwards each code cell's source here and only
shows what comes back: nothing is installed into the kernel, so Run all needs no restart.

Protocol (one JSON object per line):

* requests on stdin: ``{"op": "exec", "name": ..., "source": ...}`` runs a cell in one persistent
  ``__main__`` namespace (the last bare expression is displayed, as in a notebook);
  ``{"op": "set", "values": {name: value}}`` copies plain values (the notebook controls) into it.
* events on stdout: ``ready``, ``out`` (print text), ``display`` (a MIME bundle: text/plain,
  text/html or image/png), ``call`` (ask the kernel for something only it has, such as the Colab
  upload dialog; the reply arrives as the next stdin line) and ``done``.

At start-up the original stdout descriptor is kept for events and descriptor 1 is pointed at
stderr, so output that compiled libraries write straight to descriptors 1 or 2 reaches the kernel
through the stderr pipe and can never corrupt the event stream.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import linecache
import os
import signal
import sys
import threading
import traceback
import types
from pathlib import PurePath

PROTOCOL = "dimer.counting-workshop.env-cell.v1"


class _Channel:
    """Line-delimited JSON events to the kernel; thread-safe."""

    def __init__(self, fd: int) -> None:
        self._stream = os.fdopen(fd, "w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def send(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=True)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()


class _EventStream(io.TextIOBase):
    """A text stream whose writes become ``out`` events (stdout or stderr), a line at a time."""

    def __init__(self, channel: _Channel, name: str) -> None:
        self._channel = channel
        self._name = name
        self._pending = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self._pending += text
        if "\n" in text or "\r" in text or len(self._pending) > 8192:
            self.flush()
        return len(text)

    def flush(self) -> None:
        if self._pending:
            text, self._pending = self._pending, ""
            self._channel.send({"t": "out", "name": self._name, "text": text})

    def isatty(self) -> bool:
        return False


def _bundle(obj: object) -> dict:
    """The MIME bundle a notebook would show for ``obj``."""
    data = {"text/plain": repr(obj)}
    savefig = getattr(obj, "savefig", None)
    if callable(savefig) and type(obj).__name__ == "Figure":
        buffer = io.BytesIO()
        savefig(buffer, format="png", bbox_inches="tight")
        data["image/png"] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return data
    to_html = getattr(obj, "_repr_html_", None)
    if callable(to_html):
        try:
            html = to_html()
        except Exception:  # noqa: BLE001 - fall back to text/plain, as a notebook does
            html = None
        if html:
            data["text/html"] = html
    return data


class CellRunner:
    """Runs forwarded cells in one persistent namespace that plays the role of ``__main__``."""

    def __init__(self, channel: _Channel, replies) -> None:
        self.channel = channel
        self.replies = replies
        module = types.ModuleType("__main__")
        module.__dict__.update(
            {"__builtins__": __builtins__, "display": self.display, "host_upload": self.host_upload}
        )
        sys.modules["__main__"] = module  # pickling cell-defined classes works as in a kernel
        self.namespace = module.__dict__
        self._pyplot = None

    # ---- what cells can call
    def display(self, *objects: object) -> None:
        sys.stdout.flush()
        for obj in objects:
            self.channel.send({"t": "display", "data": _bundle(obj)})

    def host_upload(self, destination) -> dict:
        """Open the kernel's upload dialog (Colab); save a single uploaded file to ``destination``.

        Returns ``{"files": n}`` (``n`` files were chosen; the file is saved only when
        ``n == 1``) or ``{"files": None}`` when the kernel has no upload dialog.
        """
        self.channel.send({"t": "call", "name": "upload", "destination": os.fspath(destination)})
        line = self.replies.readline()
        if not line:
            raise RuntimeError("the notebook kernel closed the connection during an upload")
        return json.loads(line)

    # ---- figures, as a notebook's inline backend shows them
    def _install_pyplot(self) -> None:
        try:
            import matplotlib
        except ImportError:  # the hash lock always installs it; tests may run without it
            return

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        def show(*_args, **_kwargs) -> None:
            self._flush_figures()

        plt.show = show
        self._pyplot = plt

    def _flush_figures(self) -> None:
        plt = self._pyplot
        if plt is None:
            return
        for number in plt.get_fignums():
            figure = plt.figure(number)
            self.display(figure)
        plt.close("all")

    # ---- requests
    def run(self, name: str, source: str) -> None:
        filename = f"<cell {name}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        tree = ast.parse(source, filename=filename)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, filename, "exec"), self.namespace)  # noqa: S102 - the notebook's own cell
        if last is not None:
            value = eval(compile(last, filename, "eval"), self.namespace)  # noqa: S307
            if value is not None:
                self.display(value)
        self._flush_figures()

    def set_values(self, values: dict) -> None:
        self.namespace.update(values)


def _decode(value):
    """Rebuild the plain values the kernel encoded (tuples and paths are tagged)."""
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(_decode(v) for v in value["__tuple__"])
        if set(value) == {"__path__"}:
            from pathlib import Path

            return Path(value["__path__"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def encode_value(value):
    """Kernel side of ``_decode`` (also used by the notebook to forward its controls)."""
    if isinstance(value, tuple):
        return {"__tuple__": [encode_value(v) for v in value]}
    if isinstance(value, PurePath):
        return {"__path__": os.fspath(value)}
    if isinstance(value, dict):
        return {str(k): encode_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encode_value(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="isolated-environment cell runner").parse_args(argv)
    sys.stdout.flush()
    channel = _Channel(os.dup(1))
    os.dup2(2, 1)
    replies = sys.stdin
    sys.stdin = io.StringIO("")  # cells never read the request pipe
    sys.stdout = _EventStream(channel, "stdout")
    sys.stderr = _EventStream(channel, "stderr")
    runner = CellRunner(channel, replies)
    runner._install_pyplot()
    channel.send(
        {
            "t": "ready",
            "protocol": PROTOCOL,
            "python": sys.version.split()[0],
            "executable": sys.executable,
        }
    )
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # an interrupt only stops a running cell
    for line in replies:
        request = json.loads(line)
        error = None
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            if request["op"] == "exec":
                runner.run(request["name"], request["source"])
            elif request["op"] == "set":
                runner.set_values(_decode(request["values"]))
            else:
                raise ValueError(f"unknown request {request['op']!r}")
        except BaseException as exc:  # noqa: BLE001 - report every failure, then keep serving
            if isinstance(exc, SystemExit):
                error = {"type": "SystemExit", "message": str(exc.code), "traceback": ""}
            else:
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": "".join(traceback.format_exception(exc)),
                }
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        sys.stdout.flush()
        sys.stderr.flush()
        channel.send({"t": "done", "ok": error is None, "error": error})
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The workshop install cell must run in one pass on a Colab kernel with NumPy loaded."""
import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "tutorials"
    / "DIMER_Open_Vocabulary_Detection_and_Counting_Workshop.ipynb"
)


def install_cell():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    return next(cell for cell in cells if "pip" in cell and "install" in cell)


def test_install_keeps_a_numpy_the_kernel_already_loaded():
    # Colab imports NumPy 2.1.3 at startup; installing 2.5.3 underneath it broke later imports
    # (Notebook Spec RUN10 forbids a manual restart).
    cell = install_cell()
    assert "NUMPY_PRELOADED" in cell and '"numpy" in sys.modules' in cell
    assert "if stale:" in cell and "Restart session" in cell

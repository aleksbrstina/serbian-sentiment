"""Execute Jupyter notebooks in place so committed outputs match the current code.

Usage:
    python scripts/execute_notebooks.py notebooks/01_eda.ipynb [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute(path: Path, timeout: int = 600) -> None:
    """Run every cell of the notebook at ``path`` with the repo root as working dir."""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": "."}}
    )
    client.execute()
    nbformat.write(nb, path)


def main(argv: list[str]) -> int:
    """Execute each notebook given on the command line."""
    for arg in argv:
        execute(Path(arg))
        print(f"Executed {arg}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

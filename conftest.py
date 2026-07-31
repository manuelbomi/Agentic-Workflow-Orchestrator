"""Pytest root conftest.

Makes the ``src``-layout ``orchestrator`` package importable without
requiring an editable ``pip install -e .`` step -- this keeps the project's
setup down to ``pip install -r requirements.txt`` as documented in the
README.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

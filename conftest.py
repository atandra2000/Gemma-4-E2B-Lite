"""Puts the repo root on sys.path so tests can `import models`.

Also puts the LLM workspace root on sys.path so `shared_data` (the workspace
data pipeline the data adapter reads through) is importable from tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

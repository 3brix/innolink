"""Make the project root importable so `config`, `preprocessing`, `analysis`
resolve no matter how pytest is invoked."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

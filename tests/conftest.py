"""Shared test setup: src/ on the import path, no log files, and a loader for the numbered pipeline scripts."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

os.environ["PIPELINE_NO_FILE_LOG"] = "1"  # importing a pipeline script must never overwrite the log of a real run
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _load(filename: str):
    """Import a pipeline script whose name starts with a digit, e.g. 03_prepare_audio.py (imported once)."""
    name = "pipeline_" + filename.removesuffix(".py")
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SRC / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(scope="session")
def script():
    return _load

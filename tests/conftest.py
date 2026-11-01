import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)


@pytest.fixture(scope="session")
def demo_out():
    from dsp import demo
    return demo.run()

from pathlib import Path

import pytest

from intents.data import read_csv

FIXTURE = Path(__file__).parent / "fixtures" / "mini.csv"


@pytest.fixture
def mini():
    return read_csv(FIXTURE)

from pathlib import Path
import pytest


@pytest.fixture
def fixture_dir():
    yield Path("./tests/fixtures/")

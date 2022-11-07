import pytest

from .asserts import _assert_tokenize
from musicaiz.datasets import JSBChorales
from musicaiz.tokenizers import MMMTokenizerArguments


@pytest.fixture
def dataset_path(fixture_dir):
    return fixture_dir / "datasets" / "jsbchorales"


def test_JSBChorales_tokenize(dataset_path):
    # initialize tokenizer args
    args = MMMTokenizerArguments(
        prev_tokens="",
        windowing=False,
        time_unit="SIXTEENTH",
        num_programs=None,
        shuffle_tracks=True,
        track_density=False,
        time_sig=True,
        velocity=True,
        tempo=False
    )
    # initialize dataset
    dataset = JSBChorales()
    assert dataset.name == "jsb_chorales"
    _assert_tokenize(dataset_path, dataset, args)

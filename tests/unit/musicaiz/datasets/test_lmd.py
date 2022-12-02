import pytest

from .asserts import _assert_tokenize
from musicaiz.datasets import LakhMIDI
from musicaiz.tokenizers import MMMTokenizerArguments


@pytest.fixture
def dataset_path(fixture_dir):
    return fixture_dir / "datasets" / "lmd"


def test_LakhMIDI_get_metadata(dataset_path):

    expected = {
        "ABBA/Andante, Andante.mid": {
            "composer": "ABBA",
            "period": "",
            "genre": "",
            "split": "train",
        }
    }

    dataset = LakhMIDI()
    got = dataset.get_metadata(dataset_path)

    assert got.keys() == expected.keys()
    for got_v, exp_v in zip(got.values(), expected.values()):
        assert set(got_v.values()) == set(exp_v.values())


def test_LakhMIDI_tokenize(dataset_path):
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
    dataset = LakhMIDI()
    assert dataset.name == "lakh_midi"
    _assert_tokenize(dataset_path, dataset, args)

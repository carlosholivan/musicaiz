import pytest

from .asserts import _assert_tokenize
from musicaiz.datasets import Maestro
from musicaiz.tokenizers import MMMTokenizerArguments


@pytest.fixture
def dataset_path(fixture_dir):
    return fixture_dir / "datasets" / "maestro"


def test_Maestro_get_metadata(dataset_path):

    expected = {
        "2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi": {
            "composer": "ALBAN_BERG",
            "period": "ROMANTICISM",
            "genre": "SONATA",
            "split": "train",
        }
    }

    dataset = Maestro()
    got = dataset.get_metadata(dataset_path)

    assert got.keys() == expected.keys()
    for got_v, exp_v in zip(got.values(), expected.values()):
        assert set(got_v.values()) == set(exp_v.values())


def test_Maestro_tokenize(dataset_path):
    # initialize tokenizer args
    args = MMMTokenizerArguments(
        prev_tokens="",
        windowing=False,
        time_unit="HUNDRED_TWENTY_EIGHT",
        num_programs=None,
        shuffle_tracks=True,
        track_density=False,
        time_sig=True,
        velocity=True,
        tempo=False
    )
    # initialize dataset
    dataset = Maestro()
    assert dataset.name == "maestro"
    _assert_tokenize(dataset_path, dataset, args)

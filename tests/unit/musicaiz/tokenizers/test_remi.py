import pytest


from musicaiz.tokenizers import (
    REMITokenizer,
    REMITokenizerArguments,
)
from .test_mmm import (
    _assert_valid_musa_obj,
    musa_obj_tokens,
    musa_obj_abs,
    midi_sample,
)


@pytest.fixture
def remi_tokens(fixture_dir):
    tokens_path = fixture_dir / "tokenizers" / "remi_tokens.txt"
    text_file = open(tokens_path, "r")
    # read whole file to a string
    yield text_file.read()


def test_REMITokenizer_split_tokens_by_bar(remi_tokens):
    tokens = remi_tokens.split(" ")
    expected_bar_1 = [
        [
            "BAR=0",
            "TIME_SIG=4/4",
            "SUB_BEAT=4",
            "TEMPO=120",
            "INST=30",
            "PITCH=69",
            "DUR=4",
            "VELOCITY=127",
            "PITCH=64",
            "DUR=8",
            "VELOCITY=127",
            "SUB_BEAT=8",
            "PITCH=67",
            "DUR=4",
            "VELOCITY=127",
            "SUB_BEAT=12",
            "PITCH=64",
            "DUR=4",
            "VELOCITY=127",
        ]
    ]
    got = REMITokenizer.split_tokens_by_bar(tokens)
    assert set(expected_bar_1[0]) == set(got[0])


def test_REMITokenizer_split_tokens_by_subbeat(remi_tokens):
    tokens = remi_tokens.split(" ")
    expected_subbeats_bar_1 = [
        [
            "BAR=0",
            "TIME_SIG=4/4",
        ],
        [
            "SUB_BEAT=4",
            "TEMPO=120",
            "INST=30",
            "PITCH=69",
            "DUR=4",
            "VELOCITY=127",
            "PITCH=64",
            "DUR=8",
            "VELOCITY=127",
        ],
        [
            "SUB_BEAT=8",
            "PITCH=67",
            "DUR=4",
            "VELOCITY=127",
        ],
        [
            "SUB_BEAT=12",
            "PITCH=64",
            "DUR=4",
            "VELOCITY=127",
        ]
    ]
    got = REMITokenizer.split_tokens_by_subbeat(tokens)
    for i in range(len(expected_subbeats_bar_1)):
        assert set(expected_subbeats_bar_1[i]) == set(got[i])


def test_REMITokenizer_tokens_to_musa_a(remi_tokens, musa_obj_abs):
    # Test case: 1 polyphonic instrument, absolute timings
    got = REMITokenizer.tokens_to_musa(
        tokens=remi_tokens,
        sub_beat="SIXTEENTH"
    )
    _assert_valid_musa_obj(got, musa_obj_abs)


def test_REMITokenizer_get_tokens_analytics(remi_tokens):
    got = REMITokenizer.get_tokens_analytics(remi_tokens)
    expected_total_tokens = 33
    expected_unique_tokens = 16
    expected_total_notes = 7
    expected_unique_notes = 4
    expected_total_bars = 2
    expected_total_instruments = 1
    expected_total_pieces = 1

    assert expected_total_pieces == got["total_pieces"]
    assert expected_total_tokens == got["total_tokens"]
    assert expected_unique_tokens == got["unique_tokens"]
    assert expected_total_notes == got["total_notes"]
    assert expected_unique_notes == got["unique_notes"]
    assert expected_total_bars == got["total_bars"]
    assert expected_total_instruments == got["total_instruments"]


def test_REMITokenizer_tokenize_bars(midi_sample, remi_tokens):

    expected = remi_tokens

    args = REMITokenizerArguments(sub_beat="SIXTEENTH")
    tokenizer = REMITokenizer(
        midi_sample,
        args=args
    )
    got = tokenizer.tokenize_bars()
    assert got == expected


def test_REMITokenizer_tokenize_file(midi_sample):
    args = REMITokenizerArguments(sub_beat="SIXTEENTH")
    tokenizer = REMITokenizer(midi_sample, args)
    got = tokenizer.tokenize_file()
    assert got != ""

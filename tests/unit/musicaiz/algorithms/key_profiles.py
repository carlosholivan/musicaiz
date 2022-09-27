import pytest
import math

from musicaiz.structure import Note
from musicaiz.algorithms import (
    _keys_correlations,
    key_detection
)


@pytest.fixture
def durations():
    return {
        "C": 432,
        "C_SHARP": 231,
        "D": 0,
        "D_SHARP": 405,
        "E": 12,
        "F": 316,
        "F_SHARP": 4,
        "G": 126,
        "G_SHARP": 612,
        "A": 0,
        "A_SHARP": 191,
        "B": 1,
    }


@pytest.fixture
def expected_k_k_corr():
    # from http://rnhart.net/articles/key-finding/
    return {
        "C_MAJOR": -0.00009,
        "C_MINOR": 0.622,
        "C_SHARP_MAJOR": 0.538,
        "C_SHARP_MINOR": 0.094,
        "D_MAJOR": -0.741,
        "D_MINOR": -0.313,
        "D_SHARP_MAJOR": 0.579,
        "D_SHARP_MINOR": 0.152,
        "E_MAJOR": -0.269,
        "E_MINOR": -0.4786,
        "F_MAJOR": 0.101,
        "F_MINOR": 0.775,
        "F_SHARP_MAJOR": -0.043,
        "F_SHARP_MINOR": -0.469,
        "G_MAJOR": -0.464,
        "G_MINOR": -0.127,
        "G_SHARP_MAJOR": 0.970,
        "G_SHARP_MINOR": 0.391,
        "A_MAJOR": -0.582,
        "A_MINOR": -0.176,
        "A_SHARP_MAJOR": 0.113,
        "A_SHARP_MINOR": 0.250,
        "B_MAJOR": -0.201,
        "B_MINOR": -0.721,
    }


@pytest.fixture
def notes():
    return [
        Note(pitch=18, start=0.0, end=1.0, velocity=127), # F#
        Note(pitch=16, start=1.0, end=2.0, velocity=127), # E
        Note(pitch=18, start=2.0, end=4.0, velocity=127), # F#
        Note(pitch=18, start=2.0, end=4.0, velocity=127), # D
    ]


def test_keys_correlations(durations, expected_k_k_corr):
    corr = _keys_correlations(durations, "k-k")
    assert len(corr.keys()) == len(expected_k_k_corr.keys())
    for got_corr_v, expected_corr_v in zip(corr.values(), expected_k_k_corr.values()):
        assert math.isclose(got_corr_v, expected_corr_v, abs_tol=0.001)


def test_key_detection_krumhansl(notes):
    # Test KrumhanslKessler with different num of notes
    expected_key_1_note = "F_SHARP_MAJOR"
    got_key_1_note = key_detection(notes[0:1], "k-k")
    assert expected_key_1_note == got_key_1_note

    expected_key_2_notes = "E_MAJOR"
    got_key_2_notes = key_detection(notes[0:2], "k-k")
    assert expected_key_2_notes == got_key_2_notes

    expected_key_all_notes = "F_SHARP_MINOR"
    got_key_all_notes = key_detection(notes, "k-k")
    assert expected_key_all_notes == got_key_all_notes


def test_key_detection_temperley(notes):
    # Test Temperley with different num of notes
    expected_key_1_note = "B_MINOR"
    got_key_1_note = key_detection(notes[0:1], "temperley")
    assert expected_key_1_note == got_key_1_note

    expected_key_2_notes = "B_MINOR"
    got_key_2_notes = key_detection(notes[0:2], "temperley")
    assert expected_key_2_notes == got_key_2_notes

    expected_key_all_notes = "B_MINOR"
    got_key_all_notes = key_detection(notes, "temperley")
    assert expected_key_all_notes == got_key_all_notes


def test_key_detection_albrecht_shanahan(notes):
    # Test AlbrechtShanahan with different num of notes
    expected_key_1_note = "F_SHARP_MAJOR"
    got_key_1_note = key_detection(notes[0:1], "a-s")
    assert expected_key_1_note == got_key_1_note

    expected_key_2_notes = "E_MAJOR"
    got_key_2_notes = key_detection(notes[0:2], "a-s")
    assert expected_key_2_notes == got_key_2_notes

    expected_key_all_notes = "B_MINOR"
    got_key_all_notes = key_detection(notes, "a-s")
    assert expected_key_all_notes == got_key_all_notes


def test_key_detection_albrecht_shanahan(notes):
    # Test SignatureFifths with different num of notes
    expected_key_1_note = "D_MAJOR"
    got_key_1_note = key_detection(notes[0:1], "5ths")
    assert expected_key_1_note == got_key_1_note

    expected_key_2_notes = "D_MAJOR"
    got_key_2_notes = key_detection(notes[0:2], "5ths")
    assert expected_key_2_notes == got_key_2_notes

    expected_key_all_notes = "D_MAJOR"
    got_key_all_notes = key_detection(notes, "5ths")
    assert expected_key_all_notes == got_key_all_notes

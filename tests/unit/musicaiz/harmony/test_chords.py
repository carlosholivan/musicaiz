import pytest


# Our modules
from musicaiz.harmony import (
    AllChords,
    Chord,
)


# ===============AllChords class Tests=================
# =====================================================
def test_Chord_split_chord_name_a():
    # Test case: Valid chord
    chord_name = "C#mb5"
    expected_note = "C#"
    expected_quality = "mb5"
    got_note, got_quality = Chord.split_chord_name(chord_name)
    assert expected_note == got_note
    assert expected_quality == got_quality


def test_Chord_split_chord_name_b():
    # Test case: Valid chord, note with double sharps (not valid)
    chord_name = "C##mb5"
    with pytest.raises(ValueError):
        Chord.split_chord_name(chord_name)


def test_Chords_split_chord_name_c():
    # Test case: Bad character
    chord_name = "---"
    with pytest.raises(ValueError):
        Chord.split_chord_name(chord_name)


def test_Chords_split_chord_name_d():
    # Test case: Invalid chord quality (valid note)
    chord_name = "Cmb55"
    with pytest.raises(ValueError):
        Chord.split_chord_name(chord_name)


def test_Chord_get_chord_from_name_a():
    # Test case: Invalid chord quality (valid note)
    chord_name = "Cmb5"
    expected = AllChords.DIMINISHED_SEVENTH
    got = AllChords.get_chord_from_name(chord_name)
    assert expected == got


def test_AllChords_a():
    # Test case: Initialize with valid chord
    chord_name = "Cm7b5"
    got = Chord(chord_name)
    assert got.chord == AllChords.HALF_DIMINISHED_SEVENTH
    assert got.quality == "m7b5"
    assert got.root_note == "C"


def test_AllChords_b():
    # Test case: Initialize with no input chord
    got = Chord()
    assert got.chord is None
    assert got.quality is None
    assert got.root_note is None


@pytest.mark.skip("Fix this when it's implemented")
def test_AllChords_get_notes_a():
    # Test case: Initialize with valid chord
    chord_name = "Gm7b5"
    expected = ["G", "Bb", "Db", "F"]
    chord = Chord(chord_name)
    got = chord.get_notes()
    assert set(expected) == set(got)


@pytest.mark.skip("Fix this when it's implemented")
def test_AllChords_get_notes_b():
    # Test case: Initialize with valid chord
    chord_name = "G#M7"
    expected = ["G#", "B#", "D#", "F##"]
    chord = Chord(chord_name)
    got = chord.get_notes()
    assert set(expected) == set(got)


@pytest.mark.skip("Fix this when it's implemented")
def test_AllChords_get_notes_c():
    # Test case: Add inversion diverse than 0
    chord_name = "G#M7"
    inversion = 2
    expected = ["Eb", "G", "G#", "C"]
    chord = Chord(chord_name)
    got = chord.get_notes(inversion)
    assert set(expected) == set(got)


def test_AllChords_get_notes_d():
    # Test case: Add invalid inversion
    chord_name = "G#M7"
    inversion = 8
    chord = Chord(chord_name)
    with pytest.raises(ValueError):
        chord.get_notes(inversion)

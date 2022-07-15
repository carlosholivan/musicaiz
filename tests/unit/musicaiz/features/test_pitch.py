# Our modules
from musicaiz.features import (
    get_highest_lowest_pitches,
    get_pitch_range,
    get_pitch_classes,
    get_note_density,
)
from musicaiz.structure import (
    Note
)


# ===============PitchStatistics class Tests===========
# =====================================================
def test_get_highest_lowest_pitches_a():
    notes = [
        Note(pitch=75, start=0.0, end=1.0, velocity=127),
        Note(pitch=9, start=1.0, end=2.0, velocity=127),
        Note(pitch=127, start=1.2, end=1.6, velocity=127)
    ]
    expected_highest = 127
    expected_lowest = 9
    got_highest, got_lowest = get_highest_lowest_pitches(notes)
    assert got_highest == expected_highest
    assert got_lowest == expected_lowest


def test_get_pitch_range():
    notes = [
        Note(pitch=75, start=0.0, end=1.0, velocity=127),
        Note(pitch=127, start=1.2, end=1.6, velocity=127),
        Note(pitch=17, start=1.9, end=2.0, velocity=127),
        Note(pitch=127, start=2.0, end=3.0, velocity=127)
    ]
    expected = 110
    got = get_pitch_range(notes)
    assert expected == got


def test_get_pitch_classes():
    notes = [
        Note(pitch=75, start=0.0, end=1.0, velocity=127),
        Note(pitch=9, start=1.0, end=2.0, velocity=127),
        Note(pitch=127, start=1.2, end=1.6, velocity=127),
        Note(pitch=127, start=1.9, end=2.0, velocity=127),
        Note(pitch=127, start=2.0, end=3.0, velocity=127)
    ]
    expected = {
        "127": 3,
        "9": 1,
        "75": 1
    }
    got = get_pitch_classes(notes)
    assert expected.keys() == got.keys()
    for k in expected.keys():
        assert expected[k] == got[k]


def test_get_note_density():
    notes = [
        Note(pitch=75, start=0.0, end=1.0, velocity=127),
        Note(pitch=9, start=1.0, end=2.0, velocity=127),
        Note(pitch=127, start=1.2, end=1.6, velocity=127),
        Note(pitch=127, start=1.9, end=2.0, velocity=127),
        Note(pitch=127, start=2.0, end=3.0, velocity=127)
    ]
    expected = 5
    got = get_note_density(notes)
    assert expected == got

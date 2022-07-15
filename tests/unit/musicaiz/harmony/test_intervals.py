import pytest


# Our modules
from musicaiz.structure import (
    NoteTiming,
)
from musicaiz.harmony import (
    IntervalClass,
    IntervalQuality,
    IntervalSemitones,
    Interval,
)


# ===============IntervalClass class Tests=============
# =====================================================
def test_IntervalClass_quality():
    expected = "2"
    got = IntervalClass.SECOND.quality
    assert expected == got


def test_IntervalClass_get_all_interval_classes():
    expected = ["1", "2", "3", "4", "5", "6", "7", "8"]
    got = IntervalClass.get_all_interval_classes()
    assert expected == got


# ===============IntervalQuality class Tests===========
# =====================================================
def test_IntervalQuality_a():
    got = IntervalQuality.MINOR
    assert got.large == "minor"
    assert got.medium == "min"
    assert got.contracted == "m"
    assert got.symbol == "-"


def test_IntervalQuality_get_all_interval_qualities():
    got = IntervalQuality.get_all_interval_qualities()
    assert len(got) != 0


# ===============IntervalSemitones class Tests=========
# =====================================================
def test_IntervalSemitones_a():
    got = IntervalSemitones.SEVENTH_MAJOR
    assert got.semitones == 11
    assert got.large == "7major"
    assert got.medium == "7maj"
    assert got.contracted == "7M"


def test_IntervalSemitones_all_interval_names():
    got = IntervalSemitones.all_interval_names()
    assert len(got) != 1


def test_IntervalSemitones_check_interval_exists_a():
    interval = "7m"
    got = IntervalSemitones.check_interval_exists(interval)
    assert got is True


def test_IntervalSemitones_check_interval_exists_b():
    interval = "0-"
    got = IntervalSemitones.check_interval_exists(interval)
    assert got is False


def test_IntervalSemitones_get_interval_from_semitones():
    semitones = 2
    expected = [
        IntervalSemitones.UNISON_DOUBLY_AUGMENTED,
        IntervalSemitones.SECOND_MAJOR,
        IntervalSemitones.THIRD_DIMINISHED
    ]
    got = IntervalSemitones.get_interval_from_semitones(semitones)
    assert set(expected) == set(got)


def test_IntervalSemitones_get_qualities_from_semitones_a():
    # Test case: Valid semitones
    semitones = 2
    expected = [
        IntervalQuality.AUGMENTED,
        IntervalQuality.MAJOR,
        IntervalQuality.DIMINISHED
    ]
    got = IntervalSemitones.get_qualities_from_semitones(semitones)
    assert set(expected) == set(got)


def test_IntervalSemitones_get_qualities_from_semitones_b():
    # Test case: Invalid semitones
    semitones = 20
    with pytest.raises(ValueError):
        IntervalSemitones.get_qualities_from_semitones(semitones)


def test_IntervalSemitones_get_classes_from_semitones():
    semitones = 2
    expected = [
        IntervalClass.UNISON,
        IntervalClass.SECOND,
        IntervalClass.THIRD
    ]
    got = IntervalSemitones.get_classes_from_semitones(semitones)
    assert set(expected) == set(got)


def test_IntervalSemitones_get_class_from_quality_semitones_a():
    # Test case: No interval found (bad combination of semitones and quality)
    semitones = 2
    quality = "m"
    with pytest.raises(ValueError):
        IntervalSemitones.get_class_from_quality_semitones(quality, semitones)


def test_IntervalSemitones_get_class_from_quality_semitones_b():
    # Test case: interval found
    semitones = 2
    quality = "M"
    expected = IntervalClass.SECOND
    got = IntervalSemitones.get_class_from_quality_semitones(quality, semitones)
    assert expected == got


def test_IntervalSemitones_get_quality_from_class_semitones_a():
    # Test case: No interval found (bad combination of semitones and quality)
    semitones = 2
    interval_class = "5"
    with pytest.raises(ValueError):
        IntervalSemitones.get_quality_from_class_semitones(interval_class, semitones)


def test_IntervalSemitones_get_quality_from_class_semitones_b():
    # Test case: interval found
    semitones = 2
    interval_class = "2"
    expected = IntervalQuality.MAJOR
    got = IntervalSemitones.get_quality_from_class_semitones(interval_class, semitones)
    assert expected == got


def test_IntervalSemitones_get_interval_from_name_a():
    # Test case: interval found
    name = "7M"
    expected = IntervalSemitones.SEVENTH_MAJOR
    got = IntervalSemitones.get_interval_from_name(name)
    assert expected == got


def test_IntervalSemitones_get_interval_from_name_b():
    # Test case: interval not found
    name = "9M"
    with pytest.raises(ValueError):
        IntervalSemitones.get_interval_from_name(name)


# ===============Interval class Tests==================
# =====================================================
def test_Interval_a():
    # Test case: Initializing with interval name (simple interval)
    interval = "6M"
    got = Interval(interval)
    assert got.interval_class == IntervalClass.SIXTH
    assert got.quality == IntervalQuality.MAJOR
    assert got.semitones == 9


def test_Interval_b():
    # Test case: Not initializing the interval name
    got = Interval()
    assert got.interval_class is None
    assert got.quality is None
    assert got.interval is None
    assert got.semitones is None


def test_Interval_transpose_note_a():
    # Test case: Valid interval class and note as str
    note = "C#1"
    interval = "7M"
    got = Interval(interval)
    note_transposed = got.transpose_note(note)
    assert note_transposed.pitch == 36
    assert note_transposed.note_name == "C"
    assert note_transposed.octave == "2"


def test_Interval_transpose_note_b():
    # Test case: Valid interval class and note as str
    pitch = 49
    start_sec = 1.0
    end_sec = 2.0
    note = NoteTiming(pitch, start_sec, end_sec)
    interval = "7M"
    got = Interval(interval)
    note_transposed = got.transpose_note(note)
    assert note_transposed.pitch == 60
    assert note_transposed.note_name == "C"
    assert note_transposed.octave == "4"


def test_Interval_get_interval_a():
    # Test case: notes are the note names as str and notes in same 8ve
    note1 = "C#0"  # C# or Db in MIDI
    note2 = "D#0"  # D# or Eb in MIDI
    expected = [
        ("C#-D#", IntervalSemitones.SECOND_MAJOR),
        ("C#-Eb", IntervalSemitones.THIRD_DIMINISHED),
        ("Db-D#", IntervalSemitones.UNISON_DOUBLY_AUGMENTED),
        ("Db-Eb", IntervalSemitones.SECOND_MAJOR),
    ]
    got = Interval.get_possible_intervals(note1, note2)
    assert set(expected) == set(got)


def test_Interval_get_interval_b():
    # Test case: notes are the note pitches as int
    note1 = 124  # E or Fb
    note2 = 42  # F# or Gb
    expected = [
        ("E-F#", IntervalSemitones.SECOND_MAJOR),
        ("E-Gb", IntervalSemitones.THIRD_DIMINISHED),
        ("Fb-F#", IntervalSemitones.UNISON_DOUBLY_AUGMENTED),
        ("Fb-Gb", IntervalSemitones.SECOND_MAJOR),
    ]
    got = Interval.get_possible_intervals(note1, note2)
    assert set(expected) == set(got)


def test_Interval_get_interval_c():
    note1 = 126  # F# or Gb
    note2 = 42  # F# or Gb
    expected = [
        ("F#-F#", IntervalSemitones.OCTAVE_PERFECT),
        ("F#-Gb", IntervalSemitones.SECOND_DIMINISHED),
        ("Gb-F#", IntervalSemitones.SEVENTH_AUGMENTED),
        ("Gb-Gb", IntervalSemitones.OCTAVE_PERFECT),
    ]
    got = Interval.get_possible_intervals(note1, note2)
    assert set(expected) == set(got)


def test_Interval_get_interval_d():
    note1 = 26  # D
    note2 = 68  # G# or Ab
    expected = [
        ("D-G#", IntervalSemitones.FOURTH_AUGMENTED),
        ("D-Ab", IntervalSemitones.FIFTH_DIMINISHED),
    ]
    got = Interval.get_possible_intervals(note1, note2)
    assert set(expected) == set(got)


def test_Interval_get_interval_e():
    # Edge case: same interval as before but inverted (not descendent)
    note1 = 68  # G# or Ab
    note2 = 26  # D
    expected = [
        ("G#-D", IntervalSemitones.FIFTH_DIMINISHED),
        ("Ab-D", IntervalSemitones.FOURTH_AUGMENTED),
    ]
    got = Interval.get_possible_intervals(note1, note2)
    assert set(expected) == set(got)


def test_Interval_get_interval_f():
    note1 = 67  # G
    note2 = 64  # E or Fb
    expected = [
        ("G-E", IntervalSemitones.SIXTH_MAJOR),
        ("G-Fb", IntervalSemitones.SEVENTH_DIMINISHED),
    ]
    got = Interval.get_possible_intervals(note1, note2)
    assert set(expected) == set(got)


def test_Interval_get_interval_g():
    note1 = 24  # C or B#
    note2 = 25  # C# or Db
    expected = [
        ("C-C#", IntervalSemitones.UNISON_AUGMENTED),  # C to C#
        ("C-Db", IntervalSemitones.SECOND_MINOR),  # C to Db
        ("B#-C#", IntervalSemitones.SECOND_MINOR),
        ("B#-Db", IntervalSemitones.SECOND_MINOR)
    ]
    got = Interval.get_possible_intervals(note1, note2)
    assert set(expected) == set(got)

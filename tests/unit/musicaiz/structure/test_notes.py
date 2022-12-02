import pytest
import math


# Our modules
from musicaiz.structure import (
    AccidentalsNames,
    NoteClassNames,
    NoteClassBase,
    NoteValue,
    NoteTiming,
    Note
)


# ===============Accidentals class Tests===============
# =====================================================
def test_AccidentalsNames_contracted():
    expected = "#"
    got = AccidentalsNames.SHARP.contracted
    assert expected == got


def test_AccidentalsNames_expanded():
    expected = "flat"
    got = AccidentalsNames.FLAT.expanded
    assert expected == got


def test_AccidentalsNames_spanish():
    expected = "becuadro"
    got = AccidentalsNames.NATURAL.spanish
    assert expected == got


# ===============NoteClassNames class Tests============
# =====================================================
def test_NoteClassNames_spanish_contracted():
    expected = "Sol b"
    got = NoteClassNames.G_FLAT.spanish_contracted
    assert expected == got


def test_NoteClassNames_contracted():
    expected = "G"
    got = NoteClassNames.G.contracted
    assert expected == got


def test_NoteClassNames_spanish_expanded():
    expected = "Do sostenido"
    got = NoteClassNames.C_SHARP.spanish_expanded
    assert expected == got


def test_NoteClassNames_expanded():
    expected = "C"
    got = NoteClassNames.C.expanded
    assert expected == got


def test_NoteClassNames_get_all_names():
    got = NoteClassNames.get_all_names()
    assert len(got) != 0


def test_NoteClassNames_check_note_name_exists_a():
    got = NoteClassNames.check_note_name_exists("C")
    assert got is True


def test_NoteClassNames_check_note_name_exists_b():
    got = NoteClassNames.check_note_name_exists("Cs")
    assert got is False


def test_NoteClassNames_get_note_with_name():
    expected = NoteClassNames.C
    got = NoteClassNames.get_note_with_name("C")
    assert expected == got


# ===============NoteClassBase class Tests=============
# =====================================================
def test_NoteClassBase_natural_scale_index():
    expected = 4
    got = NoteClassBase.G.natural_scale_index
    assert expected == got


def test_NoteClassBase_chromatic_scale_index():
    expected = 11
    got = NoteClassBase.B.chromatic_scale_index
    assert expected == got


def test_NoteClassBase_get_note_from_chromatic_idx():
    semitones = 10
    expected = [NoteClassBase.A_SHARP, NoteClassBase.B_FLAT]
    got = NoteClassBase._get_note_from_chromatic_idx(semitones)
    assert expected == got


def test_NoteClassBase_add_flat():
    note = NoteClassBase.C_SHARP
    expected = NoteClassBase.C
    got = note.add_flat
    assert expected == got


def test_NoteClassBase_add_sharp_a():
    note = NoteClassBase.A
    expected = NoteClassBase.A_SHARP
    got = note.add_sharp
    assert expected == got


def test_NoteClassBase_add_sharp_b():
    note = NoteClassBase.B
    expected = NoteClassBase.C
    got = note.add_sharp
    assert expected == got


def test_NoteClassBase_all_chromatic_scale_indexes():
    expected = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 11]
    got = NoteClassBase.all_chromatic_scale_indexes()
    assert expected == got


def test_NoteClassBase_all_natural_scale_indexes():
    expected = [0, None, None, None, 1, None, None, 2, None, None, 3, None, None, 4, None, None, 5, None, None, 6, None]
    got = NoteClassBase.all_natural_scale_indexes()
    assert expected == got


def test_NoteClassBase_get_note_with_name():
    expected = NoteClassBase.G
    got = NoteClassBase.get_note_with_name("G")
    assert expected == got


# ===============NoteValue class Tests=================
# =====================================================
def test_NoteValue_split_pitch_name_a():
    # Test case: Note without `#` or `b` and positive octave
    pitch_name = "C1"
    expected_note_name = "C"
    expected_octave = "1"
    got_note_name, got_octave = NoteValue.split_pitch_name(pitch_name)

    assert expected_note_name == got_note_name
    assert expected_octave == got_octave


def test_NoteValue_split_pitch_name_b():
    # Test case: Note with `#` or `b` and positive octave
    pitch_name = "C#1"
    expected_note_name = "C#"
    expected_octave = "1"
    got_note_name, got_octave = NoteValue.split_pitch_name(pitch_name)

    assert expected_note_name == got_note_name
    assert expected_octave == got_octave


def test_NoteValue_split_pitch_name_c():
    # Test case: Note with `#` or `b` and negative octave
    pitch_name = "Cb-1"
    expected_note_name = "Cb"
    expected_octave = "-1"
    got_note_name, got_octave = NoteValue.split_pitch_name(pitch_name)

    assert expected_note_name == got_note_name
    assert expected_octave == got_octave


def test_NoteValue_a():
    # Test case: Initialize with a valid pitch value
    pitch_value = 15
    got = NoteValue(pitch_value)

    assert got.note == NoteClassBase.D_SHARP
    assert got.pitch_name == "D#0"
    assert got.octave == "0"
    assert got.note_name == "D#"


def test_NoteValue_b():
    # Test case: Initialize with an invalid pitch value
    pitch_value = 150
    with pytest.raises(ValueError):
        NoteValue(pitch_value)


def test_NoteValue_c():
    # Test case: Initialize with a valid pitch name
    pitch_name = "G2"
    got = NoteValue(pitch_name)

    assert got.pitch == 43
    assert got.octave == "2"
    assert got.note_name == "G"


def test_NoteValue_d():
    # Test case: Initialize with an invalid pitch name
    pitch_name = "J2"
    with pytest.raises(ValueError):
        NoteValue(pitch_name)


# ===============NoteTiming class Tests================
# =====================================================
def test_NoteTiming_a():
    # Test case: Initialize with seconds (floats)
    resolution = 960
    bpm = 120
    pitch = 12
    start_sec = 1.0
    end_sec = 2.0

    got = NoteTiming(pitch, start_sec, end_sec, bpm, resolution)

    assert got.start_sec == 1
    assert got.end_sec == 2
    assert got.start_ticks == 1920
    assert got.end_ticks == 3840


def test_NoteTiming_b():
    # Test case: Initialize with ticks (floats)
    resolution = 960
    bpm = 120
    pitch = 12
    start_ticks = 384
    end_ticks = 576

    got = NoteTiming(pitch, start_ticks, end_ticks, bpm, resolution)

    assert got.start_sec == 0.2
    assert got.end_sec == 0.3
    assert got.start_ticks == 384
    assert got.end_ticks == 576


def test_NoteTiming_c():
    # Test case: Wrong initialization, end < start
    pitch = 12
    start_sec = 6.0
    end_sec = 1.0

    with pytest.raises(ValueError):
        NoteTiming(pitch, start_sec, end_sec)


def test_NoteTiming_d():
    # Test case: Wrong initialization, end or start < 0
    pitch = 12
    start_sec = 6.0
    end_sec = -9.0

    with pytest.raises(ValueError):
        NoteTiming(pitch, start_sec, end_sec)


def test_NoteTiming_e():
    # Test case: Wrong initialization, writing seconds as ints
    # TODO: any error is raised but instance attributes are wrong
    pitch = 12
    start_sec = 1
    end_sec = 2

    got = NoteTiming(pitch, start_sec, end_sec)

    assert got.start_sec != 1
    assert got.end_sec != 2
    assert got.start_ticks != 3840
    assert got.end_ticks != 5760


# ===============Note class Tests======================
# =====================================================
def test_Note_a():
    # Test case: Initialize with note on and note off and pitch value
    resolution = 960
    bpm = 120
    pitch = 12
    velocity = 120
    note_on = 0.0
    note_off = 1.0
    ligated = True

    got_note = Note(pitch, note_on, note_off, velocity, ligated, bpm, resolution=resolution)

    assert got_note.start_ticks == 0
    assert got_note.end_ticks == 1920
    assert got_note.octave == "0"
    assert got_note.pitch_name == "C0"
    assert got_note.note_name == "C"


def test_Note_b():
    # Test case: Initialize with note on and note off and pitch name
    # (other note on and note off values)
    resolution = 960
    bpm = 120
    pitch_name = "D#5"
    velocity = 120
    note_on = 10.0
    note_off = 20.0
    ligated = True

    got_note = Note(pitch_name, note_on, note_off, velocity, ligated, bpm, resolution=resolution)

    assert got_note.start_ticks == 19200
    assert got_note.end_ticks == 38400
    assert got_note.octave == "5"
    assert got_note.pitch == 75
    assert got_note.note_name == "D#"


def test_Note_c():
    # Test case: Initialize with start ticks and end ticks
    resolution = 960
    bpm = 120
    pitch_name = "G#7"
    velocity = 8
    start_ticks = 10
    end_ticks = 20
    ligated = True

    got_note = Note(pitch_name, start_ticks, end_ticks, velocity, ligated, bpm, resolution=resolution)

    assert math.isclose(round(got_note.start_sec, 3), 0.005)
    assert math.isclose(round(got_note.end_sec, 3), 0.01)
    assert got_note.octave == "7"
    assert got_note.pitch == 104
    assert got_note.note_name == "G#"


def test_Note_d():
    # Test case: Initialize with invalid pitch name
    pitch_name = "G#97"
    velocity = 8
    start_ticks = 10
    end_ticks = 20

    with pytest.raises(ValueError):
        Note(pitch_name, start_ticks, end_ticks, velocity)

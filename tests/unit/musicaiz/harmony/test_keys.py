import pytest


from musicaiz.structure import NoteClassBase
from musicaiz.harmony import (
    DegreesRoman,
    AllChords,
    MajorTriadDegrees,
    Tonality,
    ModeConstructors,
)


def test_MajorTriadDegrees_a():
    expected_short = "I"
    expected_large = "Imajor"
    expected_chord = AllChords.MAJOR_TRIAD
    got_short = MajorTriadDegrees.FIRST.contracted_name
    got_large = MajorTriadDegrees.FIRST.large_name
    got_chord = MajorTriadDegrees.FIRST.chord
    assert got_short == expected_short
    assert got_large == expected_large
    assert got_chord == expected_chord


def test_Tonality_altered_notes():
    expected = [
        NoteClassBase.B_FLAT,
        NoteClassBase.E_FLAT,
        NoteClassBase.A_FLAT,
    ]
    got = Tonality.E_FLAT_MAJOR.altered_notes
    assert set(got) == set(expected)


def test_Tonality_scale_notes_a():
    # Lydian will have the same accidentals as major
    # but with the 4th (A) altered with a sharp
    scale = "LYDIAN"
    expected = [
        NoteClassBase.E_FLAT,
        NoteClassBase.F,
        NoteClassBase.G,
        NoteClassBase.A,
        NoteClassBase.B_FLAT,
        NoteClassBase.C,
        NoteClassBase.D,
    ]
    tonality = Tonality.E_FLAT_MAJOR
    got = tonality.scale_notes(scale)
    for n in range(len(expected)):
        assert got[n] == expected[n]


def test_Tonality_scale_notes_b():
    scale = "MAJOR"
    expected = [
        NoteClassBase.C,
        NoteClassBase.D,
        NoteClassBase.E,
        NoteClassBase.F,
        NoteClassBase.G,
        NoteClassBase.A,
        NoteClassBase.B,
    ]
    tonality = Tonality.C_MAJOR
    got = tonality.scale_notes(scale)
    for n in range(len(expected)):
        assert got[n] == expected[n]


def test_Tonality_scale_notes_c():
    # Test case: Minor mode, dorian scale.
    # It inherits accidentals from major mode
    scale = "DORIAN"
    expected = [
        NoteClassBase.A,
        NoteClassBase.B,
        NoteClassBase.C,
        NoteClassBase.D,
        NoteClassBase.E,
        NoteClassBase.F_SHARP,
        NoteClassBase.G,
    ]
    tonality = Tonality.A_MINOR
    got = tonality.scale_notes(scale)
    for n in range(len(expected)):
        assert got[n] == expected[n]


def test_Tonality_scale_notes_d():
    # Test case: Minor mode, dorian scale.
    # It inherits accidentals from major mode
    scale = "DORIAN"
    expected = [
        NoteClassBase.B,
        NoteClassBase.C_SHARP,
        NoteClassBase.D,
        NoteClassBase.E,
        NoteClassBase.F_SHARP,
        NoteClassBase.G_SHARP,
        NoteClassBase.A,
    ]
    tonality = Tonality.B_MINOR
    got = tonality.scale_notes(scale)
    for n in range(len(expected)):
        assert got[n] == expected[n]


def test_Tonality_notes_a():
    expected = [
        NoteClassBase.E_FLAT,
        NoteClassBase.F,
        NoteClassBase.G,
        NoteClassBase.A_FLAT,
        NoteClassBase.B_FLAT,
        NoteClassBase.C,
        NoteClassBase.D,
    ]
    got = Tonality.E_FLAT_MAJOR.notes
    for n in range(len(expected)):
        assert got[n] == expected[n]


def test_Tonality_notes_b():
    # Test case: Minor mode, G# minor scale.
    # It inherits accidentals from minor mode
    expected = [
        NoteClassBase.G_SHARP,
        NoteClassBase.A_SHARP,
        NoteClassBase.B,
        NoteClassBase.C_SHARP,
        NoteClassBase.D_SHARP,
        NoteClassBase.E,
        NoteClassBase.F_SHARP,
    ]
    got = Tonality.G_SHARP_MINOR.notes
    for n in range(len(expected)):
        assert got[n] == expected[n]


def test_Tonality_scale_names():
    expected = [
        ModeConstructors.MAJOR,
        ModeConstructors.LYDIAN,
        ModeConstructors.MIXOLYDIAN,
    ]
    got = Tonality.E_FLAT_MAJOR.all_scales
    assert set(got) == set(expected)


def test_Tonality_get_chord_from_degree_b():
    # Test case: scale selected, major mode, triad chord, I degree
    tonality = "E_MAJOR"
    degree = "IV"
    scale = "LYDIAN"
    expected = (
        NoteClassBase.A_SHARP,
        AllChords.DIMINISHED_TRIAD
    )
    got = Tonality.get_chord_from_degree(tonality, degree, scale)
    assert set(got) == set(expected)


def test_Tonality_get_all_chords_from_scale_a():
    # Test case: Do not specify the scale, it'll work with the default
    tonality = "E_MAJOR"
    chord_type = "triad"
    expected = [
        (NoteClassBase.E, AllChords.MAJOR_TRIAD),
        (NoteClassBase.F_SHARP, AllChords.MINOR_TRIAD),
        (NoteClassBase.G_SHARP, AllChords.MINOR_TRIAD),
        (NoteClassBase.A, AllChords.MAJOR_TRIAD),
        (NoteClassBase.B, AllChords.MAJOR_TRIAD),
        (NoteClassBase.C_SHARP, AllChords.MINOR_TRIAD),
        (NoteClassBase.D_SHARP, AllChords.DIMINISHED_TRIAD),
    ]
    got = Tonality.get_all_chords_from_scale(
        tonality=tonality,
        chord_type=chord_type
    )
    assert set(got) == set(expected)


def test_Tonality_get_all_chords_from_scale_b():
    # Test case: Specify the scale
    tonality = "E_MAJOR"
    chord_type = "triad"
    scale = "LYDIAN"
    expected = [
        (NoteClassBase.E, AllChords.MAJOR_TRIAD),
        (NoteClassBase.F_SHARP, AllChords.MAJOR_TRIAD),
        (NoteClassBase.G_SHARP, AllChords.MINOR_TRIAD),
        (NoteClassBase.A_SHARP, AllChords.DIMINISHED_TRIAD),
        (NoteClassBase.B, AllChords.MAJOR_TRIAD),
        (NoteClassBase.C_SHARP, AllChords.MINOR_TRIAD),
        (NoteClassBase.D_SHARP, AllChords.MINOR_TRIAD),
    ]
    got = Tonality.get_all_chords_from_scale(
        tonality=tonality,
        scale=scale,
        chord_type=chord_type
    )
    assert set(got) == set(expected)


def test_Tonality_get_all_chords_from_scale_c():
    # Test case: Specify the scale
    tonality = "A_MINOR"
    chord_type = "triad"
    scale = "DORIAN"
    expected = [
        (NoteClassBase.A, AllChords.MINOR_TRIAD),
        (NoteClassBase.B, AllChords.MINOR_TRIAD),
        (NoteClassBase.C, AllChords.MAJOR_TRIAD),
        (NoteClassBase.D, AllChords.MAJOR_TRIAD),
        (NoteClassBase.E, AllChords.MINOR_TRIAD),
        (NoteClassBase.F_SHARP, AllChords.DIMINISHED_TRIAD),
        (NoteClassBase.G, AllChords.MAJOR_TRIAD),
    ]
    got = Tonality.get_all_chords_from_scale(
        tonality=tonality,
        scale=scale,
        chord_type=chord_type
    )
    assert set(got) == set(expected)


def test_Tonality_get_chord_notes_from_degree_a():
    tonality = "C_MAJOR"
    degree = "I"
    chord_type = "triad"
    expected = [
        NoteClassBase.C,
        NoteClassBase.E,
        NoteClassBase.G,
    ]
    got = Tonality.get_chord_notes_from_degree(tonality, degree, None, chord_type)
    assert set(got) == set(expected)


def test_Tonality_get_scales_degrees_from_chord_a():
    # Test case: input is C Major chord
    chord = (NoteClassBase.C, AllChords.MAJOR_TRIAD)
    expected = [
        (DegreesRoman.FIRST, Tonality.C_MAJOR, ModeConstructors.MAJOR),
        (DegreesRoman.FIRST, Tonality.C_MAJOR, ModeConstructors.LYDIAN),
        (DegreesRoman.FIRST, Tonality.C_MAJOR, ModeConstructors.MIXOLYDIAN),
        (DegreesRoman.THIRD, Tonality.A_MINOR, ModeConstructors.NATURAL),
        (DegreesRoman.THIRD, Tonality.A_MINOR, ModeConstructors.DORIAN),
        (DegreesRoman.THIRD, Tonality.A_MINOR, ModeConstructors.PHRYGIAN),
        (DegreesRoman.FOURTH, Tonality.G_MAJOR, ModeConstructors.MAJOR),
        (DegreesRoman.FOURTH, Tonality.G_MAJOR, ModeConstructors.MIXOLYDIAN),
        (DegreesRoman.SIXTH, Tonality.E_MINOR, ModeConstructors.NATURAL),
        (DegreesRoman.SIXTH, Tonality.E_MINOR, ModeConstructors.HARMONIC),
        (DegreesRoman.SIXTH, Tonality.E_MINOR, ModeConstructors.PHRYGIAN),
        (DegreesRoman.SIXTH, Tonality.E_MINOR, ModeConstructors.LOCRIAN),
        (DegreesRoman.FIFTH, Tonality.F_MAJOR, ModeConstructors.MAJOR),
        (DegreesRoman.FIFTH, Tonality.F_MAJOR, ModeConstructors.LYDIAN),
        (DegreesRoman.SEVENTH, Tonality.D_MINOR, ModeConstructors.NATURAL),
        (DegreesRoman.SEVENTH, Tonality.D_MINOR, ModeConstructors.DORIAN),
        (DegreesRoman.SEVENTH, Tonality.D_MAJOR, ModeConstructors.MIXOLYDIAN),
        (DegreesRoman.SECOND, Tonality.B_MINOR, ModeConstructors.PHRYGIAN),
        (DegreesRoman.SECOND, Tonality.B_MINOR, ModeConstructors.LOCRIAN),
        (DegreesRoman.SECOND, Tonality.B_FLAT_MAJOR, ModeConstructors.LYDIAN),
        (DegreesRoman.FOURTH, Tonality.G_MINOR, ModeConstructors.MELODIC),
        (DegreesRoman.FOURTH, Tonality.G_MINOR, ModeConstructors.DORIAN),
        (DegreesRoman.FIFTH, Tonality.F_SHARP_MINOR, ModeConstructors.LOCRIAN),
        (DegreesRoman.FIFTH, Tonality.F_MINOR, ModeConstructors.HARMONIC),
        (DegreesRoman.FIFTH, Tonality.F_MINOR, ModeConstructors.MELODIC)
    ]
    got = Tonality.get_scales_degrees_from_chord(chord)
    assert set(got) == set(expected)


def test_Tonality_get_scales_degrees_from_chord_b():
    # Test case: input is C minor chord.
    # In this test we didn't write all the possible chords in the expected values
    chord = (NoteClassBase.C, AllChords.MINOR_TRIAD)
    expected = [
        (DegreesRoman.FIRST, Tonality.C_MINOR, ModeConstructors.NATURAL),
        (DegreesRoman.THIRD, Tonality.A_FLAT_MAJOR, ModeConstructors.MAJOR),
        (DegreesRoman.FOURTH, Tonality.G_MINOR, ModeConstructors.NATURAL),
        (DegreesRoman.SIXTH, Tonality.E_FLAT_MAJOR, ModeConstructors.MAJOR),
        (DegreesRoman.FIFTH, Tonality.F_MINOR, ModeConstructors.NATURAL),
        (DegreesRoman.SECOND, Tonality.B_FLAT_MAJOR, ModeConstructors.MAJOR),
    ]
    got = Tonality.get_scales_degrees_from_chord(chord)
    assert set(expected).issubset(set(got))


def test_Tonality_get_modes_degrees_from_chord_a():
    # Test case: input is C Major chord
    chord = (NoteClassBase.C, AllChords.MAJOR_TRIAD)
    expected = [
        (DegreesRoman.FIRST, Tonality.C_MAJOR),
        (DegreesRoman.THIRD, Tonality.A_MINOR),
        (DegreesRoman.FOURTH, Tonality.G_MAJOR),
        (DegreesRoman.SIXTH, Tonality.E_MINOR),
        (DegreesRoman.FIFTH, Tonality.F_MAJOR),
        (DegreesRoman.SEVENTH, Tonality.D_MINOR),
        (DegreesRoman.SECOND, Tonality.B_MINOR),
        (DegreesRoman.FOURTH, Tonality.G_MINOR),
        (DegreesRoman.FIFTH, Tonality.F_SHARP_MINOR),
    ]
    got = Tonality.get_modes_degrees_from_chord(chord)
    assert set(got) == set(expected)

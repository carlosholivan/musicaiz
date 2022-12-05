# Our modules
from musicaiz.features import (
    get_chord_type_from_note_seq,
    get_intervals_note_seq,
    predict_chords,
    predict_scales_degrees,
    predict_possible_progressions,
    predict_progression,
    _all_note_seq_permutations,
    _delete_repeated_note_names,
    _order_note_seq_by_chromatic_idx,
    get_harmonic_density,
)
from musicaiz.structure import (
    Note,
    NoteClassBase
)
from musicaiz.harmony import (
    ChordType,
    AllChords,
    IntervalSemitones,
    DegreesRoman,
    Tonality,
    ModeConstructors,
)


def test_order_note_seq_by_chromatic_idx_a():
    # We are not sorting by pitch value but for note name index
    notes = [
        Note(pitch=80, start=0.0, end=1.0, velocity=127),
        Note(pitch=78, start=1.0, end=2.0, velocity=127),
        Note(pitch=89, start=1.1, end=1.3, velocity=127),
    ]
    expected = [
        Note(pitch=89, start=1.1, end=1.3, velocity=127),
        Note(pitch=78, start=1.0, end=2.0, velocity=127),
        Note(pitch=80, start=0.0, end=1.0, velocity=127),
    ]
    got = _order_note_seq_by_chromatic_idx(notes)
    for idx in range(len(expected)):
        assert got[idx].note_name == expected[idx].note_name


def test_get_chord_type_from_note_seq_a():
    notes = [
        Note(pitch=80, start=0.0, end=1.0, velocity=127),
        Note(pitch=78, start=1.0, end=2.0, velocity=127),
        Note(pitch=89, start=1.1, end=1.3, velocity=127),
    ]
    expected = ChordType.TRIAD
    got = get_chord_type_from_note_seq(notes)
    assert got == expected


def test_get_intervals_note_seq_a():
    # Measures intervals between 1st note and the others
    # (our chords definitions are intervals from the root note)
    notes = [
        Note(pitch=78, start=1.0, end=2.0, velocity=127),  # F#
        Note(pitch=80, start=0.0, end=1.0, velocity=127),  # G#
        Note(pitch=84, start=1.1, end=1.3, velocity=127),  # C
    ]
    expected = [
        [
            ("F#-G#", IntervalSemitones.SECOND_MAJOR),  # 1st to 2nd note
            ("Gb-G#", IntervalSemitones.UNISON_DOUBLY_AUGMENTED),
            ("F#-Ab", IntervalSemitones.THIRD_DIMINISHED),
            ("Gb-Ab", IntervalSemitones.SECOND_MAJOR)
        ],
        [
            ("F#-C", IntervalSemitones.FIFTH_DIMINISHED),  # 1st to 3rd note
            ("Gb-C", IntervalSemitones.FOURTH_AUGMENTED),
            ("F#-B#", IntervalSemitones.FOURTH_AUGMENTED),
            ("Gb-B#", IntervalSemitones.THIRD_DOUBLY_AUGMENTED)
        ],
    ]
    got = get_intervals_note_seq(notes)

    for i in range(len(expected)):
        assert set(got[i]) == set(expected[i])


def test_all_note_seq_permutations():
    # Test case: diminished seventh (G# minor)
    # input note seq is not sorted and notes are in different octaves
    notes = [
        Note(pitch=26, start=1.0, end=2.0, velocity=127),  # D
        Note(pitch=65, start=0.0, end=1.0, velocity=127),  # F
    ]
    expected = [
        [
            Note(pitch=26, start=1.0, end=2.0, velocity=127),
            Note(pitch=65, start=0.0, end=1.0, velocity=127),
        ],
        [
            Note(pitch=65, start=1.0, end=2.0, velocity=127),
            Note(pitch=26, start=0.0, end=1.0, velocity=127),
        ],
    ]
    got = _all_note_seq_permutations(notes)
    for i in range(len(expected)):
        for j in range(len(expected[i])):
            assert got[i][j].pitch == expected[i][j].pitch


def test_delete_repeated_note_names():
    # Test case: Same note names but different 8ves
    notes = [
        Note(pitch=67, start=1.0, end=2.0, velocity=127),
        Note(pitch=67, start=0.0, end=1.0, velocity=127),
        Note(pitch=60, start=1.1, end=1.3, velocity=127),
        Note(pitch=72, start=1.1, end=1.3, velocity=127),  # 8ve up
    ]
    expected = [
        Note(pitch=67, start=0.0, end=1.0, velocity=127),
        Note(pitch=60, start=1.1, end=1.3, velocity=127),
    ]
    got = _delete_repeated_note_names(notes)
    assert len(got) == len(expected)
    for i in range(len(expected)):
        assert got[i].pitch == expected[i].pitch


def test_predict_chords_a():
    # Test case: simple triad (C Major)
    # input note seq is not sorted
    notes = [
        Note(pitch=67, start=1.0, end=2.0, velocity=127),
        Note(pitch=64, start=0.0, end=1.0, velocity=127),
        Note(pitch=60, start=1.1, end=1.3, velocity=127),
    ]
    expected = [
        (NoteClassBase.C, AllChords.MAJOR_TRIAD),
        (NoteClassBase.B_SHARP, AllChords.MAJOR_TRIAD),  # enharmonic chord of C Maj
    ]
    got = predict_chords(notes)
    assert set(expected) == set(got)


def test_predict_chords_b():
    # Test case: diminished seventh (G# minor) or dim triad other root note
    # input note seq is not sorted and notes are in different octaves
    notes = [
        Note(pitch=26, start=1.0, end=2.0, velocity=127),  # D
        Note(pitch=65, start=0.0, end=1.0, velocity=127),  # F or E#
        Note(pitch=68, start=1.1, end=1.3, velocity=127),  # G# or Ab
        Note(pitch=47, start=1.1, end=1.3, velocity=127),  # B or Cb
    ]
    expected = [
        (NoteClassBase.D, AllChords.DIMINISHED_TRIAD),  # D root note
        (NoteClassBase.F, AllChords.DIMINISHED_SEVENTH),  # F root note
        (NoteClassBase.G_SHARP, AllChords.DIMINISHED_SEVENTH),  # G# root note
        (NoteClassBase.B, AllChords.DIMINISHED_SEVENTH),  # B root note
        (NoteClassBase.E_SHARP, AllChords.DIMINISHED_SEVENTH),  # enharmonic of F
        (NoteClassBase.A_FLAT, AllChords.DIMINISHED_SEVENTH),  # enharmonic of G#
        (NoteClassBase.C_FLAT, AllChords.DIMINISHED_SEVENTH),  # enharmonic of B
    ]
    got = predict_chords(notes)
    assert set(expected) == set(got)


def test_predict_chords_c():
    # Test case: repeated notes
    notes = [
        Note(pitch=26, start=1.0, end=2.0, velocity=127),  # D
        Note(pitch=65, start=0.0, end=1.0, velocity=127),  # F or E#
        Note(pitch=68, start=1.1, end=1.3, velocity=127),  # G# or Ab
        Note(pitch=47, start=1.1, end=1.3, velocity=127),  # B or Cb
        Note(pitch=38, start=1.0, end=2.0, velocity=127),  # D
        Note(pitch=47, start=1.1, end=1.3, velocity=127),  # B or Cb
    ]
    expected = [
        (NoteClassBase.D, AllChords.DIMINISHED_TRIAD),  # D root note
        (NoteClassBase.F, AllChords.DIMINISHED_SEVENTH),  # F root note
        (NoteClassBase.G_SHARP, AllChords.DIMINISHED_SEVENTH),  # G# root note
        (NoteClassBase.B, AllChords.DIMINISHED_SEVENTH),  # B root note
        (NoteClassBase.E_SHARP, AllChords.DIMINISHED_SEVENTH),  # F enharmonic
        (NoteClassBase.A_FLAT, AllChords.DIMINISHED_SEVENTH),  # G# enharmonic
        (NoteClassBase.C_FLAT, AllChords.DIMINISHED_SEVENTH),  # B enharmonic
    ]
    got = predict_chords(notes)
    assert set(expected) == set(got)


def test_predict_chords_d():
    # Test case: 1 note not belowing to the chord (passing note).
    notes = [
        Note(pitch=24, start=1.0, end=2.0, velocity=127),  # C or B#
        Note(pitch=64, start=0.0, end=1.0, velocity=127),  # E or Fb
        Note(pitch=67, start=1.1, end=1.3, velocity=127),  # G
        Note(pitch=68, start=1.1, end=1.3, velocity=127),  # G# or Ab
    ]
    expected = [
        (NoteClassBase.C, AllChords.MAJOR_TRIAD),  # C-E-G
        (NoteClassBase.C, AllChords.AUGMENTED_TRIAD),  # C-E-G#
        (NoteClassBase.E, AllChords.AUGMENTED_TRIAD),  # E-G#-B#
        (NoteClassBase.G_SHARP, AllChords.AUGMENTED_TRIAD),  # G#-B#-E
        (NoteClassBase.B_SHARP, AllChords.MAJOR_TRIAD),  # C enharmonic
        (NoteClassBase.B_SHARP, AllChords.AUGMENTED_TRIAD),  # C enharmonic
        (NoteClassBase.F_FLAT, AllChords.AUGMENTED_TRIAD),  # E enharmonic
        (NoteClassBase.A_FLAT, AllChords.AUGMENTED_TRIAD),  # G# enharmonic
    ]
    got = predict_chords(notes)
    assert set(expected) == set(got)


def test_predict_chords_e():
    notes = [
        Note(pitch=75, start=1.0, end=2.0, velocity=127),  # Eb
        Note(pitch=72, start=0.0, end=1.0, velocity=127),  # C
        Note(pitch=79, start=1.1, end=1.3, velocity=127),  # G
        Note(pitch=36, start=1.1, end=1.3, velocity=127),  # C
    ]
    expected = [
        (NoteClassBase.C, AllChords.MINOR_TRIAD),  # C root note
        (NoteClassBase.B_SHARP, AllChords.MINOR_TRIAD),  # C enharmonic
    ]
    got = predict_chords(notes)
    assert set(expected) == set(got)


def test_predict_chords_f():
    # Edge case: Empty note_seq list
    notes = []
    expected = []  # C root note
    got = predict_chords(notes)
    assert expected == got


def test_predict_scales_degrees():
    notes = [
        Note(pitch=75, start=1.0, end=2.0, velocity=127),  # Eb
        Note(pitch=72, start=0.0, end=1.0, velocity=127),  # C
        Note(pitch=79, start=1.1, end=1.3, velocity=127),  # G
        Note(pitch=36, start=1.1, end=1.3, velocity=127),  # C
    ]
    expected = [
        # Degrees corresponding to C minor triad and B# minor triad
        (DegreesRoman.FIRST, Tonality.C_MINOR, ModeConstructors.NATURAL),
        (DegreesRoman.THIRD, Tonality.A_FLAT_MAJOR, ModeConstructors.MAJOR),
        (DegreesRoman.FOURTH, Tonality.G_MINOR, ModeConstructors.NATURAL),
        (DegreesRoman.SIXTH, Tonality.E_FLAT_MAJOR, ModeConstructors.MAJOR),
        (DegreesRoman.FIFTH, Tonality.F_MINOR, ModeConstructors.NATURAL),
        (DegreesRoman.SECOND, Tonality.B_FLAT_MAJOR, ModeConstructors.MAJOR),
    ]
    got = predict_scales_degrees(notes)
    assert set(expected).issubset(set(got))


def test_predict_possible_progressions_a():
    possible_chords = [
        # time step 1
        [
            # possible degrees of chords predicted in time step 1
            (DegreesRoman.FOURTH, Tonality.E_FLAT_MAJOR, ModeConstructors.MAJOR),
            (DegreesRoman.FOURTH, Tonality.E_FLAT_MAJOR, ModeConstructors.MAJOR),
            (DegreesRoman.SIXTH, Tonality.C_MINOR, ModeConstructors.DORIAN),
            (DegreesRoman.FIRST, Tonality.A_FLAT_MAJOR, ModeConstructors.MAJOR),
            (DegreesRoman.THIRD, Tonality.F_MINOR, ModeConstructors.NATURAL),
            (DegreesRoman.FIFTH, Tonality.D_FLAT_MAJOR, ModeConstructors.MAJOR),
            (DegreesRoman.SEVENTH, Tonality.B_FLAT_MINOR, ModeConstructors.NATURAL)
        ],
        # time step 2: any degree predicted
        [],
        # time step 3
        [
            # possible degrees of chord 1 predicted in time step 1
            (DegreesRoman.SECOND, Tonality.E_FLAT_MAJOR, ModeConstructors.MAJOR),
            (DegreesRoman.FOURTH, Tonality.C_MINOR, ModeConstructors.NATURAL),
            (DegreesRoman.SIXTH, Tonality.A_FLAT_MAJOR, ModeConstructors.MAJOR),
            (DegreesRoman.FIRST, Tonality.F_MINOR, ModeConstructors.NATURAL),
            (DegreesRoman.THIRD, Tonality.D_FLAT_MAJOR, ModeConstructors.MAJOR),
            (DegreesRoman.FIFTH, Tonality.B_FLAT_MINOR, ModeConstructors.NATURAL),
            # possible degrees of chord 2 predicted in time step 1
            (DegreesRoman.THIRD, Tonality.C_SHARP_MAJOR, ModeConstructors.MAJOR),
            (DegreesRoman.FIFTH, Tonality.A_SHARP_MINOR, ModeConstructors.NATURAL)
        ]
    ]
    expected = {
        "E_FLAT_MAJOR": [DegreesRoman.FOURTH, None, DegreesRoman.SECOND],
        "C_MINOR": [DegreesRoman.SIXTH, None, DegreesRoman.FOURTH],
        "A_FLAT_MAJOR": [DegreesRoman.FIRST, None, DegreesRoman.SIXTH],
        "F_MINOR": [DegreesRoman.THIRD, None, DegreesRoman.FIRST],
        "D_FLAT_MAJOR": [DegreesRoman.FIFTH, None, DegreesRoman.THIRD],
        "B_FLAT_MINOR": [DegreesRoman.SEVENTH, None, DegreesRoman.FIFTH],
        "C_SHARP_MAJOR": [None, None, DegreesRoman.THIRD],
        "A_SHARP_MINOR": [None, None, DegreesRoman.FIFTH]
    }
    got = predict_possible_progressions(possible_chords)
    assert set(got.keys()) == set(expected.keys())
    for (expected_k, expected_v), (got_k, got_v) in zip(expected.items(), got.items()):
        assert expected_k == got_k
        assert set(expected_v) == set(got_v)


def test_predict_progression_a():
    # Test case: Known scale, return only the progression corresponding to that scale
    scale = "C_MINOR_NATURAL"
    scales = {
        "E_FLAT_MAJOR": [DegreesRoman.FOURTH, None, DegreesRoman.SECOND],
        "C_MINOR_NATURAL": [DegreesRoman.SIXTH, None, DegreesRoman.FOURTH],
        "A_FLAT_MAJOR": [DegreesRoman.FIRST, None, DegreesRoman.SIXTH],
        "F_MINOR_NATURAL": [DegreesRoman.THIRD, None, DegreesRoman.FIRST],
        "D_FLAT_MAJOR": [DegreesRoman.FIFTH, None, DegreesRoman.THIRD],
        "B_FLAT_MINOR_NATURAL": [DegreesRoman.SEVENTH, None, DegreesRoman.FIFTH],
        "C_SHARP_MAJOR": [None, None, DegreesRoman.THIRD],
        "A_SHARP_MINOR_NATURAL": [None, None, DegreesRoman.FIFTH]
    }
    expected = (
        "C_MINOR_NATURAL", [DegreesRoman.SIXTH, None, DegreesRoman.FOURTH]
    )
    got = predict_progression(scales, scale)
    assert got[0] == expected[0]
    assert set(got[1]) == set(expected[1])


def test_get_harmonic_density():
    note_seq = [
        Note(pitch=26, start=1.0, end=2.0, velocity=127),  # D
        Note(pitch=65, start=0.0, end=1.0, velocity=127),  # F or E#
        Note(pitch=68, start=1.1, end=1.3, velocity=127),  # G# or Ab
        Note(pitch=47, start=1.1, end=1.3, velocity=127),  # B or Cb
        Note(pitch=38, start=1.0, end=2.0, velocity=127),  # D
        Note(pitch=47, start=1.1, end=1.3, velocity=127),  # B or Cb
    ]
    expected = 5
    got = get_harmonic_density(note_seq)
    assert expected == got

from musicaiz.harmony import Tonality, AllChords
from musicaiz.datasets import BPSFH
from musicaiz.structure import NoteClassBase


def test_bpsfh_key_to_musicaiz():

    note = "A-"
    expected = Tonality.A_FLAT_MAJOR

    got = BPSFH.bpsfh_key_to_musicaiz(note)
    assert got == expected


def test_bpsfh_chord_quality_to_musicaiz():

    quality = "a"
    expected = AllChords.AUGMENTED_TRIAD

    got = BPSFH.bpsfh_chord_quality_to_musicaiz(quality)
    assert got == expected


def test_bpsfh_chord_to_musicaiz():

    note = "f"
    quality = "D7"
    degree = 5
    expected = (NoteClassBase.C, AllChords.DOMINANT_SEVENTH)

    got = BPSFH.bpsfh_chord_to_musicaiz(note, degree, quality)
    assert got == expected

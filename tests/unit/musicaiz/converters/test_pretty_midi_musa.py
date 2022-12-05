from musicaiz.converters import (
    prettymidi_note_to_musicaiz,
    musicaiz_note_to_prettymidi,
    musa_to_prettymidi,
)
from musicaiz.loaders import Musa
from .test_musa_to_protobuf import midi_sample


def test_prettymidi_note_to_musicaiz():
    note = "G#4"
    expected_name = "G_SHARP"
    expected_octave = 4

    got_name, got_octave = prettymidi_note_to_musicaiz(note)

    assert got_name == expected_name
    assert got_octave == expected_octave


def test_musicaiz_note_to_prettymidi():
    note = "G_SHARP"
    octave = 4
    expected = "G#4"

    got = musicaiz_note_to_prettymidi(note, octave)

    assert got == expected


def test_musa_to_prettymidi(midi_sample):
    midi = Musa(midi_sample)
    got = musa_to_prettymidi(midi)

    assert len(got.instruments) == 2

    for inst in got.instruments:
        assert len(inst.notes) != 0

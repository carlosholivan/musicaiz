import pytest

from musicaiz.converters import musa_to_proto, proto_to_musa
from musicaiz.loaders import Musa


@pytest.fixture
def midi_sample(fixture_dir):
    return fixture_dir / "midis" / "midi_data.mid"


@pytest.fixture
def midi_data():
    return {
        "expected_instruments":  2,
        "expected_instrument_name_1": "Piano right",
        "expected_instrument_name_2": "Piano left",
    }


def _assert_midi_valid_instr_obj(midi_data, instruments):
    # check instrs
    assert midi_data["expected_instruments"] == len(instruments)
    # check instrs names
    assert midi_data["expected_instrument_name_1"] == instruments[0].name
    assert midi_data["expected_instrument_name_2"] == instruments[1].name
    # check instrs is_drum
    assert instruments[0].is_drum is False
    assert instruments[1].is_drum is False


def _assert_valid_note_obj(note):
    assert 0 <= note.pitch <= 128
    assert 0 <= note.velocity <= 128
    assert note.pitch_name != ""
    assert note.note_name != ""
    assert note.octave != ""
    assert note.symbolic != ""
    assert note.start_ticks >= 0
    assert note.end_ticks >= 0
    assert note.start_sec >= 0.0
    assert note.end_sec >= 0.0


def test_musa_to_proto(midi_sample, midi_data):
    midi = Musa(midi_sample)
    got = musa_to_proto(midi)

    _assert_midi_valid_instr_obj(midi_data, got.instruments)

    # check bars
    assert len(got.instruments) != 0
    assert len(got.bars) != 0

    # check every bar attributes are not empty
    for i, bar in enumerate(got.bars):
        # check only the first 5 bars since the midi file is large
        if i < 5:
            assert bar.start_ticks >= 0
            assert bar.end_ticks >= 0
            assert bar.start_sec >= 0.0
            assert bar.end_sec >= 0.0
    for note in got.notes:
        _assert_valid_note_obj(note)


def test_proto_to_musa(midi_sample, midi_data):
    midi = Musa(midi_sample)
    proto = musa_to_proto(midi)
    got = proto_to_musa(proto)

    _assert_midi_valid_instr_obj(midi_data, got.instruments)

    # check bars
    assert len(got.instruments) != 0

    # check every bar attributes are not empty
    for i, bar in enumerate(got.bars):
        # check only the first 5 bars since the midi file is large
        if i < 5:
            assert bar.start_ticks >= 0
            assert bar.end_ticks >= 0
            assert bar.start_sec >= 0.0
            assert bar.end_sec >= 0.0
    # check every note
    for note in got.notes:
        _assert_valid_note_obj(note)

from musicaiz.converters import (
    MusaJSON,
    BarJSON,
    InstrumentJSON,
    NoteJSON
)
from musicaiz.loaders import Musa
from .test_musa_to_protobuf import midi_sample


def test_MusaJSON(midi_sample):
    midi = Musa(
        midi_sample,
    )
    got = MusaJSON.to_json(midi)

    assert got["tonality"] is None
    assert got["resolution"] == 480
    assert len(got["instruments"]) == 2
    assert len(got["bars"]) == 3
    assert len(got["notes"]) == 37

    for inst in got["instruments"]:
        assert set(inst.keys()) == set(InstrumentJSON.__dataclass_fields__.keys())
    for bar in got["bars"]:
        assert set(bar.keys()) == set(BarJSON.__dataclass_fields__.keys())
    for note in got["notes"]:
        assert set(note.keys()) == set(NoteJSON.__dataclass_fields__.keys())

# TODO
#def test_JSONMusa(midi_sample, midi_data):

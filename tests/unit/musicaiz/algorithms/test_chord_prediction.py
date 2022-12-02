import pytest

from musicaiz.loaders import Musa
from musicaiz.algorithms import predict_chords


@pytest.fixture
def midi_sample(fixture_dir):
    return fixture_dir / "midis" / "midi_data.mid"


def test_predict_chords(midi_sample):
    # Import MIDI file
    midi = Musa(midi_sample)

    got = predict_chords(midi)
    assert len(got) == len(midi.beats)
    for i in got:
        assert len(i) != 0

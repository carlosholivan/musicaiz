import pytest

from musicaiz.features import StructurePrediction


@pytest.fixture
def midi_sample(fixture_dir):
    return fixture_dir / "midis" / "midi_changes.mid"


def test_StructurePrediction_notes(midi_sample):
    sp = StructurePrediction(midi_sample)

    dataset = "BPS"

    level = "low"
    got = sp.notes(level, dataset)
    assert len(got) != 0

    level = "mid"
    got = sp.notes(level, dataset)
    assert len(got) != 0

    level = "high"
    got = sp.notes(level, dataset)
    assert len(got) != 0


def test_StructurePrediction_beats(midi_sample):
    sp = StructurePrediction(midi_sample)

    dataset = "BPS"

    level = "low"
    got = sp.beats(level, dataset)
    assert len(got) != 0

    level = "mid"
    got = sp.beats(level, dataset)
    assert len(got) != 0

    level = "high"
    got = sp.beats(level, dataset)
    assert len(got) != 0

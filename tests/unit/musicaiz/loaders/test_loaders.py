import pytest

from musicaiz.loaders import Musa
from musicaiz.algorithms import KeyDetectionAlgorithms


@pytest.fixture
def midi_sample(fixture_dir):
    return fixture_dir / "midis" / "midi_data.mid"


def _assert_key_profiles(midi_sample, methods, expected):
    # try both instruments and bars initializations in Musa
    midi_instr = Musa(midi_sample, structure="instruments")
    midi_bars = Musa(midi_sample, structure="bars")
    for method in methods:
        got = midi_instr.predict_key(method)
        assert got == expected

        got = midi_bars.predict_key(method)
        assert got == expected


def test_predict_key_kk(midi_sample):
    # Test case: K-K
    methods = KeyDetectionAlgorithms.KRUMHANSL_KESSLER.value
    expected = "F_MAJOR"

    _assert_key_profiles(midi_sample, methods, expected)


def test_predict_key_temperley(midi_sample):
    # Test case: K-K
    methods = KeyDetectionAlgorithms.TEMPERLEY.value
    expected = "F_MAJOR"

    _assert_key_profiles(midi_sample, methods, expected)


def test_predict_key_albretch(midi_sample):
    # Test case: K-K
    methods = KeyDetectionAlgorithms.ALBRETCH_SHANAHAN.value
    expected = "F_MAJOR"

    _assert_key_profiles(midi_sample, methods, expected)


def test_predict_key_5ths(midi_sample):
    # Test case: K-K
    methods = KeyDetectionAlgorithms.SIGNATURE_FIFTHS.value
    expected = "A_SHARP_MAJOR"

    midi_instr = Musa(midi_sample, structure="instruments")
    midi_bars = Musa(midi_sample, structure="bars")
    for method in methods:
        # Signature 5ths does not work for structure="instruments"
        with pytest.raises(ValueError):
            midi_instr.predict_key(method)

        got = midi_bars.predict_key(method)
        assert got == expected


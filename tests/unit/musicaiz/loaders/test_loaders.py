import pytest

from musicaiz.loaders import Musa
from musicaiz.algorithms import KeyDetectionAlgorithms


@pytest.fixture
def midi_sample(fixture_dir):
    return fixture_dir / "midis" / "midi_changes.mid"


@pytest.fixture
def midi_sample_2(fixture_dir):
    return fixture_dir / "midis" / "midi_data.mid"


def test_Musa(midi_sample):

    # args
    quantize = True
    cut_notes = False
    absolute_timing = False
    general_midi = True
    subdivision_note = "sixteenth"
    midi = Musa(
        file=midi_sample,
        quantize=quantize,
        cut_notes=cut_notes,
        absolute_timing=absolute_timing,
        general_midi=general_midi,
        subdivision_note=subdivision_note,
    )

    # check attributes
    assert midi.file.stem == "midi_changes"
    assert midi.total_bars != 0
    assert midi.tonality is None
    assert midi.subdivision_note == subdivision_note
    assert len(midi.time_signature_changes) != 0
    assert midi.resolution != 0
    assert len(midi.instruments) != 0
    assert midi.is_quantized == quantize
    assert midi.absolute_timing == absolute_timing
    assert midi.cut_notes == cut_notes
    assert len(midi.notes) != 0
    assert len(midi.bars) != 0
    assert len(midi.subbeats) != 0
    assert len(midi.tempo_changes) != 0
    assert len(midi.instruments_progs) != 0

    midi.bar_beats_subdivs_analysis()

    # Test methods
    notes = midi.get_notes_in_subbeat(
        subbeat_idx=0, program=48, instrument_idx=None
    )
    assert len([n.subbeat_idx for n in notes if n.subbeat_idx != 0]) == 0

    notes = midi.get_notes_in_subbeat_bar(
        subbeat_idx=0, bar_idx=40, program=48, instrument_idx=None
    )
    assert len([n.bar_idx for n in notes if n.bar_idx != 40]) == 0

    notes = midi.get_notes_in_subbeats(
        subbeat_start=0, subbeat_end=4, program=48, instrument_idx=None
    )
    assert len([n.subbeat_idx for n in notes if n.subbeat_idx >= 4]) == 0

    notes = midi.get_notes_in_beat(
        beat_idx=0, program=48, instrument_idx=None
    )
    assert len([n.beat_idx for n in notes if n.beat_idx != 0]) == 0

    notes = midi.get_notes_in_subbeat_bar(
        subbeat_idx=0, bar_idx=40, program=48, instrument_idx=None
    )
    assert len([n.bar_idx for n in notes if n.bar_idx != 40]) == 0

    subbeats = midi.get_subbeats_in_beat(beat_idx=50)
    assert len([n.beat_idx for n in subbeats if n.beat_idx != 50]) == 0

    subbeat = midi.get_subbeat_in_beat(subbeat_idx=2, beat_idx=50)
    assert subbeat.beat_idx == 50

    notes = midi.get_notes_in_beats(
        beat_start=10, beat_end=20, program=48, instrument_idx=None
    )
    assert len([n.beat_idx for n in notes if n.beat_idx >= 20 and n.beat_idx < 10]) == 0

    subbeats = midi.get_subbeats_in_beats(
        beat_start=10, beat_end=20
    )
    assert len([n.beat_idx for n in subbeats if n.beat_idx >= 20 and n.beat_idx < 10]) == 0

    notes = midi.get_notes_in_bar(
        bar_idx=12, program=48, instrument_idx=None
    )
    assert len([n.bar_idx for n in notes if n.bar_idx != 12]) == 0

    beats = midi.get_beats_in_bar(
        bar_idx=30
    )
    assert len([n.bar_idx for n in beats if n.bar_idx != 30]) == 0

    beat = midi.get_beat_in_bar(
        beat_idx=0, bar_idx=60
    )
    assert beat.bar_idx == 60

    subbeats = midi.get_subbeats_in_bar(bar_idx=1)
    assert len([n.bar_idx for n in subbeats if n.bar_idx != 1]) == 0

    subbeat = midi.get_subbeat_in_bar(
        subbeat_idx=14, bar_idx=1
    )
    assert subbeat.bar_idx == 1

    notes = midi.get_notes_in_bars(
        bar_start=30, bar_end=32
    )
    assert len([n.bar_idx for n in notes if n.bar_idx < 30 and n.bar_idx >= 32]) == 0

    beats = midi.get_beats_in_bars(
        bar_start=30, bar_end=32
    )
    assert len([n.bar_idx for n in beats if n.bar_idx < 30 and n.bar_idx >= 32]) == 0

    subbeats = midi.get_subbeats_in_bars(
        bar_start=30, bar_end=32
    )
    assert len([n.bar_idx for n in subbeats if n.bar_idx < 30 and n.bar_idx >= 32]) == 0

    # Test when passing more than one instrument program number
    notes_is2 = midi.get_notes_in_bar(
        bar_idx=0,
        program=[48, 45],
        instrument_idx=[0, 1]
    )
    assert len(notes_is2) != 0
    for n in notes_is2:
        assert n.instrument_prog in [48, 45]
        assert n.instrument_idx in [0, 1]

    # Test errors
    # error when bar_idx does not exist
    with pytest.raises(ValueError):
        midi.get_notes_in_bar(bar_idx=10000)

    # error when bar_start > bar_end
    with pytest.raises(ValueError):
        midi.get_notes_in_bars(10, 1)

    # error when no program number found
    with pytest.raises(ValueError):
        midi.get_notes_in_bar(
            bar_idx=0, program=100, instrument_idx=None
        )

    # error when program does not match instrument_idx
    # instrument_idx=4 corresponds with program 49, error
    with pytest.raises(ValueError):
        midi.get_notes_in_bar(
            bar_idx=0, program=0, instrument_idx=4
        )

    # error when programs and instruments_idxs have diff len
    with pytest.raises(ValueError):
        midi.get_notes_in_bar(
            bar_idx=0, program=[100, 47], instrument_idx=[0]
        )

    # error when programs do not match instrument_idxs
    with pytest.raises(ValueError):
        midi.get_notes_in_bar(
            bar_idx=2, program=[48, 0], instrument_idx=[1, 2]
        )


# Predict key tests
def _assert_key_profiles(midi_sample_2, methods, expected):
    # try both instruments and bars initializations in Musa
    midi_instr = Musa(midi_sample_2)
    for method in methods:
        got = midi_instr.predict_key(method)
        assert got == expected


def test_predict_key_kk(midi_sample_2):
    # Test case: K-K
    methods = KeyDetectionAlgorithms.KRUMHANSL_KESSLER.value
    expected = "F_MAJOR"

    _assert_key_profiles(midi_sample_2, methods, expected)


def test_predict_key_temperley(midi_sample_2):
    # Test case: K-K
    methods = KeyDetectionAlgorithms.TEMPERLEY.value
    expected = "F_MAJOR"

    _assert_key_profiles(midi_sample_2, methods, expected)


def test_predict_key_albretch(midi_sample_2):
    # Test case: K-K
    methods = KeyDetectionAlgorithms.ALBRETCH_SHANAHAN.value
    expected = "F_MAJOR"

    _assert_key_profiles(midi_sample_2, methods, expected)


def test_predict_key_5ths(midi_sample_2):
    # Test case: K-K
    methods = KeyDetectionAlgorithms.SIGNATURE_FIFTHS.value
    expected = "A_SHARP_MAJOR"

    _assert_key_profiles(midi_sample_2, methods, expected)

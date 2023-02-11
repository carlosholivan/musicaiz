import pytest


from musicaiz.loaders import Musa
from musicaiz.structure import Instrument, Bar, Note
from musicaiz.tokenizers import MMMTokenizer, MMMTokenizerArguments


@pytest.fixture
def mmm_tokens(fixture_dir):
    tokens_path = fixture_dir / "tokenizers" / "mmm_tokens.txt"
    text_file = open(tokens_path, "r")
    # read whole file to a string
    yield text_file.read()


@pytest.fixture
def mmm_multiple_tokens(fixture_dir):
    tokens_path = fixture_dir / "tokenizers" / "mmm_multiple_tokens.txt"
    text_file = open(tokens_path, "r")
    # read whole file to a string
    yield text_file.read()


@pytest.fixture
def midi_sample(fixture_dir):
    return fixture_dir / "tokenizers" / "mmm_tokens.mid"


@pytest.fixture
def musa_obj_tokens():
    # Initialize Musa obj fot the mmm_tokens.txt sequence
    musa_obj = Musa(file=None)
    musa_obj.instruments.append(
        Instrument(
            program=30,
            general_midi=True
        )
    )
    for _ in range(0, 3, 1):
        musa_obj.bars.append(Bar())
    return musa_obj


@pytest.fixture
def musa_obj_abs(musa_obj_tokens):
    # Add notes to the Musa obj with absolute timings
    notes_bar1 = [
        Note(pitch=69, start=0.5, end=1.0, velocity=127, bar_idx=0, instrument_prog=30),
        Note(pitch=64, start=0.5, end=1.5, velocity=127, bar_idx=0, instrument_prog=30),
        Note(pitch=67, start=1.0, end=1.5, velocity=127, bar_idx=0, instrument_prog=30),
        Note(pitch=64, start=1.5, end=2.0, velocity=127, bar_idx=0, instrument_prog=30)
    ]
    musa_obj_tokens.notes.extend(notes_bar1)
    musa_obj_tokens.bars[0].start_ticks = 0
    musa_obj_tokens.bars[0].end_ticks = 96 * 4
    # bar2 is empty
    musa_obj_tokens.bars[1].start_ticks = 96 * 4
    musa_obj_tokens.bars[1].end_ticks = 96 * 8
    notes_bar3 = [
        Note(pitch=72, start=4.0, end=4.5, velocity=127, bar_idx=2, instrument_prog=30),
        Note(pitch=69, start=4.5, end=5.0, velocity=127, bar_idx=2, instrument_prog=30),
        Note(pitch=67, start=5.5, end=5.75, velocity=127, bar_idx=2, instrument_prog=30),
    ]
    musa_obj_tokens.notes.extend(notes_bar3)
    musa_obj_tokens.bars[2].start_ticks = 96 * 8
    musa_obj_tokens.bars[2].end_ticks = 96 * 12
    return musa_obj_tokens


@pytest.fixture
def musa_obj_rel(musa_obj_tokens):
    # Add notes to the Musa obj with relative timings
    notes_bar1 = [
        Note(pitch=69, start=0.5, end=1.0, velocity=127, bar_idx=0, instrument_prog=30),
        Note(pitch=64, start=0.5, end=1.5, velocity=127, bar_idx=0, instrument_prog=30),
        Note(pitch=67, start=1.0, end=1.5, velocity=127, bar_idx=0, instrument_prog=30),
        Note(pitch=64, start=1.5, end=2.0, velocity=127, bar_idx=0, instrument_prog=30)
    ]
    musa_obj_tokens.notes.extend(notes_bar1)
    musa_obj_tokens.bars[0].start_ticks = 0
    musa_obj_tokens.bars[0].end_ticks = 96 * 4
    # bar2 is empty
    musa_obj_tokens.bars[1].start_ticks = 0
    musa_obj_tokens.bars[1].end_ticks = 96 * 4
    notes_bar3 = [
        Note(pitch=72, start=0.0, end=0.5, velocity=127, bar_idx=2, instrument_prog=30),
        Note(pitch=69, start=0.5, end=1.0, velocity=127, bar_idx=2, instrument_prog=30),
        Note(pitch=67, start=1.5, end=1.75, velocity=127, bar_idx=2, instrument_prog=30),
    ]
    musa_obj_tokens.notes.extend(notes_bar3)
    musa_obj_tokens.bars[2].start_ticks = 0
    musa_obj_tokens.bars[2].end_ticks = 96 * 4
    return musa_obj_tokens


def _assert_valid_musa_obj(got_musa_obj, expected_musa_obj):
    assert len(got_musa_obj.instruments) == len(expected_musa_obj.instruments)
    assert got_musa_obj.instruments[0].program == expected_musa_obj.instruments[0].program
    assert len(got_musa_obj.bars) == len(expected_musa_obj.bars)
    for n in range(len(expected_musa_obj.notes)):
        got_note = got_musa_obj.notes[n]
        expected_note = expected_musa_obj.notes[n]
        assert expected_note.pitch == got_note.pitch
        assert expected_note.start_ticks == got_note.start_ticks
        assert expected_note.end_ticks == got_note.end_ticks
        assert expected_note.velocity == got_note.velocity


def test_MMMTokenizer_split_tokens_by_track():
    tokens = [
        "TRACK_START",
        "INST=30",
        "BAR_START",
        "NOTE_ON=30",
        "TIME_DELTA=4",
        "NOTE_OFF=30",
        "BAR_END",
        "BAR_START",
        "NOTE_ON=30",
        "TIME_DELTA=4",
        "NOTE_OFF=30",
        "BAR_END",
        "TRACK_END",
        # New track
        "TRACK_START",
        "INST=31",
        "BAR_START",
        "NOTE_ON=30",
        "TIME_DELTA=4",
        "NOTE_OFF=30",
        "BAR_END",
        "BAR_START",
        "NOTE_ON=30",
        "TIME_DELTA=4",
        "NOTE_OFF=30",
        "BAR_END",
        "TRACK_END",
    ]

    expected = [
        [
            "TRACK_START",
            "INST=30",
            "BAR_START",
            "NOTE_ON=30",
            "TIME_DELTA=4",
            "NOTE_OFF=30",
            "BAR_END",
            "BAR_START",
            "NOTE_ON=30",
            "TIME_DELTA=4",
            "NOTE_OFF=30",
            "BAR_END",
            "TRACK_END"
        ],
        # New track
        [
            "TRACK_START",
            "INST=31",
            "BAR_START",
            "NOTE_ON=30",
            "TIME_DELTA=4",
            "NOTE_OFF=30",
            "BAR_END",
            "BAR_START",
            "NOTE_ON=30",
            "TIME_DELTA=4",
            "NOTE_OFF=30",
            "BAR_END",
            "TRACK_END"
        ]
    ]
    got = MMMTokenizer.split_tokens_by_track(tokens)
    for i in range(len(expected)):
        assert set(expected[i]) == set(got[i])


def test_MMMTokenizer_split_tokens_by_bar():
    tokens = [
        "TRACK_START",
        "INST=31",
        "BAR_START",
        "NOTE_ON=30",
        "TIME_DELTA=4",
        "NOTE_OFF=30",
        "BAR_END",
        "BAR_START",
        "NOTE_ON=30",
        "TIME_DELTA=4",
        "NOTE_OFF=30",
        "BAR_END",
        "TRACK_END"
    ]

    expected = [
        [
            "BAR_START",
            "NOTE_ON=30",
            "TIME_DELTA=4",
            "NOTE_OFF=30",
            "BAR_END"
        ],
        # New bar
        [
            "BAR_START",
            "NOTE_ON=30",
            "TIME_DELTA=4",
            "NOTE_OFF=30",
            "BAR_END"
        ]
    ]
    got = MMMTokenizer.split_tokens_by_bar(tokens)
    for i in range(len(expected)):
        assert set(expected[i]) == set(got[i])


def test_MMMTokenizer_tokens_to_musa_a(musa_obj_abs, mmm_tokens):
    # Test case: 1 polyphonic instrument, absolute timings
    got = MMMTokenizer.tokens_to_musa(
        tokens=mmm_tokens,
        absolute_timing=True,
        time_unit="SIXTEENTH"
    )
    expected = musa_obj_abs
    _assert_valid_musa_obj(got, expected)


def test_MMMTokenizer_tokens_to_musa_b(musa_obj_rel, mmm_tokens):
    # Test case: 1 polyphonic instrument, relative timings
    got = MMMTokenizer.tokens_to_musa(
        tokens=mmm_tokens,
        absolute_timing=False,
        time_unit="SIXTEENTH"
    )
    expected = musa_obj_rel

    _assert_valid_musa_obj(got, expected)


def test_MMMTokenizer_get_pieces_tokens(mmm_multiple_tokens):
    got = MMMTokenizer.get_pieces_tokens(mmm_multiple_tokens)
    expected_len = 4
    assert expected_len == len(got)


def test_MMMTokenizer_get_tokens_analytics(mmm_multiple_tokens):
    got = MMMTokenizer.get_tokens_analytics(mmm_multiple_tokens)
    expected_total_tokens = 725
    expected_unique_tokens = 59
    expected_total_notes = 188
    expected_unique_notes = 23
    expected_total_bars = 112
    expected_total_instruments = 16
    expected_total_pieces = 4

    assert expected_total_pieces == got["total_pieces"]
    assert expected_total_tokens == got["total_tokens"]
    assert expected_unique_tokens == got["unique_tokens"]
    assert expected_total_notes == got["total_notes"]
    assert expected_unique_notes == got["unique_notes"]
    assert expected_total_bars == got["total_bars"]
    assert expected_total_instruments == got["total_instruments"]


def test_MMMTokenizer_tokenize_track_bars(musa_obj_abs, mmm_tokens):
    # Notes that start at the same time are not ordered in a particular way,
    # this may cause this test to fail if other data is tested.
    # However, this is not a problem for the tokenization.
    start_bar = mmm_tokens.index("BAR_START")
    end_bar = mmm_tokens.index("TRACK_END")
    expected = mmm_tokens[start_bar:end_bar]

    args = MMMTokenizerArguments(time_unit="SIXTEENTH")
    tokenizer = MMMTokenizer(args=args)

    tokenizer.midi_object.notes = musa_obj_abs.notes
    tokenizer.midi_object.bars = musa_obj_abs.bars
    tokenizer.midi_object.instruments_progs = [musa_obj_abs.instruments[0].program]
    got = tokenizer.tokenize_track_bars(
        bar_start_idx=0,
        bars=tokenizer.midi_object.bars,
        program=30
    )
    assert got == expected


def test_MMMTokenizer_tokenize_tracks(musa_obj_abs, mmm_tokens):
    start = mmm_tokens.index("TRACK_START")
    end = mmm_tokens.index("BAR_END")
    expected = mmm_tokens[start:end] + "BAR_END TRACK_END"

    args = MMMTokenizerArguments(time_unit="SIXTEENTH")
    tokenizer = MMMTokenizer(args=args)

    # since we don't pass the file as an argument, we need to
    # create the Musa attributes with the musa_obj_abs
    # When a file is provided this is done automatically
    tokenizer.midi_object.notes = musa_obj_abs.notes
    tokenizer.midi_object.bars = musa_obj_abs.bars
    tokenizer.midi_object.instruments_progs = [musa_obj_abs.instruments[0].program]

    got = tokenizer.tokenize_tracks(
        instruments=musa_obj_abs.instruments,
        bar_start=0,
        bar_end=1
    )
    assert got == expected


def test_MMMTokenizer_tokenize(midi_sample):
    args = MMMTokenizerArguments(time_unit="SIXTEENTH")
    tokenizer = MMMTokenizer(midi_sample, args)
    got = tokenizer.tokenize_file()
    assert got != ""

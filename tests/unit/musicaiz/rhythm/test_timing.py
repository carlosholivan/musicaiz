import pytest


from musicaiz.rhythm import (
    NoteLengths,
    ms_per_tick,
    _bar_str_to_tuple,
    ticks_per_bar,
    ms_per_note,
    ms_per_bar,
    get_subdivisions,
)


# ===============NoteLengths class Tests===============
# =====================================================
def test_NoteLengths_a():
    resolution = 960
    expected_ticks = 960
    expected_ms = 500
    note = NoteLengths.QUARTER
    got_ticks = note.ticks(resolution=resolution)
    got_ms = note.ms(resolution=resolution)
    assert expected_ticks == got_ticks
    assert abs(got_ms - expected_ms) <= 1e-3


def test_NoteLengths_b():
    resolution = 960
    expected = 2880
    got = NoteLengths.DOTTED_HALF.ticks(resolution=resolution)
    assert expected == got


def test_NoteLengths_c():
    resolution = 960
    expected = 640
    got = NoteLengths.QUARTER_TRIPLET.ticks(resolution=resolution)
    assert expected == got


# ===============Functions Tests=======================
# =====================================================
def test_ms_per_tick():
    bpm = 96
    resolution = 960
    expected = 0.651
    got = ms_per_tick(bpm, resolution)
    assert abs(got - expected) <= 1e-3


def test_ms_per_note_a():
    # Test case: default 120bpm and quarter note
    note_length = "quarter"
    expected = 500.0
    got = ms_per_note(note_length)
    assert abs(got - expected) <= 1e-3


def test_ms_per_note_b():
    # Test case: 90bpm and eight note
    bpm = 90
    note_length = "eight"
    expected = 333.333
    got = ms_per_note(note_length, bpm)
    assert abs(got - expected) <= 1e-3


def test_ticks_per_bar_a():
    resolution = 960
    time_sig = "4/4"
    expected_beat = 960
    expected_bar = 3840
    got_beat, got_bar = ticks_per_bar(time_sig, resolution)
    assert got_beat == expected_beat
    assert expected_bar == got_bar


def test_ticks_per_bar_b():
    resolution = 960
    time_sig = "3/8"
    expected_beat = 480
    expected_bar = 1440
    got_beat, got_bar = ticks_per_bar(time_sig, resolution)
    assert got_beat == expected_beat
    assert expected_bar == got_bar


def test_ms_per_bar_a():
    # Test case: Default 120bpm and 3/8 bar
    resolution = 960
    bpm = 120
    time_sig = "3/8"
    expected_beat = 250
    expected_bar = 750
    got_beat, got_bar = ms_per_bar(time_sig, bpm, resolution)
    assert int(got_beat) == expected_beat
    assert int(expected_bar) == got_bar


def test_ms_per_bar_b():
    # Test case: Same time sig as prev. test, other resolution
    time_sig = "3/8"
    bpm = 120
    resolution = 1536  # Cubase's resolution
    expected_beat = 250
    expected_bar = 750
    got_beat, got_bar = ms_per_bar(time_sig, bpm, resolution)
    assert abs(got_beat - expected_beat) <= 1e-3
    assert abs(got_bar - expected_bar) <= 1e-3


def test_ms_per_bar_c():
    # Test case: Non common time sig
    bar = "1/16"
    bpm = 65
    resolution = 960
    expected_beat = 230.77
    expected_bar = 230.77
    got_beat, got_bar = ms_per_bar(bar, bpm, resolution)
    assert abs(got_beat - expected_beat) <= 1e-3
    assert abs(got_bar - expected_bar) <= 1e-3


def test_ms_per_bar_d():
    bar = "1/16"
    bpm = 65
    resolution = 1536  # Cubase's resolution
    expected_beat = 230.77
    expected_bar = 230.77
    got_beat, got_bar = ms_per_bar(bar, bpm, resolution)
    assert abs(got_beat - expected_beat) <= 1e-3
    assert abs(got_bar - expected_bar) <= 1e-3


def test_bar_str_to_tuple_a():
    bar = "3/4"
    expected = (3, 4)
    got_n, got_d = _bar_str_to_tuple(bar)
    assert got_n == expected[0]
    assert got_d == expected[1]


def test_bar_str_to_tuple_b():
    bar = "4"
    with pytest.raises(ValueError):
        _bar_str_to_tuple(bar)


def test_get_subdivisions_a():
    # We assume a resolution of 960 PPQ
    time_sig = "2/4"
    subdivision = "quarter"
    bpm = 120
    total_bars = 2
    resolution = 960

    expected = [
        {
            "bar": 1,
            "piece_beat": 1,
            "piece_subdivision": 1,
            "bar_beat": 1,
            "bar_subdivision": 1,
            "ticks": 0,
            "sec": 0,
        },
        {
            "bar": 1,
            "piece_beat": 2,
            "piece_subdivision": 2,
            "bar_beat": 2,
            "bar_subdivision": 2,
            "ticks": 960,
            "sec": 0.5,
        },
        {
            "bar": 2,
            "piece_beat": 3,
            "piece_subdivision": 3,
            "bar_beat": 1,
            "bar_subdivision": 1,
            "ticks": 1920,
            "sec": 1.0,
        },
        {
            "bar": 2,
            "piece_beat": 4,
            "piece_subdivision": 4,
            "bar_beat": 2,
            "bar_subdivision": 2,
            "ticks": 2880,
            "sec": 1.5,
        }
    ]
    got = get_subdivisions(total_bars, subdivision, time_sig, bpm, resolution)
    assert len(expected) == len(got)  # same total bars
    for subdiv_idx in range(len(expected)):
        assert set(expected[subdiv_idx].keys()) == set(got[subdiv_idx].keys())
        assert expected[subdiv_idx]["bar"] == got[subdiv_idx]["bar"]
        assert expected[subdiv_idx]["piece_subdivision"] == got[subdiv_idx]["piece_subdivision"]
        assert expected[subdiv_idx]["bar_subdivision"] == got[subdiv_idx]["bar_subdivision"]
        assert expected[subdiv_idx]["bar_beat"] == got[subdiv_idx]["bar_beat"]
        assert expected[subdiv_idx]["piece_beat"] == got[subdiv_idx]["piece_beat"]
        assert expected[subdiv_idx]["ticks"] == got[subdiv_idx]["ticks"]
        assert abs(expected[subdiv_idx]["sec"] - got[subdiv_idx]["sec"]) <= 1e-6


def test_get_subdivisions_b():
    # Division lower than beat
    # We assume a resolution of 960 PPQ
    time_sig = "2/4"
    subdivision = "eight"
    bpm = 120
    total_bars = 2
    resolution = 960

    expected = [
        # 1st 8th note in 1st bar (1st beat)
        {
            "bar": 1,
            "piece_beat": 1,
            "piece_subdivision": 1,
            "bar_beat": 1,
            "bar_subdivision": 1,
            "ticks": 0,
            "sec": 0,
        },
        # 2nd 8th note in 1st bar (1st beat)
        {
            "bar": 1,
            "piece_beat": 1,
            "piece_subdivision": 2,
            "bar_beat": 1,
            "bar_subdivision": 2,
            "ticks": 480,
            "sec": 0.25,
        },
        # 3rd 8th note in 1st bar (2nd beat)
        {
            "bar": 1,
            "piece_beat": 2,
            "piece_subdivision": 3,
            "bar_beat": 2,
            "bar_subdivision": 3,
            "ticks": 960,
            "sec": 0.5,
        },
        # 4th 8th note in 1st bar (2nd beat)
        {
            "bar": 1,
            "piece_beat": 2,
            "piece_subdivision": 4,
            "bar_beat": 2,
            "bar_subdivision": 4,
            "ticks": 1440,
            "sec": 0.75,
        },
        # 1st 8th note in 2nd bar (1st beat)
        {
            "bar": 2,
            "piece_beat": 3,
            "piece_subdivision": 5,
            "bar_beat": 1,
            "bar_subdivision": 1,
            "ticks": 1920,
            "sec": 1.0,
        },
        # 2nd 8th note in 1st bar (1st beat)
        {
            "bar": 2,
            "piece_beat": 3,
            "piece_subdivision": 6,
            "bar_beat": 1,
            "bar_subdivision": 2,
            "ticks": 2400,
            "sec": 1.25,
        },
        # 3rd 8th note in 1st bar (2nd beat)
        {
            "bar": 2,
            "piece_beat": 4,
            "piece_subdivision": 7,
            "bar_beat": 2,
            "bar_subdivision": 3,
            "ticks": 2880,
            "sec": 1.5,
        },
        {
            "bar": 2,
            "piece_beat": 4,
            "piece_subdivision": 8,
            "bar_beat": 2,
            "bar_subdivision": 4,
            "ticks": 3360,
            "sec": 1.75,
        }
    ]
    got = get_subdivisions(total_bars, subdivision, time_sig, bpm, resolution)
    assert len(expected) == len(got)  # same total bars
    for subdiv_idx in range(len(expected)):
        assert set(expected[subdiv_idx].keys()) == set(got[subdiv_idx].keys())
        assert expected[subdiv_idx]["bar"] == got[subdiv_idx]["bar"]
        assert expected[subdiv_idx]["piece_subdivision"] == got[subdiv_idx]["piece_subdivision"]
        assert expected[subdiv_idx]["bar_subdivision"] == got[subdiv_idx]["bar_subdivision"]
        assert expected[subdiv_idx]["bar_beat"] == got[subdiv_idx]["bar_beat"]
        assert expected[subdiv_idx]["piece_beat"] == got[subdiv_idx]["piece_beat"]
        assert expected[subdiv_idx]["ticks"] == got[subdiv_idx]["ticks"]
        assert abs(expected[subdiv_idx]["sec"] - got[subdiv_idx]["sec"]) <= 1e-6


def test_get_subdivisions_c():
    # Test case: subdivision greater than beat, not possible
    time_sig = "3/8"
    subdivision = "quarter"
    total_bars = 1

    with pytest.raises(ValueError):
        get_subdivisions(total_bars, subdivision, time_sig)


def test_get_subdivisions_d():
    # Test case: 3/8 time sig and 90 bpm (non default)
    # We assume a resolution of 960 PPQ (default)
    time_sig = "3/8"
    subdivision = "eight"
    bpm = 90
    total_bars = 1
    resolution = 960

    expected = [
        {
            "bar": 1,
            "piece_beat": 1,
            "piece_subdivision": 1,
            "bar_subdivision": 1,
            "bar_beat": 1,
            "ticks": 0,
            "sec": 0,
        },
        {
            "bar": 1,
            "piece_beat": 2,
            "piece_subdivision": 2,
            "bar_subdivision": 2,
            "bar_beat": 2,
            "ticks": 480,
            "sec": 0.333333,
        },
        {
            "bar": 1,
            "piece_beat": 3,
            "piece_subdivision": 3,
            "bar_subdivision": 3,
            "bar_beat": 3,
            "ticks": 960,
            "sec": 0.666666,
        }
    ]
    got = get_subdivisions(total_bars, subdivision, time_sig, bpm, resolution)
    assert len(expected) == len(got)  # same total bars
    for subdiv_idx in range(len(expected)):
        assert set(expected[subdiv_idx].keys()) == set(got[subdiv_idx].keys())
        assert expected[subdiv_idx]["bar"] == got[subdiv_idx]["bar"]
        assert expected[subdiv_idx]["piece_subdivision"] == got[subdiv_idx]["piece_subdivision"]
        assert expected[subdiv_idx]["bar_subdivision"] == got[subdiv_idx]["bar_subdivision"]
        assert expected[subdiv_idx]["bar_beat"] == got[subdiv_idx]["bar_beat"]
        assert expected[subdiv_idx]["piece_beat"] == got[subdiv_idx]["piece_beat"]
        assert expected[subdiv_idx]["ticks"] == got[subdiv_idx]["ticks"]
        assert abs(expected[subdiv_idx]["sec"] - got[subdiv_idx]["sec"]) <= 1e-6


def test_get_subdivisions_a():
    # With absolute_timing = False which means relative timing
    time_sig = "2/4"
    subdivision = "quarter"
    bpm = 120
    total_bars = 2
    resolution = 960
    absolute_timing = False

    expected = [
        {
            "bar": 1,
            "piece_beat": 1,
            "piece_subdivision": 1,
            "bar_beat": 1,
            "bar_subdivision": 1,
            "ticks": 0,
            "sec": 0,
        },
        {
            "bar": 1,
            "piece_beat": 2,
            "piece_subdivision": 2,
            "bar_beat": 2,
            "bar_subdivision": 2,
            "ticks": 960,
            "sec": 0.5,
        },
        {
            "bar": 2,
            "piece_beat": 3,
            "piece_subdivision": 3,
            "bar_beat": 1,
            "bar_subdivision": 1,
            "ticks": 0,  # in relative timing, each abr starts at 0
            "sec": 0.0,
        },
        {
            "bar": 2,
            "piece_beat": 4,
            "piece_subdivision": 4,
            "bar_beat": 2,
            "bar_subdivision": 2,
            "ticks": 2880 - 1920,
            "sec": 1.5 - 1.0,
        }
    ]
    got = get_subdivisions(
        total_bars, subdivision, time_sig, bpm, resolution, absolute_timing
    )
    assert len(expected) == len(got)  # same total bars
    for subdiv_idx in range(len(expected)):
        assert set(expected[subdiv_idx].keys()) == set(got[subdiv_idx].keys())
        assert expected[subdiv_idx]["bar"] == got[subdiv_idx]["bar"]
        assert expected[subdiv_idx]["piece_subdivision"] == got[subdiv_idx]["piece_subdivision"]
        assert expected[subdiv_idx]["bar_subdivision"] == got[subdiv_idx]["bar_subdivision"]
        assert expected[subdiv_idx]["bar_beat"] == got[subdiv_idx]["bar_beat"]
        assert expected[subdiv_idx]["piece_beat"] == got[subdiv_idx]["piece_beat"]
        assert expected[subdiv_idx]["ticks"] == got[subdiv_idx]["ticks"]
        assert abs(expected[subdiv_idx]["sec"] - got[subdiv_idx]["sec"]) <= 1e-6

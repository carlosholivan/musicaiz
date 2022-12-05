import pytest
import numpy as np

from musicaiz import rhythm
from musicaiz.structure import Note
from musicaiz.rhythm.quantizer import (
    QuantizerConfig,
    basic_quantizer,
    get_ticks_from_subdivision,
    advanced_quantizer,
    _find_nearest,
)


@pytest.fixture
def grid_16():
    grid = rhythm.get_subdivisions(
        total_bars=1, subdivision="sixteenth", time_sig="4/4", bpm=120, resolution=96
    )
    v_grid = get_ticks_from_subdivision(grid)

    return v_grid


@pytest.fixture
def grid_8():
    grid = rhythm.get_subdivisions(
        total_bars=1, subdivision="eight", time_sig="4/4", bpm=120, resolution=96
    )

    v_grid = get_ticks_from_subdivision(grid)

    return v_grid


def test_find_nearest_a():
    input = [1, 2, 3, 4, 5, 6, 7, 8]
    value = 2.2
    expected = 2

    got = _find_nearest(input, value)

    assert got == expected


def test_find_nearest_b():
    input = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    value = 6.51
    expected = 7

    got = _find_nearest(input, value)

    assert got == expected


def test_basic_quantizer(grid_16):

    notes_bar1 = [
        Note(pitch=69, start=1, end=24, velocity=127),
        Note(pitch=64, start=12, end=24, velocity=127),
        Note(pitch=67, start=121, end=250, velocity=127),
        Note(pitch=64, start=0, end=162, velocity=127),
    ]

    basic_quantizer(notes_bar1, grid_16)

    expected = [
        Note(pitch=69, start=0, end=23, velocity=127),
        Note(pitch=64, start=0, end=12, velocity=127),
        Note(pitch=67, start=120, end=249, velocity=127),
        Note(pitch=64, start=0, end=162, velocity=127),
    ]

    for i in range(len(notes_bar1)):
        assert notes_bar1[i].start_ticks == expected[i].start_ticks
        assert notes_bar1[i].end_ticks == expected[i].end_ticks


def test_basic_quantizer_2(grid_8):

    notes_bar1 = [
        Note(pitch=69, start=1, end=24, velocity=127),
        Note(pitch=64, start=12, end=24, velocity=127),
        Note(pitch=67, start=121, end=250, velocity=127),
        Note(pitch=64, start=0, end=162, velocity=127),
    ]

    basic_quantizer(notes_bar1, grid_8)

    expected = [
        Note(pitch=69, start=0, end=23, velocity=127),
        Note(pitch=64, start=0, end=12, velocity=127),
        Note(pitch=67, start=144, end=273, velocity=127),
        Note(pitch=64, start=0, end=162, velocity=127),
    ]

    for i in range(len(notes_bar1)):
        assert notes_bar1[i].start_ticks == expected[i].start_ticks
        assert notes_bar1[i].end_ticks == expected[i].end_ticks


def test_advanced_quantizer_1(grid_16):

    config = QuantizerConfig(
        strength=1,
        delta_qr=12,
        type_q="positive",
    )

    notes_bar1 = [
        Note(pitch=69, start=1, end=24, velocity=127),
        Note(pitch=64, start=12, end=24, velocity=127),
        Note(pitch=67, start=121, end=250, velocity=127),
        Note(pitch=64, start=13, end=18, velocity=127),
    ]

    advanced_quantizer(notes_bar1, grid_16, config, 120, 96)

    expected = [
        Note(pitch=69, start=0, end=23, velocity=127),
        Note(pitch=64, start=12, end=24, velocity=127),
        Note(pitch=67, start=120, end=249, velocity=127),
        Note(pitch=64, start=24, end=29, velocity=127),
    ]

    for i in range(len(notes_bar1)):
        assert notes_bar1[i].start_ticks == expected[i].start_ticks
        assert notes_bar1[i].end_ticks == expected[i].end_ticks


def test_advanced_quantizer_2(grid_16):

    config = QuantizerConfig(
        strength=1,
        delta_qr=12,
        type_q=None,
    )

    notes_bar1 = [
        Note(pitch=69, start=1, end=24, velocity=127),
        Note(pitch=64, start=12, end=24, velocity=127),
        Note(pitch=67, start=121, end=250, velocity=127),
        Note(pitch=64, start=13, end=18, velocity=127),
    ]

    advanced_quantizer(notes_bar1, grid_16, config, 120, 96)

    expected = [
        Note(pitch=69, start=0, end=23, velocity=127),
        Note(pitch=64, start=0, end=12, velocity=127),
        Note(pitch=67, start=120, end=249, velocity=127),
        Note(pitch=64, start=24, end=29, velocity=127),
    ]

    for i in range(len(notes_bar1)):
        assert notes_bar1[i].start_ticks == expected[i].start_ticks
        assert notes_bar1[i].end_ticks == expected[i].end_ticks


def test_advanced_quantizer_3(grid_16):

    config = QuantizerConfig(
        strength=1,
        delta_qr=12,
        type_q=None,
    )

    notes_bar1 = [  # i dont know why but it changes when asing to a object
        Note(pitch=69, start=1, end=24, velocity=127),
        Note(pitch=64, start=12, end=24, velocity=127),
        Note(pitch=67, start=121, end=250, velocity=127),
        Note(pitch=64, start=13, end=18, velocity=127),
    ]

    advanced_quantizer(notes_bar1, grid_16, config, 120, 96)

    expected = [
        Note(pitch=69, start=0, end=23, velocity=127),
        Note(pitch=64, start=0, end=12, velocity=127),
        Note(pitch=67, start=120, end=249, velocity=127),
        Note(pitch=64, start=24, end=29, velocity=127),
    ]

    for i in range(len(notes_bar1)):
        assert notes_bar1[i].start_ticks == expected[i].start_ticks
        assert notes_bar1[i].end_ticks == expected[i].end_ticks


def test_advanced_quantizer_4(grid_16):

    config = QuantizerConfig(
        strength=0.75,
        delta_qr=12,
        type_q=None,
    )

    notes_bar1 = [
        Note(pitch=69, start=1, end=24, velocity=127),
        Note(pitch=64, start=12, end=24, velocity=127),
        Note(pitch=67, start=121, end=250, velocity=127),
        Note(pitch=64, start=30, end=50, velocity=127),
    ]

    advanced_quantizer(notes_bar1, grid_16, config, 120, 96)

    expected = [
        Note(pitch=69, start=1, end=24, velocity=127),
        Note(pitch=64, start=3, end=15, velocity=127),
        Note(pitch=67, start=121, end=250, velocity=127),
        Note(pitch=64, start=26, end=46, velocity=127),
    ]

    for i in range(len(notes_bar1)):
        assert notes_bar1[i].start_ticks == expected[i].start_ticks
        assert notes_bar1[i].end_ticks == expected[i].end_ticks

import pytest
import matplotlib.pyplot as plt

from musicaiz.plotters import Pianoroll, PianorollHTML
from musicaiz.loaders import Musa


@pytest.fixture
def midi_sample(fixture_dir):
    return fixture_dir / "tokenizers" / "mmm_tokens.mid"

@pytest.fixture
def midi_multiinstr(fixture_dir):
    return fixture_dir / "midis" / "midi_changes.mid"


def test_Pianoroll_plot_instrument(midi_sample):
    # Test case: plot one instrument
    musa_obj = Musa(midi_sample)
    plot = Pianoroll(musa_obj)
    plot.plot_instruments(
        program=30,
        bar_start=0,
        bar_end=4,
        print_measure_data=True,
        show_bar_labels=False,
        show_grid=False,
        show=False,
    )
    plt.close("all")


def test_Pianoroll_plot_instruments(midi_multiinstr):
    # Test case: plot multiple instruments
    musa_obj = Musa(midi_multiinstr)
    plot = Pianoroll(musa_obj)
    plot.plot_instruments(
        program=[48, 45, 74, 49, 49, 42, 25, 48, 21, 46, 0, 15, 72, 44],
        bar_start=0,
        bar_end=4,
        print_measure_data=True,
        show_bar_labels=False,
        show_grid=True,
        show=False,
    )
    plt.close("all")


def test_PianorollHTML_plot_instrument(midi_sample):
    musa_obj = Musa(midi_sample)
    plot = PianorollHTML(musa_obj)
    plot.plot_instruments(
        program=30,
        bar_start=0,
        bar_end=2,
        show_grid=False,
        show=False
    )
    plt.close("all")


def test_PianorollHTML_plot_instruments(midi_multiinstr):
    musa_obj = Musa(midi_multiinstr)
    plot = PianorollHTML(musa_obj)
    plot.plot_instruments(
        program=[48, 45, 74, 49, 49, 42, 25, 48, 21, 46, 0, 15, 72, 44],
        bar_start=0,
        bar_end=4,
        show_grid=False,
        show=False
    )
    plt.close("all")

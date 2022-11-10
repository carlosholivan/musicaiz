import pytest

from musicaiz.plotters import Pianoroll
from musicaiz.loaders import Musa


@pytest.fixture
def midi_sample(fixture_dir):
    return fixture_dir / "tokenizers" / "mmm_tokens.mid"


def test_Pianoroll_plot_instrument(midi_sample):
    plot = Pianoroll()
    musa_obj = Musa(midi_sample, structure="bars")
    plot.plot_instrument(
        track=musa_obj.instruments[0].notes,
        total_bars=2,
        subdivision="quarter",
        time_sig=musa_obj.time_sig.time_sig,
        print_measure_data=False,
        show_bar_labels=False
    )

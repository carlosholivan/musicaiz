import pytest

import matplotlib.pyplot as plt
import networkx as nx

from musicaiz.loaders import Musa
from musicaiz.features import (
    musa_to_graph,
    plot_graph,
)


@pytest.fixture
def midi_sample_2(fixture_dir):
    return fixture_dir / "midis" / "midi_data.mid"


def test_musa_to_graph(midi_sample_2):
    musa_obj = Musa(midi_sample_2)
    graph = musa_to_graph(musa_obj)

    # n notes must be equal to n nodes
    assert len(musa_obj.notes) == len(graph.nodes)

    # adjacency matrix
    mat = nx.attr_matrix(graph)[0]

    # n notes must be equal to n nodes
    assert len(musa_obj.notes) == mat.shape[0]


def test_plot_graph(midi_sample_2):
    musa_obj = Musa(midi_sample_2)
    graph = musa_to_graph(musa_obj)

    plot_graph(graph, show=False)
    plt.close("all")

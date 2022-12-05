import matplotlib.pyplot as plt
import networkx as nx


def musa_to_graph(musa_object) -> nx.graph:
    """Converts a Musa object into a Graph where nodes are
    the notes and edges are connections between notes.

    A similar symbolic music graph representation was introduced in:

    Jeong, D., Kwon, T., Kim, Y., & Nam, J. (2019, May).
    Graph neural network for music score data and modeling expressive piano performance.
    In International Conference on Machine Learning (pp. 3060-3070). PMLR.

    Parameters
    ----------
        musa_object

    Returns
    -------
        _type_: _description_
    """
    g = nx.Graph()
    for i, note in enumerate(musa_object.notes):
        g.add_node(i, pitch=note.pitch, velocity=note.velocity, start=note.start_ticks, end=note.end_ticks)
    nodes = list(g.nodes(data=True))

    # Add edges
    for i, node in enumerate(nodes):
        for j, next_node in enumerate(nodes):
            # if note has already finished it's not in the current subdivision
            # TODO: Check these conditions
            if i >= j:
                continue
            if node[1]["start"] >= next_node[1]["start"] and next_node[1]["end"] <= node[1]["end"]:
                g.add_edge(i, j, weight=5, color="violet")
            elif node[1]["start"] <= next_node[1]["start"] and next_node[1]["end"] <= node[1]["end"]:
                g.add_edge(i, j, weight=5, color="violet")
            if (j - i == 1) and (not g.has_edge(i, j)):
                g.add_edge(i, j, weight=5, color="red")
        if g.has_edge(i, i):
            g.remove_edge(i, i)
    return g


def plot_graph(graph: nx.graph, show: bool = False):
    """Plots a graph with matplotlib.

    Args:
        graph: nx.graph
    """
    plt.figure(figsize=(50, 10), dpi=100)
    "Plots a networkx graph."
    pos = {i: (data["start"], data["pitch"]) for i, data in list(graph.nodes(data=True))}
    if nx.get_edge_attributes(graph, 'color') == {}:
        colors = ["violet" for _ in range(len(graph.edges()))]
    else:
        colors = nx.get_edge_attributes(graph, 'color').values()
    if nx.get_edge_attributes(graph, 'weight') == {}:
        weights = [1 for _ in range(len(graph.edges()))]
    else:
        weights = nx.get_edge_attributes(graph, 'weight').values()
    nx.draw(
        graph,
        pos,
        with_labels=True,
        edge_color=colors,
        width=list(weights),
        node_color='lightblue'
    )
    if show:
        plt.show()

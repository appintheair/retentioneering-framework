from sklearn.preprocessing import MinMaxScaler
from grandalf.graphs import Vertex, Edge, Graph
from grandalf.layouts import SugiyamaLayout
import numpy as np


class _DefaultView(object):
    w, h = 1, 1


def sugiyama_layout(g):
    """
    Position nodes using Sugiyama algorithm.
    Returns dictionary of positions keyed by node.

    :param g: NetworkX graph. A position will be assigned to every node in G.

    :type g: networkx.classes.digraph.DiGraph
    :return: dict
    """
    
    # verteces (for grandalf lib)
    vertices = {node: Vertex(node) for node in g.nodes}
    # edges (for grandalf lib)
    edges = [Edge(vertices[v_from], vertices[v_to]) for v_from, v_to in g.edges]
    # build graph
    for v_name in vertices:
        vertices[v_name].view = _DefaultView()
    g = Graph(vertices.values(), edges)

    sug = SugiyamaLayout(g.C[0])
    sug.init_all(optimize=True)
    sug.draw()

    pos_names = []
    pos_xs = []
    pos_ys = []
    for v in g.C[0].sV:
        pos_names.append(v.data)
        pos_xs.append(v.view.xy[0])
        pos_ys.append(v.view.xy[1])

    scaler = MinMaxScaler()
    scaled_x = scaler.fit_transform(np.array(pos_xs).reshape(-1, 1))[:, 0]
    scaled_y = scaler.fit_transform(np.array(pos_ys).reshape(-1, 1))[:, 0]

    pos = {pos_names[i]: np.array([scaled_x[i], scaled_y[i]]) for i in range(len(pos_names))}
    return pos

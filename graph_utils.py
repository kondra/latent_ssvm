import numpy as np
import pygraphviz as pgv


def create_graph(n_nodes, edges):
    G = pgv.AGraph()
    for i in xrange(n_nodes):
        G.add_node(i)
    for i, j in edges:
        G.add_edge(i, j)
    return G


def monotonic_chains(n_nodes, edges):
    used_edges = set()
    incidence = [[] for i in xrange(n_nodes)]
    edge_index = {}
    contains = [[] for i in xrange(n_nodes)]
    chains = []

    for i, j in edges:
        assert i < j
        incidence[i].append(j)
    for i, edge in enumerate(edges):
        edge_index[(edge[0], edge[1])] = i

    start_node = 0
    k = 0
    chains.append([])

    while start_node < n_nodes:
        i = start_node
        flag1 = False
        while True:
            flag2 = False
            for j in incidence[i]:
                assert i < j
                if (i, j) not in used_edges:
                    chains[k].append(i)
                    contains[i].append(k)
                    used_edges.add((i, j))
                    i = j
                    flag1 = True
                    flag2 = True
                    break
            if not flag2 and flag1:
                chains[k].append(i)
                contains[i].append(k)
                break
            if not flag2:
                break
        if not flag1:
            start_node += 1
        else:
#            print chains[-1]
            chains.append([])
            k += 1

    _chains = []
    for chain in chains[:-1]:
        assert len(chain) > 0
        _chains.append(np.array(chain, dtype=np.int32))

    return contains, _chains, edge_index


def decompose_graph(X):
    contains = []
    chains = []
    edge_index = []

    for x in X:
        _contains, _chains, _edge_index = monotonic_chains(x[0].shape[0], x[1])
        contains.append(_contains)
        chains.append(_chains)
        edge_index.append(_edge_index)

    return contains, chains, edge_index


def decompose_grid_graph(X, get_sign=False):
    contains_node = []
    chains = []
    edge_index = []
    sign = []

    n_states = 10
    n_nodes = 400
    width = 20
    height = 20

    for k in xrange(len(X)):
        x = X[k]
        _edge_index = {}
        for i, edge in enumerate(x[1]):
            _edge_index[(edge[0], edge[1])] = i

        _sign = []
        _chains = []
        _contains = [[] for i in xrange(n_nodes)]

        #vertical
        for i in xrange(0, n_nodes, width):
            _sign.append(1)
            _chains.append(np.arange(i, i + width))
            assert _chains[-1].shape[0] == width
            tree_number = len(_chains) - 1
            for node in _chains[-1]:
                _contains[node].append(tree_number)

        #horizontal
        for i in xrange(0, width):
            _sign.append(-1)
            _chains.append(np.arange(i, n_nodes, width))
            assert _chains[-1].shape[0] == height
            tree_number = len(_chains) - 1
            for node in _chains[-1]:
                _contains[node].append(tree_number)

        sign.append(_sign)
        contains_node.append(_contains)
        chains.append(_chains)
        edge_index.append(_edge_index)

    if get_sign:
        return contains_node, chains, edge_index, sign
    else:
        return contains_node, chains, edge_index

import numpy as np
import sys
import time

from trw_utils import *
from pyqpbo import binary_general_graph
from scipy.optimize import fmin_l_bfgs_b


def trw(node_weights, edges, edge_weights,
        max_iter=100, verbose=0, tol=1e-3):

    n_nodes, n_states = node_weights.shape
    n_edges = edges.shape[0]

    y_hat = []
    lambdas = np.zeros(n_nodes)

    learning_rate = 0.1
    energy_history = []
    primal_history = []

    pairwise = []
    for k in xrange(n_states):
        y_hat.append(np.zeros(n_states))
        _pairwise = np.zeros((n_edges, 2, 2))
        for i in xrange(n_edges):
            assert -0.5 * edge_weights[i,k,k] >= 0
            _pairwise[i,1,0] = _pairwise[i,0,1] = -0.5 * edge_weights[i,k,k]
        pairwise.append(_pairwise)


    for i in xrange(n_edges):
        e1, e2 = edges[i]
        node_weights[e1,:] += 0.5 * np.diag(edge_weights[i,:,:])
        node_weights[e2,:] += 0.5 * np.diag(edge_weights[i,:,:])

    info = {}
    info['dual'] = []
    info['iteration'] = 0
    info['verbose'] = verbose

    start = time.time()
    x, f_val, d = fmin_l_bfgs_b(f, np.zeros(n_nodes),
                                args=(node_weights, pairwise, edges, info),
                                maxiter=max_iter,
                                pgtol=tol)
    stop = time.time()

    info['time'] = stop - start

    for k in xrange(n_states):
        new_unaries = np.zeros((n_nodes, 2))
        new_unaries[:,1] = node_weights[:,k] + x
        y_hat[k], energy = binary_general_graph(edges, new_unaries, pairwise[k])

    labelling = np.zeros((n_nodes, n_states))
    for k in xrange(n_states):
        labelling[:,k] = y_hat[k]
    labelling = labelling / np.sum(labelling, axis=1, keepdims=True)

    return labelling, info


def f(x, node_weights, pairwise, edges, info):
    n_nodes, n_states = node_weights.shape

    dual = 0
    dlambda = np.zeros(n_nodes)

    for k in xrange(n_states):
        new_unaries = np.zeros((n_nodes, 2))
        new_unaries[:,1] = node_weights[:,k] + x
        y_hat, energy = binary_general_graph(edges, new_unaries, pairwise[k])
        dual += 0.5 * energy
        dlambda += y_hat
    dlambda -= 1

    dual -= np.sum(x)

    info['iteration'] += 1
    info['dual'].append(dual)

    if info['verbose']:
        print dual
        sys.stdout.flush()

    return -dual, -dlambda



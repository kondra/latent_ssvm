import numpy as np
import sys

from trw_utils import *
from heterogenous_crf import inference_gco
from pyqpbo import binary_general_graph
from scipy.optimize import fmin_l_bfgs_b


def trw(node_weights, edges, edge_weights, y,
        max_iter=100, verbose=0, tol=1e-3,
        get_energy=None):

    n_nodes, n_states = node_weights.shape
    n_edges = edges.shape[0]

    y_hat = []
    lambdas = np.zeros(n_nodes)
    mu = np.zeros((n_nodes, n_states))

    learning_rate = 0.1
    energy_history = []
    primal_history = []

    pairwise = []
    for k in xrange(n_states):
        y_hat.append(np.zeros(n_states))
        _pairwise = np.zeros((n_edges, 2, 2))
        for i in xrange(n_edges):
#            assert 0.5 * edge_weights[i,k,k] >= 0
            _pairwise[i,1,0] = _pairwise[i,0,1] = -0.5 * edge_weights[i,k,k]
        pairwise.append(_pairwise)

    for i in xrange(n_edges):
        e1, e2 = edges[i]
        node_weights[e1,:] += 0.5 * np.diag(edge_weights[i,:,:])
        node_weights[e2,:] += 0.5 * np.diag(edge_weights[i,:,:])

    for iteration in xrange(max_iter):
        dmu = np.zeros((n_nodes, n_states))

        unaries = node_weights + mu

        x, f_val, d = fmin_l_bfgs_b(f, np.zeros(n_nodes),
                                    args=(unaries, pairwise, edges),
                                    maxiter=50,
                                    pgtol=1e-5)
        
        E = np.sum(x)
        for k in xrange(n_states):
            new_unaries = np.zeros((n_nodes, 2))
            new_unaries[:,1] = unaries[:,k] + x
            y_hat[k], energy = binary_general_graph(edges, new_unaries, pairwise[k])
            E -= 0.5*energy
            dmu[:,k] -= y_hat[k]

        y_hat_kappa, energy = optimize_kappa(y, mu, 1, n_nodes, n_states)
        E += energy
        dmu[np.ogrid[:dmu.shape[0]], y_hat_kappa] += 1

        mu -= learning_rate * dmu

        energy_history.append(E)

        lambda_sum = np.zeros((n_nodes, n_states))
        for k in xrange(n_states):
            lambda_sum[:,k] = y_hat[k]
        lambda_sum = lambda_sum / np.sum(lambda_sum, axis=1, keepdims=True)

        if get_energy is not None:
            primal = get_energy(get_labelling(lambda_sum))
            primal_history.append(primal)
        else:
            primal = 0

        if iteration:
            learning_rate = 1. / np.sqrt(iteration)

        if verbose:
            print 'Iteration {}: energy={}, primal={}'.format(iteration, E, primal)

        if iteration > 0 and np.abs(E - energy_history[-2]) < tol:
            if verbose:
                print 'Converged'
            break

    info = {'primal': primal_history,
            'dual': energy_history,
            'iteration': iteration}

    return lambda_sum, y_hat_kappa, info


def f(x, node_weights, pairwise, edges):
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
    #print dual

    return -dual, -dlambda

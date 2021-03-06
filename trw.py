import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from graph_utils import decompose_graph, decompose_grid_graph
from trw_utils import *


def trw(node_weights, edges, edge_weights,
        max_iter=100, verbose=0, tol=1e-3,
        strategy='sqrt',
        r0=1.5, r1=0.5, gamma=0.1):

    assert strategy in ['best-dual', 'best-primal', 'sqrt', 'linear']

    result = decompose_grid_graph([(node_weights, edges, edge_weights)])
    contains_node, chains, edge_index = result[0][0], result[1][0], result[2][0]

    n_nodes, n_states = node_weights.shape

    y_hat = []
    lambdas = []
    multiplier = []

    for p in xrange(n_nodes):
        multiplier.append(1.0 / len(contains_node[p]))
    for chain in chains:
        lambdas.append(np.zeros((len(chain), n_states)))
        y_hat.append(np.zeros(len(chain)))

    multiplier = np.array(multiplier)
    multiplier.shape = (n_nodes, 1)

    delta = 1.
    learning_rate = 0.1
    dual_history = []
    primal_history = []

    best_dual = np.inf
    best_primal = -np.inf

    for iteration in xrange(max_iter):
        dual = 0.0
        unaries = node_weights * multiplier

        for i, chain in enumerate(chains):
            y_hat[i], e = optimize_chain(chain,
                                         lambdas[i] + unaries[chain,:],
                                         edge_weights,
                                         edge_index)

            dual += e

        lambda_sum = np.zeros((n_nodes, n_states), dtype=np.float64)
        for p in xrange(n_nodes):
            for i in contains_node[p]:
                pos = np.where(chains[i] == p)[0][0]
                lambda_sum[p, y_hat[i][pos]] += multiplier[p]

        p_norm = 0.0
        for i in xrange(len(chains)):
            N = lambdas[i].shape[0]

            dlambda = lambda_sum[chains[i],:].copy()
            dlambda[np.ogrid[:N], y_hat[i]] -= 1

            p_norm += np.sum(dlambda ** 2)

            lambdas[i] += learning_rate * dlambda

        primal = compute_energy(get_labelling(lambda_sum), unaries, edge_weights, edges)
        primal_history.append(primal)
        dual_history.append(dual)

        if iteration and (np.abs(dual - dual_history[-2]) < tol or p_norm < tol):
            if verbose:
                print 'Converged'
            break

        if iteration:
            if strategy == 'sqrt':
                learning_rate = 1. / np.sqrt(iteration)
            elif strategy == 'linear':
                learning_rate = 1. / iteration
            elif strategy == 'best-dual':
                best_dual = min(best_dual, dual)
                approx = best_dual - delta
                if dual <= dual_history[-2]:
                    delta *= r0
                else:
                    delta = max(r1 * delta, 1e-4)
                learning_rate = gamma * (dual - approx) / p_norm
            elif strategy == 'best-primal':
                best_primal = max(best_primal, primal)
                learning_rate = gamma * (dual - best_primal) / p_norm


        if verbose:
            print 'iteration {}: dual energy = {}'.format(iteration, dual)

    info = {}
    info['dual_energy'] = dual_history
    info['primal_energy'] = primal_history

    return lambda_sum, info


def trw_unconstr(node_weights, edges, edge_weights,
                 max_iter=100, verbose=0, tol=1e-3,
                 strategy='sqrt',
                 r0=1.5, r1=0.5, gamma=0.1):

    assert strategy in ['best-dual', 'best-primal', 'sqrt', 'linear']

    result = decompose_grid_graph([(node_weights, edges, edge_weights)], get_sign=True)
    contains_node, chains, edge_index, sign = result[0][0], result[1][0], result[2][0], result[3][0]

    n_nodes, n_states = node_weights.shape

    y_hat = []
    lambdas = np.zeros((n_nodes, n_states))
    multiplier = []

    for p in xrange(n_nodes):
        multiplier.append(1.0 / len(contains_node[p]))
        assert len(contains_node[p]) == 2
    for chain in chains:
        y_hat.append(np.zeros(len(chain)))

    multiplier = np.array(multiplier)
    multiplier.shape = (n_nodes, 1)

    delta = 1.
    learning_rate = 0.1
    dual_history = []
    primal_history = []

    best_dual = np.inf
    best_primal = -np.inf

    for iteration in xrange(max_iter):
        dual = 0.0
        unaries = node_weights * multiplier

        for i, chain in enumerate(chains):
            y_hat[i], e = optimize_chain(chain,
                                         sign[i] * lambdas[chain,:] + unaries[chain,:],
                                         edge_weights,
                                         edge_index)

            dual += e

        p_norm = 0.0
        lambda_sum = np.zeros((n_nodes, n_states), dtype=np.float64)
        for p in xrange(n_nodes):
            dlambda = np.zeros(n_states)
            for i in contains_node[p]:
                pos = np.where(chains[i] == p)[0][0]
                lambda_sum[p, y_hat[i][pos]] += multiplier[p]
                dlambda[y_hat[i][pos]] += sign[i]
            p_norm += np.sum(dlambda ** 2)
            lambdas[p] -= learning_rate * dlambda

        primal = compute_energy(get_labelling(lambda_sum), unaries, edge_weights, edges)
        primal_history.append(primal)
        dual_history.append(dual)

        if iteration and (np.abs(dual - dual_history[-2]) < tol or p_norm < tol):
            if verbose:
                print 'Converged'
            break

        if iteration:
            if strategy == 'sqrt':
                learning_rate = 1. / np.sqrt(iteration)
            elif strategy == 'linear':
                learning_rate = 1. / iteration
            elif strategy == 'best-dual':
                best_dual = min(best_dual, dual)
                approx = best_dual - delta
                if dual <= dual_history[-2]:
                    delta *= r0
                else:
                    delta = max(r1 * delta, 1e-4)
                learning_rate = gamma * (dual - approx) / p_norm
            elif strategy == 'best-primal':
                best_primal = max(best_primal, primal)
                learning_rate = gamma * (dual - best_primal) / p_norm


        if verbose:
            print 'iteration {}: dual energy = {}'.format(iteration, dual)

    info = {}
    info['dual_energy'] = dual_history
    info['primal_energy'] = primal_history

    return lambda_sum, info


def f(x, node_weights, multiplier, chains, edge_weights, edge_index, sign, y_hat, contains_node, history):
    dual = 0
    n_nodes, n_states = node_weights.shape
    unaries = node_weights * multiplier
    lambdas = x.reshape((400, 10))

    for i, chain in enumerate(chains):
        y_hat[i], e = optimize_chain(chain,
                                     sign[i] * lambdas[chain,:] + unaries[chain,:],
                                     edge_weights,
                                     edge_index)

        dual += e

    dlambda = np.zeros((n_nodes, n_states), dtype=np.float64)
    for p in xrange(n_nodes):
        for i in contains_node[p]:
            pos = np.where(chains[i] == p)[0][0]
            dlambda[p][y_hat[i][pos]] += sign[i]

    history.append(dual)

    return dual, dlambda.reshape(4000)


def trw_lbfgs(node_weights, edges, edge_weights,
              max_iter=100, verbose=1, tol=1e-3):

    result = decompose_grid_graph([(node_weights, edges, edge_weights)], get_sign=True)
    contains_node, chains, edge_index, sign = result[0][0], result[1][0], result[2][0], result[3][0]

    n_nodes, n_states = node_weights.shape

    y_hat = []
    lambdas = np.zeros((n_nodes, n_states))
    multiplier = []

    for p in xrange(n_nodes):
        multiplier.append(1.0 / len(contains_node[p]))
        assert len(contains_node[p]) == 2
    for chain in chains:
        y_hat.append(np.zeros(len(chain)))

    multiplier = np.array(multiplier)
    multiplier.shape = (n_nodes, 1)

    history = []
    x, f_val, d = fmin_l_bfgs_b(f, np.zeros((n_nodes, n_states)),
                                args=(node_weights, multiplier, chains, edge_weights, edge_index, sign, y_hat, contains_node, history),
                                maxiter=max_iter,
                                disp=verbose,
                                pgtol=tol)

    lambdas = x.reshape((400, 10))
    unaries = node_weights * multiplier
    for i, chain in enumerate(chains):
        y_hat[i], e = optimize_chain(chain,
                                     sign[i] * lambdas[chain,:] + unaries[chain,:],
                                     edge_weights,
                                     edge_index)


    lambda_sum = np.zeros((n_nodes, n_states), dtype=np.float64)
    for p in xrange(n_nodes):
        for i in contains_node[p]:
            pos = np.where(chains[i] == p)[0][0]
            lambda_sum[p, y_hat[i][pos]] += multiplier[p]

    info = {}
    info['x'] = x
    info['f'] = f_val
    info['d'] = d
    info['history'] = history

    return lambda_sum, info



import numpy as np

from graph_utils import decompose_graph, decompose_grid_graph
from trw_utils import *


def trw(node_weights, edges, edge_weights, y,
        max_iter=100, verbose=0, tol=1e-3,
        update_mu=50, get_energy=None):

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

    mu = np.zeros((n_nodes, n_states))

    learning_rate = 0.1
    energy_history = []
    primal_history = []

    for iteration in xrange(max_iter):
        dmu = np.zeros((n_nodes, n_states))
        unaries = (node_weights - mu) * multiplier

        inner_energy = []
        for inner in xrange(update_mu):
            E = 0
            for i, chain in enumerate(chains):
                y_hat[i], energy = optimize_chain(chain,
                                                  lambdas[i] + unaries[chain,:],
                                                  edge_weights,
                                                  edge_index)

                E += energy

            inner_energy.append(E)

            lambda_sum = np.zeros((n_nodes, n_states), dtype=np.float64)
            for p in xrange(n_nodes):
                for i in contains_node[p]:
                    pos = np.where(chains[i] == p)[0][0]
                    lambda_sum[p, y_hat[i][pos]] += multiplier[p]

            for i in xrange(len(chains)):
                N = lambdas[i].shape[0]

                lambdas[i][np.ogrid[:N], y_hat[i]] -= learning_rate
                lambdas[i] += learning_rate * lambda_sum[chains[i],:]

            if inner > 0 and np.abs(inner_energy[-2] - E) < 1e-2:
                break

        E = inner_energy[-1]

        y_hat_kappa, energy = optimize_kappa(y, mu, 1, n_nodes, n_states)
        E += energy

        for i in xrange(len(chains)):
            dmu[chains[i], y_hat[i]] -= multiplier[chains[i]].flatten()
        dmu[np.ogrid[:dmu.shape[0]], y_hat_kappa] += 1

        mu -= learning_rate * dmu

        energy_history.append(E)

        if get_energy is not None:
            primal = get_energy(get_labelling(lambda_sum))
            primal_history.append(primal)

        if iteration:
            learning_rate = 1. / np.sqrt(iteration)

        if verbose:
            print 'Iteration {}: inner={} energy={}'.format(iteration, inner, E)

        if iteration > 0 and np.abs(E - energy_history[-2]) < tol:
            if verbose:
                print 'Converged'
            break

    info = {'primal': primal_history,
            'dual': energy_history,
            'iteration': iteration}

    return lambda_sum, y_hat_kappa, info


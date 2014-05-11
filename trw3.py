import numpy as np

from graph_utils import decompose_graph, decompose_grid_graph
from trw_utils import optimize_chain, optimize_kappa


def trw(node_weights, edges, edge_weights, y,
        max_iter=100, verbose=0, tol=1e-3):

    result = decompose_grid_graph([(node_weights, edges, edge_weights)])
    contains_node, chains, edge_index = result[0][0], result[1][0], result[2][0]

    n_nodes, n_states = node_weights.shape

    y_hat = []
    lambdas = []
    multiplier = []

    for p in xrange(n_nodes):
        multiplier.append(1.0 / (len(contains_node[p]) + 1))
    for chain in chains:
        lambdas.append(np.zeros((len(chain), n_states)))
        y_hat.append(np.zeros(len(chain)))

    multiplier = np.array(multiplier)
    multiplier.shape = (n_nodes, 1)

    mu = np.zeros((n_nodes, n_states))

    learning_rate = 0.1
    energy_history = []

    for iteration in xrange(max_iter):
        E = 0
        unaries = node_weights.copy()
        for label in xrange(n_states):
            if label not in y.weak:
                unaries[:,label] += y.weights
        unaries *= multiplier

        for i, chain in enumerate(chains):
            y_hat[i], energy = optimize_chain(chain,
                                              lambdas[i] + unaries[chain,:],
                                              edge_weights,
                                              edge_index)

            E += energy

        y_hat_kappa, energy = optimize_kappa(y, mu + unaries, 1, n_nodes, n_states, augment=False)
        E += energy

        lambda_sum = np.zeros((n_nodes, n_states), dtype=np.float64)
        for p in xrange(n_nodes):
            assert len(contains_node[p]) == 2
            for i in contains_node[p]:
                pos = np.where(chains[i] == p)[0][0]
                lambda_sum[p, y_hat[i][pos]] += multiplier[p]

        lambda_sum[np.ogrid[:n_nodes], y_hat_kappa] += multiplier.flatten()

        for i in xrange(len(chains)):
            N = lambdas[i].shape[0]

            lambdas[i][np.ogrid[:N], y_hat[i]] -= learning_rate
            lambdas[i] += learning_rate * lambda_sum[chains[i],:]

        mu[np.ogrid[:n_nodes], y_hat_kappa] -= learning_rate
        mu += learning_rate * lambda_sum

        test_l = np.zeros((n_nodes, n_states))
        for p in xrange(n_nodes):
            for i in contains_node[p]:
                pos = np.where(chains[i] == p)[0][0]
                test_l[p, :] += lambdas[i][pos,:]
        test_l += mu

        assert np.sum(test_l) < 1e-10

        energy_history.append(E)

        if iteration:
            learning_rate = 1. / np.sqrt(iteration)

        if verbose:
            print 'Iteration {}: energy {}'.format(iteration, E)

        if iteration > 300 and np.abs(E - energy_history[-2]) < tol:
            if verbose:
                print 'Converged'
            break

    return lambda_sum, y_hat_kappa, energy_history, iteration


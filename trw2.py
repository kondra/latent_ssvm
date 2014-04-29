import numpy as np

from graph_utils import decompose_graph, decompose_grid_graph
from trw_utils import optimize_chain, optimize_kappa


def trw(node_weights, edges, edge_weights, y,
        max_iter=100, verbose=0, tol=1e-3,
        relaxed=False):

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

    for iteration in xrange(max_iter):
        E = 0
        dmu = np.zeros((n_nodes, n_states))
        unaries = (node_weights - mu) * multiplier

        for i, chain in enumerate(chains):
            y_hat[i], energy = optimize_chain(chain,
                                              lambdas[i] + unaries[chain,:],
                                              edge_weights,
                                              edge_index)

            dmu[chains[i], y_hat[i]] -= multiplier[chains[i]].flatten()
            E += energy

        y_hat_kappa, energy = optimize_kappa(y, mu, 1, n_nodes, n_states)
        E += energy

        dmu[np.ogrid[:dmu.shape[0]], y_hat_kappa] += 1

        mu -= learning_rate * dmu

        lambda_sum = np.zeros((n_nodes, n_states), dtype=np.float64)
        for p in xrange(n_nodes):
            assert len(contains_node[p]) == 2
            for i in contains_node[p]:
                pos = np.where(chains[i] == p)[0][0]
                lambda_sum[p, y_hat[i][pos]] += multiplier[p]

        for i in xrange(len(chains)):
            N = lambdas[i].shape[0]

            lambdas[i][np.ogrid[:N], y_hat[i]] -= learning_rate
            lambdas[i] += learning_rate * lambda_sum[chains[i],:]

        energy_history.append(E)

        if iteration:
            learning_rate = 1. / np.sqrt(iteration)

        if verbose:
            print 'Iteration {}: energy {}'.format(iteration, E)

        if iteration > 300 and np.abs(E - energy_history[-2]) < tol:
            if verbose:
                print 'Converged'
            break

    if relaxed:
        unaries = np.zeros((n_nodes, n_states), dtype=np.float64)
        for p in xrange(n_nodes):
            mult = 1.0 / (len(contains_node[p]) + 1)
            for i in contains_node[p]:
                pos = np.where(chains[i] == p)[0][0]
                unaries[p, y_hat[i][pos]] += mult
            unaries[p, y_hat_kappa[p]] += mult
        return get_relaxed(unaries, edges), energy_history, iteration
    else:
        return lambda_sum, y_hat_kappa, energy_history, iteration


def get_relaxed(unaries, edges):
    n_edges = edges.shape[0]
    n_states = unaries.shape[1]

    pairwise = np.zeros((n_edges, n_states ** 2))
    for i in xrange(n_edges):
        (e1, e2) = edges[i]
        pairwise[i,:] = np.kron(unaries[e1,:], unaries[e2,:])

    return (unaries, pairwise)


import numpy as np

from graph_utils import decompose_graph, decompose_grid_graph
from trw_utils import optimize_chain, optimize_kappa
from heterogenous_crf import inference_gco

# gco instead of first argument

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
        unaries = node_weights - mu

        y_hat_gco, energy = inference_gco(unaries, edge_weights, edges,
                                          n_iter=5, return_energy=True)
        E -= energy

        y_hat_kappa, energy = optimize_kappa(y, mu, 1, n_nodes, n_states)
        E += energy

        dmu[np.ogrid[:dmu.shape[0]], y_hat_gco] -= 1
        dmu[np.ogrid[:dmu.shape[0]], y_hat_kappa] += 1

        mu -= learning_rate * dmu

        energy_history.append(E)

        if iteration:
            learning_rate = 1. / np.sqrt(iteration)

        if verbose:
            print 'Iteration {}: energy {}'.format(iteration, E)

        if iteration and np.abs(E - energy_history[-2]) < tol:
            if verbose:
                print 'Converged'
            break

    return y_hat_gco, y_hat_kappa, energy_history, iteration

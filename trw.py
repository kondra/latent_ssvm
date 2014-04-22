import numpy as np

from graph_utils import decompose_graph, decompose_grid_graph
from trw_utils import optimize_chain, optimize_kappa

def trw(node_weights, edges, edge_weights,
        max_iter=100, verbose=0, tol=1e-3):

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

    learning_rate = 0.1
    energy_history = []

    adaptive_rate = False

#    alpha = 1.
#    gamma = 1.
#    delta = 1.
#    rho0 = 1.5
#    rho1 = 0.1
#    eps = 0.001
#    best_dual = -np.inf
    for iteration in xrange(max_iter):
        energy = 0.0
        unaries = node_weights * multiplier

        for i, chain in enumerate(chains):
            y_hat[i], e = optimize_chain(chain,
                                         lambdas[i] + unaries[chain,:],
                                         edge_weights,
                                         edge_index)

            energy += e

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

        if iteration:
            learning_rate = 1. / np.sqrt(iteration)

        energy_history.append(energy)

        if verbose:
            print 'iteration {}: energy = {}'.format(iteration, energy)

        if iteration and np.abs(energy - energy_history[-2]) < tol:
            if verbose:
                print 'Converged'
            break

#        if energy > best_dual:
#            best_dual = energy
#        if iteration:
#            if smart_step:
#                grad_norm = np.sum((mean_x * 2) ** 2)
#                if energy > prev_energy:
#                    delta *= rho0
#                else:
#                    delta = max(rho1 * delta, eps)
#                alpha = (best_dual + delta) - energy
#                alpha /= grad_norm
#                alpha *= gamma
#            else:
#                alpha = 2 / np.sqrt(iteration)

    return lambda_sum, energy_history


import numpy as np
import itertools

from graph_utils import decompose_graph, decompose_grid_graph


def optimize_chain(chain, unary_cost, pairwise_cost, edge_index):
    n_nodes = chain.shape[0]
    n_states = unary_cost.shape[1]

    p = np.zeros((n_states, n_nodes))
    track = np.zeros((n_states, n_nodes), dtype=np.int32)
    p[:,0] = unary_cost[0,:]
    track[:,0] = -1

    for i in xrange(1, n_nodes):
        p[:,i] = unary_cost[i,:]
        p_cost = pairwise_cost[edge_index[(chain[i - 1], chain[i])]]
        for k in xrange(n_states):
            p[k,i] += np.max(p[:,i - 1] + p_cost[:,k])
            track[k,i] = np.argmax(p[:,i - 1] + p_cost[:,k])

    x = np.zeros(n_nodes, dtype=np.int32)
    current = np.argmax(p[:,n_nodes - 1])
    for i in xrange(n_nodes - 1, -1, -1):
        x[i] = current
        current = track[current,i]

    return x, np.max(p[:,n_nodes - 1])


def optimize_kappa(y, mu, alpha, n_nodes, n_states):
    unaries = mu.copy()

    c = np.sum(y.weights) / float(n_states)
    c *= alpha

    for label in xrange(n_states):
        if label not in y.weak:
            unaries[:,label] += y.weights * alpha

    max_energy = -np.Inf
    best_y = None
    all_labels = set([l for l in xrange(n_states)])
    gt_labels = set(y.weak)

    for k in xrange(len(y.weak) + 1):
        for l in itertools.combinations(y.weak, k):
            labels = list(all_labels - set(l))
            l = list(l)
            t_unaries = unaries.copy()
            t_unaries[:, l] = -np.Inf
            y_hat = np.argmax(t_unaries, axis=1)
            energy = np.sum(np.max(unaries[:,labels], axis=1))
            energy2 = np.sum(t_unaries[np.ogrid[:unaries.shape[0]],y_hat])
            present_labels = set(np.unique(y_hat))
            if len(present_labels.intersection(gt_labels)):
                energy -= c
            if energy > max_energy:
                max_energy = energy
                best_y = y_hat

    return best_y, max_energy


def get_labelling(relaxed_x):
    n_nodes = relaxed_x.shape[0]
    x = np.zeros(n_nodes)
    for i in xrange(n_nodes):
        x[i] = np.where(relaxed_x[i,:])[0][0]
    return x.astype(np.int32)


def trw(node_weights, edges, edge_weights, y,
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

    return lambda_sum, y_hat_kappa, energy_history, iteration


import numpy as np
import itertools


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


def optimize_kappa(y, unaries, alpha, n_nodes, n_states, augment=True):
    unaries = unaries.copy()

    c = np.sum(y.weights) / float(n_states)
    c *= alpha

    if augment:
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
            energy -= c * len(present_labels.intersection(gt_labels))
            if energy > max_energy:
                max_energy = energy
                best_y = y_hat

    return best_y, max_energy


def get_labelling(relaxed_y):
    n_nodes = relaxed_y.shape[0]
    y = np.zeros(n_nodes)
    for i in xrange(n_nodes):
        y[i] = np.where(relaxed_y[i,:])[0][0]
    return y.astype(np.int32)


def compute_energy(y, unaries, pairwise, edges):
    energy = 0.0
    for i, (u, v) in enumerate(edges):
        energy += pairwise[i, y[u], y[v]]
    energy += np.sum(unaries[np.ogrid[:y.shape[0]],y])
    return energy

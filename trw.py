import numpy as np


def optimize_chain(chain, unary_cost, pairwise_cost,
                   return_energy=True):
    n_states = unary_cost.shape[1]
    n_nodes = chain.shape[0]

    p = np.zeros((n_states, n_nodes))
    track = np.zeros((n_states, n_nodes), dtype=np.int32)
    p[:,0] = unary_cost[0,:]
    track[:,0] = -1

    for i in xrange(1, n_nodes):
        p[:,i] = unary_cost[i,:]
        p_cost = pairwise_cost[(chain[i - 1], chain[i])]
        for k in xrange(n_states):
            p[k,i] += np.min(p[:,i - 1] + p_cost[:,k])
            track[k,i] = np.argmin(p[:,i - 1] + p_cost[:,k])

    x = np.zeros(n_nodes, dtype=np.int32)
    current = np.argmin(p[:,n_nodes - 1])
    for i in xrange(n_nodes - 1, -1, -1):
        x[i] = current
        current = track[current,i]

    return x, np.min(p[:,n_nodes - 1])


def get_labelling(mean_x):
    n_nodes = mean_x.shape[0]
    x = np.zeros(n_nodes)
    for i in xrange(n_nodes):
        x[i] = np.where(mean_x[i,:])[0][0]
    return x


def trw(node_weights, edges, edge_weights,
        max_iter=100, relaxed=False, verbose=0,
        smart_step=False, return_energy_history=False):
    n_nodes, n_states = node_weights.shape
    n_edges = edges.shape[0]

    pairwise_cost = {}
    for i in xrange(n_edges):
        pairwise_cost[(edges[i,0], edges[i,1])] = edge_weights[i,:]

    x_opt = []
    chains = []
    unaries = []
    dlambda = []
    trees = {k : [] for k in xrange(n_nodes)}
    decode = {}
    for i in xrange(1, 400, 20):
        chains.append(np.arange(i, i + 20) - 1)
        x_opt.append(np.zeros(20))
        dlambda.append(np.zeros((20, n_states)))
        unaries.append(node_weights[chains[-1],:] * 0.5)
        tree_number = len(chains) - 1
        for j, k in enumerate(chains[-1]):
            decode[(k,tree_number)] = j
            trees[k].append(tree_number)
    for i in xrange(1, 21):
        chains.append(np.arange(i, 401, 20) - 1)
        x_opt.append(np.zeros(20))
        dlambda.append(np.zeros((20, n_states)))
        unaries.append(node_weights[chains[-1],:] * 0.5)
        tree_number = len(chains) - 1
        for j, k in enumerate(chains[-1]):
            decode[(k,tree_number)] = j
            trees[k].append(tree_number)

    energy_history = []
    alpha = 1.
    gamma = 1.
    delta = 1.
    rho0 = 1.5
    rho1 = 0.1
    eps = 0.001
    best_dual = -np.inf
    for iteration in xrange(max_iter):
        energy = 0.
        for i, chain in enumerate(chains):
            x_opt[i], e = optimize_chain(chain, unaries[i], pairwise_cost)
            energy += e
        
        mean_x = np.zeros((n_nodes, n_states), dtype=np.float64)
        for i in xrange(n_nodes):
            k = trees[i][0]
            mean_x[i][x_opt[k][decode[(i,k)]]] += 1
            k = trees[i][1]
            mean_x[i][x_opt[k][decode[(i,k)]]] += 1
        mean_x *= 0.5

        for i, chain in enumerate(chains):
            x = np.zeros((20, n_states), dtype=np.float64)
            x[np.arange(20), x_opt[i]] = 1
            dlambda[i] = alpha * (x - mean_x[chain,:])
            unaries[i] += dlambda[i]

        if verbose:
            print 'iteration {}: energy = {}'.format(iteration, energy)

        if energy > best_dual:
            best_dual = energy
        if iteration:
            if smart_step:
                grad_norm = np.sum((mean_x * 2) ** 2)
                if energy > prev_energy:
                    delta *= rho0
                else:
                    delta = max(rho1 * delta, eps)
                alpha = (best_dual + delta) - energy
                alpha /= grad_norm
                alpha *= gamma
            else:
                alpha = 2 / np.sqrt(iteration)

        prev_energy = energy
        energy_history.append(energy)

    if verbose:
        print 'number of unconsistent labels: {}'.format(np.sum(mean_x == 0.5)/2)

    if relaxed:
        return mean_x
    else:
        if return_energy_history:
            return get_labelling(mean_x), energy_history
        else:
            return get_labelling(mean_x)

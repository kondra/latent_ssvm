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


def get_labelling(relaxed_x):
    n_nodes = relaxed_x.shape[0]
    x = np.zeros(n_nodes)
    for i in xrange(n_nodes):
        x[i] = np.where(relaxed_x[i,:])[0][0]
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
    lambdas = []
    trees = {k : [] for k in xrange(n_nodes)}
    decode = {}
    for i in xrange(1, 400, 20):
        chains.append(np.arange(i, i + 20) - 1)
        x_opt.append(np.zeros(20))
        lambdas.append(np.zeros((20, n_states)))
        unaries.append(node_weights[chains[-1],:] * 0.5)
        tree_number = len(chains) - 1
        for j, k in enumerate(chains[-1]):
            decode[(k,tree_number)] = j
            trees[k].append(tree_number)
    for i in xrange(1, 21):
        chains.append(np.arange(i, 401, 20) - 1)
        x_opt.append(np.zeros(20))
        lambdas.append(np.zeros((20, n_states)))
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
            x_opt[i], e = optimize_chain(chain, unaries[i]+lambdas[i], pairwise_cost)
            energy += e
        
        mean_x = np.zeros((n_nodes, n_states), dtype=np.float64)
        for i in xrange(n_nodes):
            for k in trees[i]:
                mean_x[i][x_opt[k][decode[(i,k)]]] += 1
        mean_x *= 0.5

        for i, chain in enumerate(chains):
            x = np.zeros((20, n_states), dtype=np.float64)
            x[np.arange(20), x_opt[i]] = 1
            lambdas[i] += alpha * (x - mean_x[chain,:])

        if verbose > 2:
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

    if verbose > 0:
        print 'number of unconsistent labels: {}'.format(np.sum(mean_x == 0.5)/2)

    if relaxed:
        return mean_x
    else:
        if return_energy_history:
            return get_labelling(mean_x), energy_history
        else:
            return get_labelling(mean_x)


class TRW(object):
    def __init__(self, n_nodes, n_states, max_iter=1, relaxed=False, verbose=0,
                 smart_step=False):
        self.max_iter = max_iter
        self.iteration = 0
        self.relaxed = relaxed
        self.verbose = verbose
        self.smart_step = smart_step

        self.x_opt = []
        self.chains = []
        self.unaries = []
        self.lambdas = []
        self.trees = {k : [] for k in xrange(n_nodes)}
        self.decode = {}
        for i in xrange(1, 400, 20):
            self.chains.append(np.arange(i, i + 20) - 1)
            self.x_opt.append(np.zeros(20))
            self.lambdas.append(np.zeros((20, n_states)))
            tree_number = len(self.chains) - 1
            for j, k in enumerate(self.chains[-1]):
                self.decode[(k,tree_number)] = j
                self.trees[k].append(tree_number)
        for i in xrange(1, 21):
            self.chains.append(np.arange(i, 401, 20) - 1)
            self.x_opt.append(np.zeros(20))
            self.lambdas.append(np.zeros((20, n_states)))
            tree_number = len(self.chains) - 1
            for j, k in enumerate(self.chains[-1]):
                self.decode[(k,tree_number)] = j
                self.trees[k].append(tree_number)
    
        self.energy_history = []
        self.alpha = 1.
        self.gamma = 1.
        self.delta = 1.
        self.rho0 = 1.5
        self.rho1 = 0.1
        self.eps = 0.001
        self.best_dual = -np.inf
        self.inconsistent = []

    def do_step(self, node_weights, edges, pairwise_cost):
        n_nodes, n_states = node_weights.shape
        for iteration in xrange(self.max_iter):
            energy = 0.
            for i, chain in enumerate(self.chains):
                self.x_opt[i], e = optimize_chain(chain, 0.5 * node_weights[chain] + self.lambdas[i], pairwise_cost)
                energy += e
            
            mean_x = np.zeros((n_nodes, n_states), dtype=np.float64)
            for i in xrange(n_nodes):
                for k in self.trees[i]:
                    mean_x[i][self.x_opt[k][self.decode[(i,k)]]] += 1
            mean_x *= 0.5
    
            for i, chain in enumerate(self.chains):
                x = np.zeros((20, n_states), dtype=np.float64)
                x[np.arange(20), self.x_opt[i]] = 1
                self.lambdas[i] += self.alpha * (x - mean_x[chain,:])
    
            if self.verbose:
                print 'iteration {}: energy = {}'.format(iteration, energy)
    
            if energy > self.best_dual:
                self.best_dual = energy
            if iteration:
                if self.smart_step:
                    grad_norm = np.sum((mean_x * 2) ** 2)
                    if energy > self.prev_energy:
                        self.delta *= self.rho0
                    else:
                        self.delta = max(self.rho1 * self.delta, self.eps)
                    self.alpha = (self.best_dual + self.delta) - energy
                    self.alpha /= grad_norm
                    self.alpha *= self.gamma
                else:
                    self.alpha = 2 / np.sqrt(iteration)
    
            self.prev_energy = energy
            self.energy_history.append(energy)

            self.iteration += 1

        self.inconsistent.append(np.sum(mean_x == 0.5)/2)
    
        if self.verbose:
            print 'number of unconsistent labels: {}'.format(self.inconsistent[-1])

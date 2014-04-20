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


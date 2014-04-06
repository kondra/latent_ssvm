import logging
import time
import numpy as np

from sklearn.utils.extmath import safe_sparse_dot


def optimize_chain(chain, unary_cost, pairwise_cost, edge_index,
                   return_energy=True):
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
            p[k,i] += np.min(p[:,i - 1] + p_cost[:,k])
            track[k,i] = np.argmin(p[:,i - 1] + p_cost[:,k])

    x = np.zeros(n_nodes, dtype=np.int32)
    current = np.argmin(p[:,n_nodes - 1])
    for i in xrange(n_nodes - 1, -1, -1):
        x[i] = current
        current = track[current,i]

    return x, np.min(p[:,n_nodes - 1])


class Over(object):
    def __init__(self, n_states, n_features, n_edge_features,
                 mu=1, verbose=0, max_iter=200):
        self.n_states = n_states
        self.n_features = n_features
        self.n_edge_features = n_edge_features
        self.mu = mu
        self.verbose = verbose
        self.max_iter = max_iter
        self.size_w = (self.n_states * self.n_features +
                       self.n_states * self.n_edge_features)
        self.logger = logging.getLogger(__name__)

    def _get_edges(self, x):
        return x[1]

    def _get_features(self, x):
        return x[0]

    def _get_edge_features(self, x):
        return x[2]
    
    def _get_pairwise_potentials(self, x, w):
        edge_features = self._get_edge_features(x)
        pairwise = np.asarray(w[self.n_states * self.n_features:])
        pairwise = pairwise.reshape(self.n_edge_features, -1)
        pairwise = np.dot(edge_features, pairwise)
        res = np.zeros((edge_features.shape[0], self.n_states, self.n_states))
        for i in range(edge_features.shape[0]):
            res[i, :, :] = np.diag(pairwise[i, :])
        return res
    
    def _get_unary_potentials(self, x, w):
        features = self._get_features(x)
        unary_params = w[:self.n_states * self.n_features].reshape(self.n_states, self.n_features)
        res = safe_sparse_dot(features, unary_params.T, dense_output=True)
        return res

    def _loss_augment_unaries(self, unaries, y, weights):
        for label in xrange(self.n_states):
            mask = y != label
            unaries[mask, label] -= weights[mask]
        return unaries

    def _joint_features(self, chain, x, y, edge_index):
        features = self._get_features(x)[chain,:]

        e_ind = []
        edges = []
        for i in xrange(chain.shape[0] - 1):
            edges.append((i, i + 1))
            e_ind.append(edge_index[(chain[i], chain[i + 1])])
        edges = np.array(edges)

        edge_features = self._get_edge_features(x)[e_ind,:]
        n_nodes = features.shape[0]

        gx = np.ogrid[:n_nodes]

        unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.float64)
        unary_marginals[gx, y] = 1

        pw = np.zeros((self.n_edge_features, self.n_states))
        for label in xrange(self.n_states):
            mask = (y[edges[:, 0]] == label) & (y[edges[:, 1]] == label)
            pw[:, label] = np.sum(edge_features[mask], axis=0)

        unaries_acc = safe_sparse_dot(unary_marginals.T, features,
                                      dense_output=True)

        return np.hstack([unaries_acc.ravel(), pw.ravel()])

    def _joint_features_full(self, x, y):
        features, edges = self._get_features(x), self._get_edges(x)
        edge_features = self._get_edge_features(x)
        n_nodes = features.shape[0]

        y = y.reshape(n_nodes)
        gx = np.ogrid[:n_nodes]

        unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.float64)
        unary_marginals[gx, y] = 1

        pw = np.zeros((self.n_edge_features, self.n_states))
        for label in xrange(self.n_states):
            mask = (y[edges[:, 0]] == label) & (y[edges[:, 1]] == label)
            pw[:, label] = np.sum(edge_features[mask], axis=0)

        unaries_acc = safe_sparse_dot(unary_marginals.T, features,
                                      dense_output=True)

        return np.hstack([unaries_acc.ravel(), pw.ravel()])

    def fit(self, X, Y):
        n_nodes = X[0][0].shape[0]
        width = 20
        height = 20
    
        assert n_nodes == width * height
    
        contains_node = []
        lambdas = []
        chains = []
        edge_index = []
        y_hat = []

        self.logger.info('Initialization.')
    
        for x, y in zip(X, Y):
            _edge_index = {}
            for i, edge in enumerate(self._get_edges(x)):
                _edge_index[(edge[0], edge[1])] = i
    
            _y_hat = []
            _chains = []
            _lambdas = []
            _contains = [[] for i in xrange(n_nodes)]
            for i in xrange(0, n_nodes, width):
                _chains.append(np.arange(i, i + width))
                _lambdas.append(np.zeros((width, self.n_states)))
                _y_hat.append(np.zeros(width))
                tree_number = len(_chains) - 1
                for node in _chains[-1]:
                    _contains[node].append(tree_number)
    
            for i in xrange(0, width):
                _chains.append(np.arange(i, n_nodes, width))
                _lambdas.append(np.zeros((height, self.n_states)))
                _y_hat.append(np.zeros(width))
                tree_number = len(chains) - 1
                for node in _chains[-1]:
                    _contains[node].append(tree_number)
    
            contains_node.append(_contains)
            lambdas.append(_lambdas)
            chains.append(_chains)
            edge_index.append(_edge_index)
            y_hat.append(_y_hat)

        w = np.zeros(self.size_w)
        alpha = 0.1

        self.start_time = time.time()
        self.timestamps = [0]
        self.objective_curve = []

        for iteration in xrange(self.max_iter):
            self.logger.info('Iteration %d.', iteration)
            self.logger.info('Optimize slave MRF.')

            objective = 0

            for k in xrange(len(X)):
                x, y = X[k], Y[k]
                unaries = self._get_unary_potentials(x, w)
                aug_unaries = self._loss_augment_unaries(unaries, y.full, y.weights)
                pairwise = self._get_pairwise_potentials(x, w)
                for i in xrange(len(chains[k])):
                    y_hat[k][i], e = optimize_chain(chains[k][i], lambdas[k][i] + 0.5 * aug_unaries[chains[k][i],:], pairwise, edge_index[k])
                    objective -= e

            self.logger.info('Update w.')

            dw = 2 * self.mu * w
            for k in xrange(len(X)):
                x, y = X[k], Y[k]
                psi = self._joint_features_full(x, y.full)
                objective += np.dot(w, psi)
                for i in xrange(len(chains[k])):
                    _psi = self._joint_features(chains[k][i], x, y_hat[k][i], edge_index[k])
                    _psi[:self.n_features * self.n_states] *= 0.5 # hardcoded I_p^k
                    psi -= _psi
                dw += psi

            objective += self.mu * np.sum(w ** 2)
            w -= alpha * dw

            self.logger.info('%s', str(w))

            self.logger.info('Update lambda.')

            for k in xrange(len(X)):
                lambda_sum = np.zeros((n_nodes, self.n_states))
                for p in xrange(n_nodes):
                    for i in contains_node[k][p]:
                        lambda_sum[chains[k][i], y_hat[k][i]] += 1

                for i in xrange(len(chains[k])):
                    N = lambdas[k][i].shape[0]
                    mask = np.zeros((N, self.n_states))
                    mask[np.ogrid[:N], y_hat[k][i]] = 1

                    lambdas[k][i] += alpha * (mask - 0.5 * lambda_sum[chains[k][i],:]) # hardcoded I_p^k

            self.logger.info('%s', lambdas)

            if iteration:
                alpha = 0.1 / np.sqrt(iteration)

            self.timestamps.append(time.time() - self.start_time)
            self.objective_curve.append(objective)

            self.logger.info('Objective: %f', objective)
        
        self.w = w

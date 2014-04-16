import logging
import time
import sys
import numpy as np

from sklearn.utils.extmath import safe_sparse_dot
from chain_opt import optimize_chain_fast


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
                 C=1, verbose=0, max_iter=200, check_every=1):
        self.n_states = n_states
        self.n_features = n_features
        self.n_edge_features = n_edge_features
        self.C = C
        self.verbose = verbose
        self.max_iter = max_iter
        self.size_w = (self.n_states * self.n_features +
                       self.n_states * self.n_edge_features)
        self.logger = logging.getLogger(__name__)
        self.check_every = check_every

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
        return safe_sparse_dot(features, unary_params.T, dense_output=True)

    def _loss_augment_unaries(self, unaries, y, weights):
        unaries = unaries.copy()
        for label in xrange(self.n_states):
            mask = y != label
            unaries[mask, label] -= weights[mask]
        return unaries

    def _joint_features(self, chain, x, y, edge_index):
        features = self._get_features(x)[chain,:]
        n_nodes = features.shape[0]

        e_ind = []
        edges = []
        for i in xrange(chain.shape[0] - 1):
            edges.append((i, i + 1))
            e_ind.append(edge_index[(chain[i], chain[i + 1])])

        edges = np.array(edges)
        edge_features = self._get_edge_features(x)[e_ind,:]

        unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.float64)
        unary_marginals[np.ogrid[:n_nodes], y] = 1
        unaries_acc = safe_sparse_dot(unary_marginals.T, features,
                                      dense_output=True)

        pw = np.zeros((self.n_edge_features, self.n_states))
        for label in xrange(self.n_states):
            mask = (y[edges[:, 0]] == label) & (y[edges[:, 1]] == label)
            pw[:, label] = np.sum(edge_features[mask], axis=0)

        return np.hstack([unaries_acc.ravel(), pw.ravel()])

    def _joint_features_full(self, x, y):
        features, edges, edge_features = \
            self._get_features(x), self._get_edges(x), self._get_edge_features(x)

        n_nodes = features.shape[0]
        y = y.reshape(n_nodes)

        unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.float64)
        unary_marginals[np.ogrid[:n_nodes], y] = 1
        unaries_acc = safe_sparse_dot(unary_marginals.T, features,
                                      dense_output=True)

        pw = np.zeros((self.n_edge_features, self.n_states))
        for label in xrange(self.n_states):
            mask = (y[edges[:, 0]] == label) & (y[edges[:, 1]] == label)
            pw[:, label] = np.sum(edge_features[mask], axis=0)

        return np.hstack([unaries_acc.ravel(), pw.ravel()])

    def fit(self, X, Y, train_scorer, test_scorer):
        n_nodes = X[0][0].shape[0]
        width = 20
        height = 20
    
        assert n_nodes == width * height
    
        contains_node = []
        lambdas = []
        chains = []
        edge_index = []
        y_hat = []

        self.logger.info('Initialization')
    
        for k in xrange(len(X)):
            _edge_index = {}
            for i, edge in enumerate(self._get_edges(X[k])):
                _edge_index[(edge[0], edge[1])] = i
    
            _y_hat = []
            _chains = []
            _lambdas = []
            _contains = [[] for i in xrange(n_nodes)]
            for i in xrange(0, n_nodes, width):
                _chains.append(np.arange(i, i + width))
                assert _chains[-1].shape[0] == width
                _lambdas.append(np.zeros((width, self.n_states)))
                _y_hat.append(np.zeros(width))
                tree_number = len(_chains) - 1
                for node in _chains[-1]:
                    _contains[node].append(tree_number)
    
            for i in xrange(0, width):
                _chains.append(np.arange(i, n_nodes, width))
                assert _chains[-1].shape[0] == height
                _lambdas.append(np.zeros((height, self.n_states)))
                _y_hat.append(np.zeros(height))
                tree_number = len(_chains) - 1
                for node in _chains[-1]:
                    _contains[node].append(tree_number)
    
            contains_node.append(_contains)
            lambdas.append(_lambdas)
            chains.append(_chains)
            edge_index.append(_edge_index)
            y_hat.append(_y_hat)

        w = np.zeros(self.size_w)
        self.w = w.copy()

        self.start_time = time.time()
        self.timestamps = [0]
        self.objective_curve = []
        self.train_score = []
        self.test_score = []
        self.w_history = []

        alpha = 0.1
        mult = 0.5

        for iteration in xrange(self.max_iter):
            self.logger.info('Iteration %d', iteration)
            self.logger.info('Optimize slave MRF and update w')

            objective = 0
            dw = np.zeros(w.shape)

            for k in xrange(len(X)):
                x, y = X[k], Y[k]

                unaries = self._get_unary_potentials(x, w)
                unaries_ = self._loss_augment_unaries(unaries, y.full, y.weights)
                pairwise = self._get_pairwise_potentials(x, w)

                objective += np.dot(w, self._joint_features_full(x, y.full))

                for i in xrange(len(chains[k])):
                    y_hat[k][i], energy = optimize_chain(chains[k][i],
                                                         lambdas[k][i] + mult * unaries_[chains[k][i],:],
                                                         pairwise,
                                                         edge_index[k])

                    _psi = self._joint_features(chains[k][i], x, y.full[chains[k][i]], edge_index[k]) \
                        - self._joint_features(chains[k][i], x, y_hat[k][i], edge_index[k])
                    _psi[:self.n_features * self.n_states] *= mult

                    objective -= energy
                    dw -= _psi

            dw -= w / self.C

            w += alpha * dw
            objectvie = self.C * objective + np.sum(w ** 2) / 2

            if iteration and (iteration % self.check_every == 0):
                self.logger.info('Compute train and test scores')
                self.train_score.append(train_scorer(w))
                self.logger.info('Train SCORE: %f', self.train_score[-1])
                self.test_score.append(test_scorer(w))
                self.logger.info('Test SCORE: %f', self.test_score[-1])

            self.logger.info('Update lambda')

            for k in xrange(len(X)):
                lambda_sum = np.zeros((n_nodes, self.n_states), dtype=np.float64)

                for p in xrange(n_nodes):
                    assert len(contains_node[k][p]) == 2
                    for i in contains_node[k][p]:
                        pos = np.where(chains[k][i] == p)[0][0]
                        lambda_sum[p, y_hat[k][i][pos]] += 1

                for i in xrange(len(chains[k])):
                    N = lambdas[k][i].shape[0]

                    lambdas[k][i][np.ogrid[:N], y_hat[k][i]] += alpha
                    lambdas[k][i] -= alpha * mult * lambda_sum[chains[k][i],:]

            self.logger.info('diff: %f', np.sum((w-self.w)**2))
            if iteration:
                alpha = max(1e-10, 1.0 / iteration)

            self.timestamps.append(time.time() - self.start_time)
            self.objective_curve.append(objective)

            self.logger.info('Objective: %f', objective)

            self.w = w.copy()
            self.w_history.append(self.w)
        
        self.w = w

        self.timestamps = np.array(self.timestamps)
        self.objective_curve = np.array(self.objective_curve)
        self.train_score = np.array(self.train_score)
        self.test_score = np.array(self.test_score)
        self.w_history = np.vstack(self.w_history)

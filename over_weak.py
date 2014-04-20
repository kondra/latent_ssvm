import logging
import time
import sys
import itertools

import numpy as np

from joblib import Parallel, delayed
from sklearn.utils.extmath import safe_sparse_dot
from chain_opt import optimize_chain_fast
from common import latent


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
    unaries = mu

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
            t_unaries = unaries.copy()
            t_unaries[:, labels] = -np.Inf
            y_hat = np.argmax(t_unaries, axis=1)
            energy = np.sum(np.max(unaries[:,labels], axis=1))
            present_labels = set(np.unique(y_hat))
            if len(present_labels.intersection(gt_labels)):
                energy -= c
            if energy > max_energy:
                max_energy = energy
                best_y = y_hat

    return best_y, max_energy


class OverWeak(object):
    def __init__(self, model, n_states, n_features, n_edge_features,
                 C=1, verbose=0, max_iter=200, check_every=1,
                 complete_every=1, alpha_kappa=1):
        self.model = model
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
        self.complete_every = complete_every
        self.alpha_kappa = alpha_kappa
        self.n_jobs = 4

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
            unaries[mask, label] += weights[mask]
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
        mu = {}

        self.logger.info('Initialization')
    
        for k in xrange(len(X)):
            x, y = X[k], Y[k]
            _edge_index = {}
            for i, edge in enumerate(self._get_edges(x)):
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

            if not y.full_labeled:
                mu[k] = np.zeros((n_nodes, self.n_states))

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

                if y.full_labeled:
                    unaries = self._loss_augment_unaries(self._get_unary_potentials(x, w),
                                                         y.full, y.weights)
                    pairwise = self._get_pairwise_potentials(x, w)

                    objective -= np.dot(w, self._joint_features_full(x, y.full))

                    for i in xrange(len(chains[k])):
                        y_hat[k][i], energy = optimize_chain(chains[k][i],
                                                             lambdas[k][i] + mult * unaries[chains[k][i],:],
                                                             pairwise,
                                                             edge_index[k])

                        _psi = -self._joint_features(chains[k][i], x, y.full[chains[k][i]], edge_index[k]) \
                            + self._joint_features(chains[k][i], x, y_hat[k][i], edge_index[k])
                        _psi[:self.n_features * self.n_states] *= mult

                        objective += energy
                        dw += _psi
                else:
                    dmu = np.zeros((n_nodes, self.n_states))

                    unaries = self._get_unary_potentials(x, w) - mu[k]
                    pairwise = self._get_pairwise_potentials(x, w)

                    objective -= np.dot(w, self._joint_features_full(x, y.full))
                    dw -= self._joint_features_full(x, y.full)

                    for i in xrange(len(chains[k])):
                        y_hat[k][i], energy = optimize_chain(chains[k][i],
                                                             lambdas[k][i] + mult * unaries[chains[k][i],:],
                                                             pairwise,
                                                             edge_index[k])

                        _psi = self._joint_features(chains[k][i], x, y_hat[k][i], edge_index[k])
                        _psi[:self.n_features * self.n_states] *= mult

                        objective += energy
                        dw += _psi

                        dmu[chains[k][i], y_hat[k][i]] -= mult

                    y_hat_kappa, energy = optimize_kappa(y, mu[k], self.alpha_kappa, n_nodes, self.n_states)

                    objective += energy
                    dmu[np.ogrid[:dmu.shape[0]], y_hat_kappa] += 1

                    mu[k] -= alpha * dmu

            dw += w / self.C

            w -= alpha * dw
            objective = self.C * objective + np.sum(w ** 2) / 2

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

            if iteration % self.complete_every == 0:
                self.logger.info('Complete latent variables')
                Y_new = Parallel(n_jobs=self.n_jobs, verbose=0, max_nbytes=1e8)(
                    delayed(latent)(self.model, x, y, w) for x, y in zip(X, Y))
                changes = np.sum([np.any(y_new.full != y.full) for y_new, y in zip(Y_new, Y)])
                self.logger.info('changes in latent variables: %d', changes)
                Y = Y_new

            if iteration and (iteration % self.check_every == 0):
                self.logger.info('Compute train and test scores')
                self.train_score.append(train_scorer(w))
                self.logger.info('Train SCORE: %f', self.train_score[-1])
                self.test_score.append(test_scorer(w))
                self.logger.info('Test SCORE: %f', self.test_score[-1])

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

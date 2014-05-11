import logging
import time
import sys
import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils.extmath import safe_sparse_dot
from graph_utils import decompose_graph, decompose_grid_graph
from trw_utils import *


class OverUnconstr(object):
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
            unaries[mask, label] += weights[mask]
        return unaries

    def _joint_features(self, chain, x, y, edge_index, multiplier):
        features = self._get_features(x)[chain,:]
        n_nodes = features.shape[0]

        features *= multiplier[chain,:]

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

    def fit(self, X, Y, train_scorer, test_scorer, decompose='grid'):
        print('over unconstr begin')

        if decompose == 'general':
            contains_node, chains, edge_index = decompose_graph(X)
        elif decompose == 'grid':
            contains_node, chains, edge_index, sign = decompose_grid_graph(X, get_sign=True)
        else:
            raise ValueError

        y_hat = []
        lambdas = []
        multiplier = []
        for k in xrange(len(X)):
            n_nodes = X[k][0].shape[0]
            _y_hat = []
            _multiplier = []
            for p in xrange(n_nodes):
                _multiplier.append(1.0 / len(contains_node[k][p]))
            for chain in chains[k]:
                _y_hat.append(np.zeros(len(chain)))
            lambdas.append(np.zeros((n_nodes, self.n_states)))
            y_hat.append(_y_hat)
            _multiplier = np.array(_multiplier)
            _multiplier.shape = (n_nodes, 1)
            multiplier.append(_multiplier)

        w = np.zeros(self.size_w)
        self.w = w.copy()

        self.start_time = time.time()
        self.timestamps = [0]
        self.objective_curve = []
        self.train_score = []
        self.test_score = []
        self.w_history = []

        learning_rate = 0.1

        for iteration in xrange(self.max_iter):
            print('Iteration %d' % iteration)

            objective = 0
            dw = np.zeros(w.shape)

            for k in xrange(len(X)):
                x, y = X[k], Y[k]
                n_nodes = x[0].shape[0]

                unaries = self._loss_augment_unaries(self._get_unary_potentials(x, w), y.full, y.weights)
                unaries *= multiplier[k]

                pairwise = self._get_pairwise_potentials(x, w)

                objective -= np.dot(w, self._joint_features_full(x, y.full))
                dw -= self._joint_features_full(x, y.full)

                for i in xrange(len(chains[k])):
                    y_hat[k][i], energy = optimize_chain(chains[k][i],
                                                         sign[k][i] * lambdas[k][chains[k][i],:] + unaries[chains[k][i],:],
                                                         pairwise,
                                                         edge_index[k])

                    dw += self._joint_features(chains[k][i], x, y_hat[k][i], edge_index[k], multiplier[k])
                    objective += energy

            dw += w / self.C

            w -= learning_rate * dw
            objective = self.C * objective + np.sum(w ** 2) / 2

            if iteration and (iteration % self.check_every == 0):
                print('Compute train and test scores')
                self.train_score.append(train_scorer(w))
                print('Train SCORE: %f' % self.train_score[-1])
                self.test_score.append(test_scorer(w))
                print('Test SCORE: %f' % self.test_score[-1])

            for k in xrange(len(X)):
                n_nodes = X[k][0].shape[0]

                for p in xrange(n_nodes):
                    dlambda = np.zeros(self.n_states)
                    for i in contains_node[k][p]:
                        pos = np.where(chains[k][i] == p)[0][0]
                        dlambda[y_hat[k][i][pos]] += sign[k][i]
                    lambdas[k][p] -= learning_rate * dlambda

            if iteration:
                learning_rate = 1.0 / iteration

            self.timestamps.append(time.time() - self.start_time)
            self.objective_curve.append(objective)

            self.w = w.copy()
            self.w_history.append(self.w)
        
        self.w = w

        self.timestamps = np.array(self.timestamps)
        self.objective_curve = np.array(self.objective_curve)
        self.train_score = np.array(self.train_score)
        self.test_score = np.array(self.test_score)
        self.w_history = np.vstack(self.w_history)

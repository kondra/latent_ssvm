import logging
import time
import sys
import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils.extmath import safe_sparse_dot
from graph_utils import decompose_graph, decompose_grid_graph
from trw_utils import *
from heterogenous_crf import inference_gco


class OverLbfgs(object):
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

    def loss_augmented_inference(self, x, y, w):
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)

        for label in xrange(self.n_states):
            mask = y.full != label
            unary_potentials[mask, label] += y.weights[mask]

        return inference_gco(unary_potentials, pairwise_potentials, edges,
                          n_iter=5, return_energy=True)

    def fit(self, X, Y, train_scorer, test_scorer, decompose='grid', w0=None):
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


        history = {'train_scores': [],
                   'test_scores': [],
                   'objective': [],
                   'iteration': 0,
                   'w': []
                   }

        self.train_scorer = train_scorer
        self.test_scorer = test_scorer

        #x0 = np.zeros(self.size_w + 4000 * len(X))
        x0 = np.zeros(self.size_w)
        if w0 is not None:
            x0 = w0

#        l = 0.1
#        for iteration in xrange(100):
#            fval, grad = f2(w, self, X, Y, history)
#            w -= l * grad
#            if iteration:
#                l = 0.01 / iteration

        x, f_val, d = fmin_l_bfgs_b(f2, x0,
                                    args=(self, X, Y, history),
                                    maxiter=self.max_iter,
                                    disp=0,
                                    pgtol=1e-8)

        return w, history


def f(xx, model, X, Y, multiplier, chains, sign, edge_index, y_hat, contains_node, history):
    w = xx[:model.size_w].copy()
    sz = xx.shape

    lambdas = []
    dw = np.zeros(w.shape)
    objective = 0

    for k in xrange(len(X)):
        lambdas.append(xx[(model.size_w + k * 4000):(model.size_w + (k + 1) * 4000)].reshape((400,10)))

    for k in xrange(len(X)):
        x, y = X[k], Y[k]
        n_nodes = x[0].shape[0]

        unaries = model._loss_augment_unaries(model._get_unary_potentials(x, w), y.full, y.weights)
        unaries *= multiplier[k]

        pairwise = model._get_pairwise_potentials(x, w)

        dw -= model._joint_features_full(x, y.full)
        objective -= np.dot(w, model._joint_features_full(x, y.full))

        for i in xrange(len(chains[k])):
            y_hat[k][i], energy = optimize_chain(chains[k][i],
                                                 sign[k][i] * lambdas[k][chains[k][i],:] + unaries[chains[k][i],:],
                                                 pairwise,
                                                 edge_index[k])

            dw += model._joint_features(chains[k][i], x, y_hat[k][i], edge_index[k], multiplier[k])
            objective += energy

    dw += w / model.C
    objective = model.C * objective + np.sum(w ** 2) / 2

    grad = np.zeros(sz)
    grad[:model.size_w] = dw

    for k in xrange(len(X)):
        n_nodes = X[k][0].shape[0]

        dlambda = np.zeros((n_nodes, model.n_states))
        for p in xrange(n_nodes):
            for i in contains_node[k][p]:
                pos = np.where(chains[k][i] == p)[0][0]
                dlambda[p,y_hat[k][i][pos]] += sign[k][i]

        grad[(model.size_w + k * 4000):(model.size_w + (k + 1) * 4000)] = dlambda.reshape(4000)

    history['iteration'] += 1
    history['objective'].append(objective)
    history['w'].append(w)

    print history['iteration']

    if history['iteration'] % 2 == 0:
        train_score = model.train_scorer(w)
        history['train_scores'].append(train_score)
        print 'Train SCORE: {}'.format(train_score)

        test_score = model.test_scorer(w)
        history['test_scores'].append(test_score)
        print 'Test SCORE: {}'.format(test_score)

    print w
    #print grad[:model.size_w]
    print objective

    return objective, grad


def f2(xx, model, X, Y, history):
    w = xx.copy()

    dw = np.zeros(w.shape)
    objective = 0

    o2 = 0

    for k in xrange(len(X)):
        x, y = X[k], Y[k]
        y_hat, energy = model.loss_augmented_inference(x, y, w)

        objective -= energy
        objective -= np.dot(w, model._joint_features_full(x, y.full))

        o2 += np.dot(w, model._joint_features_full(x, y_hat)) \
            -np.dot(w, model._joint_features_full(x, y.full))

        dw += model._joint_features_full(x, y_hat)
        dw -= model._joint_features_full(x, y.full)

    print 'mean diff = {}'.format(o2 / len(X))

    dw += w / model.C
    objective = model.C * objective + np.sum(w ** 2) / 2

    history['iteration'] += 1
    history['objective'].append(objective)
    history['w'].append(w)

    print history['iteration']
    print objective

    if history['iteration'] % 10 == 0:
        train_score = model.train_scorer(w)
        history['train_scores'].append(train_score)
        print 'Train SCORE: {}'.format(train_score)

        test_score = model.test_scorer(w)
        history['test_scores'].append(test_score)
        print 'Test SCORE: {}'.format(test_score)

    return objective, dw

import logging
import time
import sys
import numpy as np

from sklearn.utils.extmath import safe_sparse_dot


def inference_gco(unary_potentials, pairwise_potentials, edges,
                  label_costs=None, **kwargs):
    from pygco import cut_from_graph_gen_potts

    shape_org = unary_potentials.shape[:-1]
    n_states = unary_potentials.shape[-1]

    pairwise_cost = {}
    count = 0
    for i in xrange(0, pairwise_potentials.shape[0]):
        count += np.sum(np.diag(pairwise_potentials[i, :]) < 0)
        pairwise_cost[(edges[i, 0], edges[i, 1])] = list(np.maximum(
            np.diag(pairwise_potentials[i, :]), 0))

    unary_potentials *= -1

    if 'n_iter' in kwargs:
        y = cut_from_graph_gen_potts(unary_potentials, pairwise_cost, 
                                     label_cost=label_costs, n_iter=kwargs['n_iter'])
    else:
        y = cut_from_graph_gen_potts(unary_potentials, pairwise_cost,
                                     label_cost=label_costs)

    if 'return_energy' in kwargs and kwargs['return_energy']:
        return y[0].reshape(shape_org), y[1]
    else:
        return y[0].reshape(shape_org)


class Subgrad(object):
    def __init__(self, model, n_states, n_features, n_edge_features,
                 mu=1, verbose=0, max_iter=200):
        self.n_states = n_states
        self.model = model
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

    def fit(self, X, Y, scorer):
        w = np.zeros(self.size_w)
        w_old = np.zeros(self.size_w)
        alpha = 0.1

        self.start_time = time.time()
        self.timestamps = [0]
        self.objective_curve = []
        self.train_score = []

        for iteration in xrange(self.max_iter):
            self.logger.info('Iteration %d', iteration)

            objective = 0
            dw = np.zeros(w.shape)

            for k in xrange(len(X)):
                x, y = X[k], Y[k]

                unaries = self._loss_augment_unaries(self._get_unary_potentials(x, w),
                                                     y.full, y.weights)
                pairwise = self._get_pairwise_potentials(x, w)
                y_hat = inference_gco(-unaries, -pairwise, self._get_edges(x), n_iter=5)
                
                dw -= self._joint_features_full(x, y.full) \
                    - self._joint_features_full(x, y_hat)

            dw -= w / self.mu

            self.logger.info('step: alpha = %f', alpha)

            w += alpha * dw

            self.train_score.append(scorer(w))
            self.logger.info('___________________SCORE: %f', self.train_score[-1])
            self.logger.info('diff: %f', np.sum((w-w_old)**2))
#            self.logger.info('Objective: %f', objective)

            if iteration:
                alpha = 0.1*max(1e-10, 1.0 / iteration)

            self.timestamps.append(time.time() - self.start_time)
            self.objective_curve.append(objective)

            w_old = w.copy()
        
        self.w = w


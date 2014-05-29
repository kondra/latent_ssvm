import logging
import time
import sys

import numpy as np

from joblib import Parallel, delayed
from sklearn.utils.extmath import safe_sparse_dot
from chain_opt import optimize_chain_fast
from common import latent
from trw_utils import optimize_chain, optimize_kappa
from graph_utils import decompose_graph, decompose_grid_graph
from heterogenous_crf import inference_gco

from pyqpbo import binary_general_graph
from scipy.optimize import fmin_l_bfgs_b
import scipy.sparse as sps

def f(x, node_weights, pairwise, edges):
    n_nodes, n_states = node_weights.shape

    dual = 0
    dlambda = np.zeros(n_nodes)

    for k in xrange(n_states):
        new_unaries = np.zeros((n_nodes, 2))
        new_unaries[:,1] = node_weights[:,k] + x
        y_hat, energy = binary_general_graph(edges, new_unaries, pairwise[k])
        dual += 0.5 * energy
        dlambda += y_hat
    dlambda -= 1

    dual -= np.sum(x)
    #print dual

    return -dual, -dlambda


class OverWeak(object):
    def __init__(self, model, n_states, n_features, n_edge_features,
                 C=1, verbose=0, max_iter=200, check_every=1,
                 complete_every=1, alpha=1, update_w_every=50,
                 update_mu=20):
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
        self.alpha = alpha
        self.n_jobs = 4
        self.update_w_every = update_w_every
        self.update_mu = update_mu

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

        label_costs = np.zeros(self.n_states)
        c = np.sum(y.weights) / float(self.n_states)
        for label in y.weak:
            label_costs[label] = c

        for label in xrange(0, self.n_states):
            if label not in y.weak:
                unary_potentials[:, label] += y.weights
    
        h = inference_gco(unary_potentials, pairwise_potentials, edges,
                          label_costs, n_iter=5, return_energy=True)
    
        return h

    def fit(self, X, Y, train_scorer, test_scorer, decompose='general',
            use_latent_first_iter=500, undergenerating_weak=True, smd=False):
        self.logger.info('Initialization')

        if decompose == 'general':
            contains_node, chains, edge_index = decompose_graph(X)
        elif decompose == 'grid':
            contains_node, chains, edge_index = decompose_grid_graph(X)
        else:
            raise ValueError
    
        y_hat = []
        lambdas = []
        multiplier = []
        xx = []
        mu = {}
        for k in xrange(len(X)):
            x, y = X[k], Y[k]
            n_nodes = x[0].shape[0]
            xx.append(np.zeros(n_nodes))
            _lambdas = []
            _y_hat = []
            _multiplier = []
            for p in xrange(n_nodes):
                _multiplier.append(1.0 / len(contains_node[k][p]))
            for chain in chains[k]:
                _lambdas.append(np.zeros((len(chain), self.n_states)))
                _y_hat.append(np.zeros(len(chain), dtype=np.int32))
            lambdas.append(_lambdas)
            y_hat.append(_y_hat)
            _multiplier = np.array(_multiplier)
            _multiplier.shape = (n_nodes, 1)
            multiplier.append(_multiplier)
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

        learning_rate1 = 0.1
        learning_rate2 = 0.1

        for iteration in xrange(self.max_iter):
            self.logger.info('Iteration %d', iteration)
            self.logger.info('Optimize slave MRF and update w')

            objective = 0
            dw = np.zeros(w.shape)

            for k in xrange(len(X)):
                x, y = X[k], Y[k]
                n_nodes = x[0].shape[0]

#                self.logger.info('object %d', k)

                if y.full_labeled:
                    unaries = self._loss_augment_unaries(self._get_unary_potentials(x, w),
                                                         y.full, y.weights)
                    unaries *= multiplier[k]
                    pairwise = self._get_pairwise_potentials(x, w)

                    jf = self._joint_features_full(x, y.full)
                    objective -= np.dot(w, jf)
                    dw -= jf

                    for i in xrange(len(chains[k])):
                        y_hat[k][i], energy = optimize_chain(chains[k][i],
                                                             lambdas[k][i] + unaries[chains[k][i],:],
                                                             pairwise,
                                                             edge_index[k])

                        dw += self._joint_features(chains[k][i], x, y_hat[k][i], edge_index[k], multiplier[k])

                        objective += energy
                elif iteration > use_latent_first_iter:
                    if undergenerating_weak:
# Use gco for full K oracle
#                        y_hat_, energy = self.loss_augmented_inference(x, y, w)
#                        jf_gt = self._joint_features_full(x, y.full)
#                        objective -= np.dot(w, jf_gt)
#                        objective += energy
#                        dw -= jf_gt
#                        dw += self._joint_features_full(x, y_hat_)

# use gco for first summand in DD
                        for mm in xrange(10):
                            dmu = np.zeros((n_nodes, self.n_states))

                            unaries = self._get_unary_potentials(x, w) - mu[k]
                            pairwise = self._get_pairwise_potentials(x, w)

                            y_hat_gco, energy = inference_gco(unaries, pairwise, self._get_edges(x),
                                                              n_iter=5, return_energy=True)
                            objective -= energy
                            dmu[np.ogrid[:dmu.shape[0]], y_hat_gco] -= 1
                            dw += self._joint_features_full(x, y_hat_gco)

                            jf = self._joint_features_full(x, y.full)
                            objective -= np.dot(w, jf)
                            dw -= jf

                            y_hat_kappa, energy = optimize_kappa(y, mu[k], self.alpha, n_nodes, self.n_states)
                            objective += energy
                            dmu[np.ogrid[:dmu.shape[0]], y_hat_kappa] += 1

                            mu[k] -= learning_rate2 * dmu
                    elif not smd:
                        dmu = np.zeros((n_nodes, self.n_states))

                        unaries = (self._get_unary_potentials(x, w) - mu[k]) * multiplier[k]
                        pairwise = self._get_pairwise_potentials(x, w)

                        jf = self._joint_features_full(x, y.full)
                        objective -= np.dot(w, jf)
                        dw -= jf

#begin inner (can remove this to restore to previous state)
                        E = 0
                        Eprev = -100
                        for j in xrange(self.update_mu):
                            E = 0
                            for i in xrange(len(chains[k])):
                                y_hat[k][i], energy = optimize_chain(chains[k][i],
                                                                     lambdas[k][i] + unaries[chains[k][i],:],
                                                                     pairwise,
                                                                     edge_index[k])
                                E += energy


                            lambda_sum = np.zeros((n_nodes, self.n_states), dtype=np.float64)

                            for p in xrange(n_nodes):
                                for i in contains_node[k][p]:
                                    pos = np.where(chains[k][i] == p)[0][0]
                                    lambda_sum[p, y_hat[k][i][pos]] += multiplier[k][p]

                            for i in xrange(len(chains[k])):
                                N = lambdas[k][i].shape[0]

                                lambdas[k][i][np.ogrid[:N], y_hat[k][i]] -= learning_rate2
                                lambdas[k][i] += learning_rate2 * lambda_sum[chains[k][i],:]

                            if np.abs(E - Eprev) < 0.1:
                                break
                            Eprev = E
#end inner

#last one
                        for i in xrange(len(chains[k])):
                            y_hat[k][i], energy = optimize_chain(chains[k][i],
                                                                 lambdas[k][i] + unaries[chains[k][i],:],
                                                                 pairwise,
                                                                 edge_index[k])

                            dw += self._joint_features(chains[k][i], x, y_hat[k][i], edge_index[k], multiplier[k])

                            objective += energy

                            dmu[chains[k][i], y_hat[k][i]] -= multiplier[k][chains[k][i]].flatten()
#

                        y_hat_kappa, energy = optimize_kappa(y, mu[k], self.alpha, n_nodes, self.n_states)

                        objective += energy
                        dmu[np.ogrid[:dmu.shape[0]], y_hat_kappa] += 1

                        mu[k] -= learning_rate2 * dmu
                    elif smd:
                        if iteration > 1500:
                            mMu = 10
                        else:
                            mMu = 1
                        for mm in xrange(mMu):
                            dmu = np.zeros((n_nodes, self.n_states))

                            jf = self._joint_features_full(x, y.full)
                            objective -= np.dot(w, jf)
                            dw -= jf

                            unaries = -self._get_unary_potentials(x, w) + mu[k]
                            edge_weights = -self._get_pairwise_potentials(x, w)
                            edges = self._get_edges(x)

                            n_edges = edges.shape[0]
                            y_hat2 = []
                            pairwise = []
                            for j in xrange(self.n_states):
                                y_hat2.append(np.zeros(self.n_states))
                                _pairwise = np.zeros((n_edges, 2, 2))
                                for i in xrange(n_edges):
                                    _pairwise[i,1,0] = _pairwise[i,0,1] = -0.5 * edge_weights[i,j,j]
                                pairwise.append(_pairwise)
                    
                            for i in xrange(n_edges):
                                e1, e2 = edges[i]
                                unaries[e1,:] += 0.5 * np.diag(edge_weights[i,:,:])
                                unaries[e2,:] += 0.5 * np.diag(edge_weights[i,:,:])
                    
                            xx[k], f_val, d = fmin_l_bfgs_b(f, xx[k],
                                                            args=(unaries, pairwise, edges),
                                                            maxiter=50,
                                                            maxfun=50,
                                                            pgtol=1e-2)
                                
                            E = np.sum(xx[k])
                            for j in xrange(self.n_states):
                                new_unaries = np.zeros((n_nodes, 2))
                                new_unaries[:,1] = unaries[:,j] + xx[k]
                                y_hat2[j], energy = binary_general_graph(edges, new_unaries, pairwise[j])
                                E -= 0.5*energy
                                dmu[:,j] -= y_hat2[j]

                                dw += self._joint_features_full(x, y_hat2[j] * j)
                    
                            y_hat_kappa, energy = optimize_kappa(y, mu[k], 1, n_nodes, self.n_states)
                            E += energy
                            dmu[np.ogrid[:dmu.shape[0]], y_hat_kappa] += 1
                            objective += E
                    
                            mu[k] -= learning_rate2 * dmu

            dw += w / self.C

            if iteration < 100 or iteration % self.update_w_every == 0:
                w -= learning_rate1 * dw
            objective = self.C * objective + np.sum(w ** 2) / 2

            self.logger.info('Update lambda')

            for k in xrange(len(X)):
                if undergenerating_weak and not Y[k].full_labeled:
                    continue
                if smd and not Y[k].full_labeled:
                    continue

                n_nodes = X[k][0].shape[0]
                lambda_sum = np.zeros((n_nodes, self.n_states), dtype=np.float64)

                for p in xrange(n_nodes):
                    for i in contains_node[k][p]:
                        pos = np.where(chains[k][i] == p)[0][0]
                        lambda_sum[p, y_hat[k][i][pos]] += multiplier[k][p]

                for i in xrange(len(chains[k])):
                    N = lambdas[k][i].shape[0]

                    lambdas[k][i][np.ogrid[:N], y_hat[k][i]] -= learning_rate2
                    lambdas[k][i] += learning_rate2 * lambda_sum[chains[k][i],:]

            if iteration % self.complete_every == 0 or iteration in [51, 80, 101, 130]:
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
                learning_rate1 = 1.0 / iteration
                learning_rate2 = 1.0 / iteration

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

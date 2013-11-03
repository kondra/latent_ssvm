######################
# (c) 2013 Dmitry Kondrashkin <kondra2lp@gmail.com>
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>

import numpy as np

from pystruct.models.base import StructuredModel
from pystruct.inference.inference_methods import inference_dispatch
from pystruct.models.utils import loss_augment_weighted_unaries

from sklearn.utils.extmath import safe_sparse_dot

from label import Label


class HCRF(StructuredModel):
    def __init__(self, n_states=2, n_features=None, n_edge_features=1,
                 inference_method='gco', n_iter=5, alpha=1):
        if inference_method != 'gco':
            # only gco inference_method supported: we need label costs
            raise NotImplementedError
        self.all_states = set(range(0, n_states))
        self.n_edge_features = n_edge_features
        self.n_states = n_states
        self.n_features = n_features
        self.inference_method = inference_method
        self.inference_calls = 0
        self.alpha = alpha
        self.n_iter = n_iter
        self.size_psi = (self.n_states * self.n_features +
                         self.n_edge_features)

    def _check_size_x(self, x):
        features = self._get_features(x)
        if features.shape[1] != self.n_features:
            raise ValueError("Unary evidence should have %d feature per node,"
                             " got %s instead."
                             % (self.n_features, features.shape[1]))

    def __repr__(self):
        return ("%s(n_states: %d, inference_method: %s, n_features: %d, "
                "n_edge_features: %d)"
                % (type(self).__name__, self.n_states, self.inference_method,
                   self.n_features, self.n_edge_features))

    def _get_edges(self, x):
        return x[1]

    def _get_features(self, x):
        return x[0]

    def latent(self, x, y, w):
        if y.full_labeled:
            return y
        unary_potentials = self._get_unary_potentials(x, w)
        # forbid h that is incompoatible with y
        # by modifying unary params
        other_states = list(self.all_states - set(y.weak))
        unary_potentials[:, other_states] = -1000
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False, n_iter=self.n_iter)
        return Label(h, y.weak, y.weights, False)

    def _get_pairwise_potentials(self, x, w):
        """Computes pairwise potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_psi,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_edges, n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        edge_features = x[2]
        pairwise = np.asarray(w[self.n_states * self.n_features:])
        pairwise = pairwise.reshape(self.n_edge_features, -1)
        pairwise = np.dot(edge_features, pairwise)
        res = np.zeros((edge_features.shape[0], self.n_states, self.n_states))
        for i in range(edge_features.shape[0]):
            res[i, :, :] = np.diag(np.repeat(pairwise[i], self.n_states))
        return res

    def _get_unary_potentials(self, x, w):
        """Computes unary potentials for x and w.

        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_psi,)
            Weight vector for CRF instance.

        Returns
        -------
        unary : ndarray, shape=(n_nodes, n_states)
            Unary weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        features, edges = self._get_features(x), self._get_edges(x)
        unary_params = w[:self.n_states * self.n_features].reshape(
            self.n_states, self.n_features)
        result = safe_sparse_dot(features, unary_params.T, dense_output=True)
        return result

    def psi(self, x, y):
        self._check_size_x(x)
        features, edges = self._get_features(x), self._get_edges(x)
        n_nodes = features.shape[0]
        edge_features = x[2]

        full_labeled = y.full_labeled

        y = y.full
        y = y.reshape(n_nodes)
        gx = np.ogrid[:n_nodes]

        unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
        gx = np.ogrid[:n_nodes]
        unary_marginals[gx, y] = 1

        pw = np.sum(edge_features[y[edges[:, 0]] == y[edges[:, 1]]], axis=0)

        unaries_acc = safe_sparse_dot(unary_marginals.T, features, dense_output=True)

        psi_vector = np.hstack([unaries_acc.ravel(), pw.ravel()])
        if not full_labeled:
            psi_vector *= self.alpha
        return psi_vector

    def loss(self, y, y_hat):
        if y.full_labeled:
            return np.sum(y.weights * (y.full != y_hat.full))
        else:
            # should use Kappa here
            loss = 0
            c = np.sum(y.weights) / float(self.n_states)
            for label in xrange(0, self.n_states):
                if label in y.weak and not np.any(y_hat.full == label):
                    loss += c
                elif label not in y.weak:
                    loss += np.sum(y.weights * (y_hat.full == label))
            return loss * self.alpha

    def loss_augmented_inference(self, x, y, w, relaxed=False,
                                 return_energy=False):
        # we do not support relaxed inference yet
        relaxed = False
        return_energy = False

        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)

        if y.full_labeled:
            loss_augment_weighted_unaries(unary_potentials, y.full,
                                          y.weights.astype(np.double))

            h = inference_dispatch(unary_potentials, pairwise_potentials,
                                   edges, self.inference_method,
                                   relaxed=relaxed,
                                   return_energy=return_energy,
                                   n_iter=self.n_iter)
            return Label(h, None, y.weights, True)
        else:
            # this is weak labeled example
            # use pygco with label costs
            label_cost = np.zeros(self.n_states)
            c = np.sum(y.weights) / float(self.n_states)
            for label in y.weak:
                label_cost[label] = c
            for label in xrange(0, self.n_states):
                if label not in y.weak:
                    unary_potentials[:, label] += y.weights

            edges = edges.copy().astype(np.int32)
            pairwise_potentials = (1000 * pairwise_potentials).copy().astype(
                np.int32)

            pairwise_cost = {}
            for i in xrange(0, edges.shape[0]):
                cost = pairwise_potentials[i, 0, 0]
                if cost >= 0:
                    pairwise_cost[(edges[i, 0], edges[i, 1])] = cost

            from pygco import cut_from_graph_gen_potts
            shape_org = unary_potentials.shape[:-1]

            unary_potentials = (-1000 * unary_potentials).copy().astype(
                np.int32)
            unary_potentials = unary_potentials.reshape(-1, self.n_states)
            label_cost = (1000 * label_cost).copy().astype(np.int32)

            h = cut_from_graph_gen_potts(unary_potentials, pairwise_cost,
                                         label_cost=label_cost, n_iter=self.n_iter)
            h = h[0].reshape(shape_org)

            return Label(h, None, y.weights, False)

    def inference(self, x, w, relaxed=False, return_energy=False):
        """Inference for x using parameters w.

        Finds (approximately)
        armin_y np.dot(w, psi(x, y))
        using self.inference_method.


        Parameters
        ----------
        x : tuple
            Instance of a graph with unary & pairwise potentials.
            x=(unaries, edges, pairwise)
            unaries are an nd-array of shape (n_nodes, n_states),
            edges are an nd-array of shape (n_edges, 2)
            pairwise are an nd-array of shape (n_edges, n_states, n_states)

        w : ndarray, shape=(size_psi,)
            Parameters for the CRF energy function.

        relaxed : bool, default=False
            We do not support it yet.

        return_energy : bool, default=False
            Whether to return the energy of the solution (x, y) that was found.

        Returns
        -------
        y_pred : ndarray
            By default an integer ndarray of shape=(width, height)
            of variable assignments for x is returned.

        """
        # we do not support relaxed inference yet
        relaxed = False
        return_energy = False

        self._check_size_w(w)
        self.inference_calls += 1
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)

        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=relaxed,
                               return_energy=return_energy, n_iter=self.n_iter)

        return Label(h, None, None, True)

import numpy as np

from pystruct.models.graph_crf import GraphCRF
from pystruct.inference.inference_methods import inference_dispatch
from pystruct.models.utils import loss_augment_unaries


class HCRF(GraphCRF):
    def __init__(self, n_states=2, n_features=None, n_edge_features=1,
                 inference_method=None, class_weight=None):
        self.all_states = set(range(0, n_states))
        self.n_edge_features = n_edge_features
        GraphCRF.__init__(self, n_states, n_features, inference_method,
                          class_weight=class_weight)

    def _set_size_psi(self):
        self.size_psi = (self.n_states * self.n_features +
                         self.n_edge_features)

    def __repr__(self):
        return ("%s(n_states: %d, inference_method: %s, n_features: %d, "
                "n_edge_features: %d)"
                % (type(self).__name__, self.n_states, self.inference_method,
                   self.n_features, self.n_edge_features))

    def label_from_latent(self, h):
        return np.unique(h)

    def init_latent(self, X, Y):
        return [self.latent(x, y, np.zeros(self.size_psi))
                for x, y in zip(X, Y)]

    def latent(self, x, y, w):
        unary_potentials = self._get_unary_potentials(x, w)
        # forbid h that is incompoatible with y
        # by modifying unary params
        other_states = list(self.all_states - set(y))
        unary_potentials[:, other_states] = -10
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
        h = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                               self.inference_method, relaxed=False)
        return h

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
        result = np.dot(features, unary_params.T)
        return result

    def psi(self, x, y):
        self._check_size_x(x)
        features, edges = self._get_features(x), self._get_edges(x)
        n_nodes = features.shape[0]
        edge_features = x[2]

        if isinstance(y, tuple):
            # take full labels
            y = y[0]

        y = y.reshape(n_nodes)
        gx = np.ogrid[:n_nodes]

        #make one hot encoding
        unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
        gx = np.ogrid[:n_nodes]
        unary_marginals[gx, y] = 1

        pw = np.sum(edge_features[y[edges[:, 0]] == y[edges[:, 1]]], axis=0)

        unaries_acc = np.dot(unary_marginals.T, features)

        psi_vector = np.hstack([unaries_acc.ravel(), pw.ravel()])
        return psi_vector

    def loss(self, y, y_hat):
        if not isinstance(y, tuple):
            if hasattr(self, 'class_weight'):
                return np.sum(self.class_weight[y] * (y != y_hat))
            return np.sum(y != y_hat)
        elif y[1] is None:
            # fully labeled example
            return np.sum(y[0] != y_hat)
        else:
            # should use Kappa here
            loss = 0
            n_nodes = y[0].ravel().shape[0]
            c = n_nodes / float(self.n_states)
            for label in xrange(0, self.n_states):
                if label in y[1] and not np.any(y[0].ravel() == label):
                    loss += c
                elif label not in y[1]:
                    loss += np.sum(y[0] == label)
            return loss

    def loss_augmented_inference(self, x, y, w, relaxed=False,
                                 return_energy=False):
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self._get_unary_potentials(x, w)
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)

        if not isinstance(y, tuple) or y[1] is None:
            if isinstance(y, tuple) and y[1] is None:
                y = y[0]

            loss_augment_unaries(unary_potentials, np.asarray(y),
                                 self.class_weight)

            return inference_dispatch(unary_potentials, pairwise_potentials,
                                      edges, self.inference_method,
                                      relaxed=relaxed,
                                      return_energy=return_energy)
        else:
            #use pygco with label costs
            label_cost = np.zeros(self.n_states)
            n_nodes = unary_potentials.shape[0]
            c = n_nodes / float(self.n_states)
            for label in y[1]:
                label_cost[label] = c
            for label in xrange(0, self.n_states):
                if label not in y[1]:
                    unary_potentials[:, label] += 1

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

            y = cut_from_graph_gen_potts(unary_potentials, pairwise_cost,
                                         label_cost=label_cost)
            return y[0].reshape(shape_org)

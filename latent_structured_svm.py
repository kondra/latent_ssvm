######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# (c) 2013 Dmitry Kondrashkin <kondra2lp@gmail.com>
#


import numpy as np
from time import time

from pystruct.learners.ssvm import BaseSSVM
from sklearn.externals.joblib import Parallel, delayed

from common import latent

class LatentSSVM(BaseSSVM):
    """

    Parameters
    ----------
    base_ssvm : object
        SSVM solver instance for solving the completed problem.

    latent_iter : int (default=5)
        Number of iterations in the completion / refit loop.

    logger : object
        Logger instance.

    verbose : int (default=0)
        Verbosity level.

    tol : float (default=0.01)
        Tolerance, when to stop iterations of latent SSVM

    Attributes
    ----------
    w : nd-array, shape=(model.size_psi,)
        The learned weights of the SVM.
    """

    def __init__(self, base_ssvm, latent_iter=5, verbose=0, tol=0.1,
                 min_changes=0, n_jobs=1, logger=None):
        self.base_ssvm = base_ssvm
        self.latent_iter = latent_iter
        self.logger = logger
        self.verbose = verbose
        self.tol = tol
        self.n_jobs = n_jobs
        self.min_changes = min_changes

    def fit(self, X, Y, initialize=True):
        """Learn parameters using the concave-convex procedure.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels (full and weak)
            for inputs in X. Needs to have the same length as X.

        initialize : boolean
            Initialize w by running SSVM on full-labeled examples.
        """

        w = np.zeros(self.model.size_psi)
        constraints = None
        ws = []
        w_deltas = []
        changes_count = []
        start_time = time()
        timestamps = [0.0]

        # all data is fully labeled, quit
        if np.all([y.full_labeled for y in Y]):
            self.base_ssvm.fit(X, Y)
            return

        # we have some fully labeled examples, others are somehow initialized
        if initialize:
            X1 = []
            Y1 = []
            for x, y in zip(X, Y):
                if y.full_labeled:
                    X1.append(x)
                    Y1.append(y)

            self.base_ssvm.fit(X1, Y1)
            w = self.base_ssvm.w

        ws.append(w)

        for iteration in xrange(self.latent_iter):
            if self.verbose:
                print("LATENT SVM ITERATION %d" % iteration)
            # find latent variables for ground truth:
            Y_new = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(latent)(self.model, x, y, w) for x, y in zip(X, Y))

            changes = [np.any(y_new.full != y.full) for y_new, y in zip(Y_new, Y)]
            if np.sum(changes) < self.min_changes:
                if self.verbose:
                    print("too few changes in latent variables of ground truth."
                          " stopping.")
                iteration -= 1
                break
            if self.verbose:
                print("changes in H: %d" % np.sum(changes))
            changes_count.append(np.sum(changes))

# update constraints:
# seems that this code should not work
#            if isinstance(self.base_ssvm, NSlackSSVM):
#                constraints = [[] for i in xrange(len(X))]
#                for sample, h, i in zip(self.base_ssvm.constraints_, H_new,
#                                        np.arange(len(X))):
#                    for constraint in sample:
#                        const = find_constraint(self.model, X[i], h, w,
#                                                constraint[0])
#                        y_hat, dpsi, _, loss = const
#                        constraints[i].append([y_hat, dpsi, loss])
            Y = Y_new

            if iteration > 0:
                self.base_ssvm.fit(X, Y, constraints=constraints,
                                   warm_start="soft", initialize=False)
            else:
                self.base_ssvm.fit(X, Y, constraints=constraints,
                                   initialize=False)
            w = self.base_ssvm.w
            ws.append(w)
            delta = np.linalg.norm(ws[-1] - ws[-2])
            w_deltas.append(delta)
            if self.verbose:
                print("|w-w_prev|: %f" % delta)
            timestamps.append(time() - start_time)
            if self.verbose:
                print("time elapsed: %f s" % timestamps[-1])
            if delta < self.tol:
                if self.verbose:
                    print("weight vector did not change a lot, break")
                iteration += 1
                break
            if self.logger is not None:
                self.logger(self, iteration)

        self.ws = ws
        self.w_deltas = w_deltas
        self.changes_count = changes_count
        self.iter_done = iteration

    def _predict_from_iter(self, X, i):
        saved_w = self.base_ssvm.w
        self.base_ssvm.w = self.ws[i]
        Y_pred = self.base_ssvm.predict(X)
        self.base_ssvm.w = saved_w
        return Y_pred

    def staged_predict_latent(self, X):
        # is this ok?
        for i in xrange(self.iter_done):
            yield self._predict_from_iter(X, i)

    def staged_score(self, X, Y):
        # is this ok?
        for i in xrange(self.iter_done):
            Y_pred = self._predict_from_iter(X, i)
            losses = [self.model.loss(y, y_pred) / np.sum(y.weights)
                      for y, y_pred in zip(Y, Y_pred)]
            score = 1. - np.sum(losses) / len(X)
            yield score

#    def predict(self, X):
#        prediction = self.base_ssvm.predict(X)
#        return [self.model.label_from_latent(h) for h in prediction]

    def predict_latent(self, X):
        return self.base_ssvm.predict(X)

    def score(self, X, Y):
        """Compute score as 1 - loss over whole data set.

        Returns the average accuracy (in terms of model.loss)
        over X and Y.

        Parameters
        ----------
        X : iterable
            Evaluation data.

        Y : iterable
            True labels.

        Returns
        -------
        score : float
            Average of 1 - loss over training examples.
        """
        # TODO: rewrite this
        #losses = [self.model.base_loss(y, y_pred)
        #          for y, y_pred in zip(Y, self.predict(X))]
        #max_losses = [self.model.max_loss(y) for y in Y]
        #return 1. - np.sum(losses) / float(np.sum(max_losses))

        losses = [self.model.loss(y, y_pred) / np.sum(y.weights)
                  for y, y_pred in zip(Y, self.predict_latent(X))]
        return 1. - np.sum(losses) / len(X)

    @property
    def model(self):
        return self.base_ssvm.model

    @model.setter
    def model(self, model_):
        self.base_ssvm.model = model_

    @property
    def w(self):
        return self.base_ssvm.w

    @w.setter
    def w(self, w_):
        self.base_ssvm.w = w_

    @property
    def C(self):
        return self.base_ssvm.C

    @C.setter
    def C(self, C_):
        self.base_ssvm.w = C_

    @property
    def n_jobs(self):
        return self.base_ssvm.n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs_):
        self.base_ssvm.n_jobs = n_jobs_

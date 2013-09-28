######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# (c) 2013 Dmitry Kondrashkin <kondra2lp@gmail.com>
#


import numpy as np

from pystruct.learners.ssvm import BaseSSVM
from pystruct.learners.n_slack_ssvm import NSlackSSVM
from pystruct.utils import find_constraint


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

    def __init__(self, base_ssvm, latent_iter=5, verbose=0, tol=0.01, logger=None):
        self.base_ssvm = base_ssvm
        self.latent_iter = latent_iter
        self.logger = logger
        self.verbose = verbose
        self.tol = tol

    def fit(self, X, Y, H_def, initialize=True, pass_labels=False):
        """Learn parameters using the concave-convex procedure.

        Parameters
        ----------
        X : iterable
            Traing instances. Contains the structured input objects.
            No requirement on the particular form of entries of X is made.

        Y : iterable
            Training labels. Contains the strctured labels for inputs in X.
            Needs to have the same length as X.

        H_def: iterable
            Labels for full-labeles samples. (known hidden variables)
            Used for heterogenous training.

        initialize: boolean
            Initialize w by running SSVM on full-labeled examples.

        pass_labels: boolean
            Pass hidden and original labels to SSVM solver as a tuple.
            Only use this if you know what you are doing!
        """

        self.model.initialize(X, Y)
        w = np.zeros(self.model.size_psi)
        constraints = None
        ws = []
        w_deltas = []
        changes_count = []

        X1 = []
        H1 = []
        H = H_def

        for i, h in enumerate(H_def):
            if h is not None:
                X1.append(X[i])
                H1.append(h)
                Y[i] = None

        # all data is fully labeled, quit
        if len(X1) == len(X):
            self.base_ssvm.fit(X1, H1)
            w = self.base_ssvm.w
            return

        # we have some fully labeled examples
        if initialize and len(X1) > 0:
            saved_C = self.base_ssvm.C
            self.base_ssvm.C = 10
            self.base_ssvm.fit(X1, H1)
            w = self.base_ssvm.w
            self.base_ssvm.C = saved_C

        ws.append(w)

        for iteration in xrange(self.latent_iter):
            if self.verbose:
                print("LATENT SVM ITERATION %d" % iteration)
            # find latent variables for ground truth:
            H_new = []
            for x, y, h in zip(X, Y, H_def):
                if h is None:
                    H_new.append(self.model.latent(x, y, w))
                else:
                    H_new.append(h)

            changes = [np.any(h_new != h) for h_new, h in zip(H_new, H)]
            if not np.any(changes):
                if self.verbose:
                    print("no changes in latent variables of ground truth."
                          " stopping.")
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
            H = H_new

            if pass_labels:
                T = zip(H, Y)
            else:
                T = H

            if iteration > 0:
                self.base_ssvm.fit(X, T, constraints=constraints,
                                   warm_start="soft", initialize=False)
            else:
                self.base_ssvm.fit(X, T, constraints=constraints,
                                   initialize=False)
            w = self.base_ssvm.w
            ws.append(w)
            delta = np.linalg.norm(ws[-1] - ws[-2])
            w_deltas.append(delta)
            if self.verbose:
                print("|w-w_prev|: %f" % delta)
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

    def staged_predict_latent(self, X):
        # is this ok?
        for i in xrange(self.iter_done):
            saved_w = self.base_ssvm.w
            self.base_ssvm.w = self.ws[i]
            H = self.base_ssvm.predict(X)
            self.base_ssvm.w = saved_w
            yield H

    def predict(self, X):
        prediction = self.base_ssvm.predict(X)
        return [self.model.label_from_latent(h) for h in prediction]

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
        losses = [self.model.base_loss(y, y_pred)
                  for y, y_pred in zip(Y, self.predict(X))]
        max_losses = [self.model.max_loss(y) for y in Y]
        return 1. - np.sum(losses) / float(np.sum(max_losses))

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

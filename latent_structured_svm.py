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
                 min_changes=0, n_jobs=1):
        self.base_ssvm = base_ssvm
        self.latent_iter = latent_iter
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
            If True initialize w by running SSVM on full-labeled examples.
            Otherwise initialize w by running SSVM on full-labeled and
            randomly initialized weak-labeled examples.
        """

        w = np.zeros(self.model.size_psi)
        constraints = None
        self.w_history_ = []
        self.delta_history_ = []
        self.base_iter_history_ = []
        self.changes_ = []
        start_time = time()
        self.timestamps_ = [0.0]
        self.qp_timestamps_ = []
        self.inference_timestamps_ = []
        self.number_of_constraints_ = []
        self.objective_curve_ = []
        self.primal_objective_curve_ = []

        # all data is fully labeled, quit
        if np.all([y.full_labeled for y in Y]):
            self.base_ssvm.fit(X, Y)
            self.w_history_ = np.array([self.base_ssvm.w])
            self.delta_history_ = np.array([])
            self.changes_ = np.array([])
            self.timestamps_ = np.array([time() - start_time])
            self.primal_objective_curve_ = np.array([self.base_ssvm.primal_objective_curve_[-1]])
            self.objective_curve_ = np.array([self.base_ssvm.objective_curve_[-1]])
            self.base_iter_history_ = np.array([len(self.base_ssvm.primal_objective_curve_)])
            self.iter_done = 1
            return

        X1, Y1 = [], []
        if initialize:
            for x, y in zip(X, Y):
                if y.full_labeled:
                    X1.append(x)
                    Y1.append(y)
        else:
            # we have some fully labeled examples, others are somehow initialized
            X1, Y1 = X, Y

        start_t = time()
        self.base_ssvm.fit(X1, Y1)
        stop_t = time()
        w = self.base_ssvm.w
        self.w_history_.append(w)
        self.objective_curve_.append(self.base_ssvm.objective_curve_[-1])
        self.primal_objective_curve_.append(self.base_ssvm.primal_objective_curve_[-1])
        self.base_iter_history_.append(len(self.base_ssvm.primal_objective_curve_))
        gap = self.primal_objective_curve_[-1] - self.objective_curve_[-1]

        print("Final primal objective: %f" % self.primal_objective_curve_[-1])
        print("Final cutting-plane objective: %f" % self.objective_curve_[-1])
        print("Duality gap: %f" % gap)
        print("Finished in %d iterations" % self.base_iter_history_[-1])
        print("Time elapsed: %f s" % (stop_t - start_t))
        print("Time spent by QP: %f s" % self.base_ssvm.qp_time)
        print("Time spent by inference: %f s" % self.base_ssvm.inference_time)
        print("Number of constraints: %d" % self.number_of_constraints_[-1])
        print("----------------------------------------")

        try:
            for iteration in xrange(self.latent_iter):
                if self.verbose:
                    print("LATENT SVM ITERATION %d" % iteration)
                # complete latent variables
                Y_new = Parallel(n_jobs=self.n_jobs, verbose=0, max_nbytes=1e8)(
                    delayed(latent)(self.model, x, y, w) for x, y in zip(X, Y))
    
                changes = [np.any(y_new.full != y.full) for y_new, y in zip(Y_new, Y)]
                if np.sum(changes) <= self.min_changes:
                    if self.verbose:
                        print("too few changes in latent variables of ground truth."
                              " stopping.")
                    iteration -= 1
                    break
                if self.verbose:
                    print("changes in H: %d" % np.sum(changes))
                self.changes_.append(np.sum(changes))
    
                Y = Y_new
                #if iteration > 1:
                #    self.base_ssvm.fit(X, Y, initialize=False, warm_start='soft')
                #else:
                self.base_ssvm.fit(X, Y, initialize=False)

                w = self.base_ssvm.w
                self.objective_curve_.append(self.base_ssvm.objective_curve_[-1])
                self.primal_objective_curve_.append(self.base_ssvm.primal_objective_curve_[-1])
                self.base_iter_history_.append(len(self.base_ssvm.primal_objective_curve_))
                self.w_history_.append(w)
                delta = np.linalg.norm(self.w_history_[-1] - self.w_history_[-2])
                self.delta_history_.append(delta)
                gap = self.primal_objective_curve_[-1] - self.objective_curve_[-1]
                self.timestamps_.append(time() - start_time)
                self.qp_timestamps_.append(self.base_ssvm.qp_time)
                self.inference_timestamps_.append(self.base_ssvm.inference_time)
                self.number_of_constraints_.append(len(self.base_ssvm.constraints_))
                if self.verbose:
                    print("|w-w_prev|: %f" % delta)
                    print("Final primal objective: %f" % self.primal_objective_curve_[-1])
                    print("Final cutting-plane objective: %f" % self.objective_curve_[-1])
                    print("Duality gap: %f" % gap)
                    print("Finished in %d iterations" % self.base_iter_history_[-1])
                    print("Time elapsed: %f s" % (self.timestamps_[-1] - self.timestamps_[-2]))
                    print("Time spent by QP: %f s" % self.base_ssvm.qp_time)
                    print("Time spent by inference: %f s" % self.base_ssvm.inference_time)
                    print("Number of constraints: %d" % self.number_of_constraints_[-1])
                    print("----------------------------------------")
                if delta < self.tol:
                    if self.verbose:
                        print("weight vector did not change a lot, break")
                    break
        except KeyboardInterrupt:
            pass

        self.w_history_ = np.array(self.w_history_)
        self.delta_history_ = np.array(self.delta_history_)
        self.changes_ = np.array(self.changes_)
        self.timestamps_ = np.array(self.timestamps_)
        self.primal_objective_curve_ = np.array(self.primal_objective_curve_)
        self.objective_curve_ = np.array(self.objective_curve_)
        self.base_iter_history_ = np.array(self.base_iter_history_)
        self.iter_done = iteration + 1

    def _predict_from_iter(self, X, i):
        saved_w = self.base_ssvm.w
        self.base_ssvm.w = self.w_history_[i]
        Y_pred = self.base_ssvm.predict(X)
        self.base_ssvm.w = saved_w
        return Y_pred

    def staged_predict_latent(self, X):
        for i in xrange(self.iter_done):
            yield self._predict_from_iter(X, i)

    def staged_score(self, X, Y):
        for i in xrange(self.iter_done):
            Y_pred = self._predict_from_iter(X, i)
            losses = [self.model.loss(y, y_pred) / float(np.sum(y.weights))
                      for y, y_pred in zip(Y, Y_pred)]
            score = 1. - np.sum(losses) / float(len(X))
            yield score

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
        losses = [self.model.loss(y, y_pred) / float(np.sum(y.weights))
                  for y, y_pred in zip(Y, self.predict_latent(X))]
        return 1. - np.sum(losses) / float(len(X))

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

######################
# (c) 2012 Andreas Mueller <amueller@ais.uni-bonn.de>
# (c) 2013 Dmitry Kondrashkin <kondra2lp@gmail.com>
#


import numpy as np
from time import time

from pystruct.learners.ssvm import BaseSSVM
from joblib import Parallel, delayed

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

    def fit(self, X, Y, initialize=True, continued=False, warm_start=False):
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

        continued : boolean
            If True than it is assumed that every internal model data are set up.
            And we continue learning. It may be used to perform additional iterations
            without restarting the method.
        """

        if not continued:
            w = np.zeros(self.model.size_psi)
            start_time = time()
            self.w_history_ = []
            self.number_of_iterations_ = []
            self.number_of_changes_ = []
            self.timestamps_ = []
            self.qp_time_ = []
            self.inference_time_ = []
            self.number_of_constraints_ = []
            self.objective_curve_ = []
            self.primal_objective_curve_ = []
            self.inference_calls_ = []
            self.latent_objective_ = []

            self.inner_w = []
            self.inner_sz = []
            self.inner_objective = []
            self.inner_primal = []
            self.inner_staged_inference = []

            # all data is fully labeled, quit
            # it should not work!
            if np.all([y.full_labeled for y in Y]):
                self.base_ssvm.fit(X, Y)
                self.w_history_ = np.array([self.base_ssvm.w])
                self.number_of_iterations_ = np.array([len(self.base_ssvm.primal_objective_curve_)])
                self.number_of_changes_ = np.array([])
                self.timestamps_ = np.array([time() - start_time])
                self.qp_time_ = np.array([self.base_ssvm.qp_time])
                self.inference_time_ = np.array([self.base_ssvm.inference_time])
                self.number_of_constraints_ = np.array([len(self.base_ssvm.constraints_)])
                self.objective_curve_ = np.array([self.base_ssvm.objective_curve_[-1]])
                self.primal_objective_curve_ = np.array([self.base_ssvm.primal_objective_curve_[-1]])
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

            # now it's a dirty hack
            # we'd like to find a good initialstarting point, let ssvm converge
            old_max_iter = None
            if warm_start:
                old_max_iter = self.base_ssvm.max_iter
                self.base_ssvm.max_iter = 10000

            self.base_ssvm.fit(X1, Y1, save_history=True)

            if warm_start:
                self.base_ssvm.max_iter = old_max_iter

            w = self.base_ssvm.w

            self.w_history_.append(w)
            self.number_of_iterations_.append(self.base_ssvm.iterations_done)
            self.timestamps_.append(time() - start_time)
            self.qp_time_.append(self.base_ssvm.qp_time)
            self.inference_time_.append(self.base_ssvm.inference_time)
            self.number_of_constraints_.append(len(self.base_ssvm.constraints_))
            self.objective_curve_.append(self.base_ssvm.objective_curve_[-1])
            self.primal_objective_curve_.append(self.base_ssvm.primal_objective_curve_[-1])
            self.inference_calls_.append(self.base_ssvm.inference_calls)
            gap = self.primal_objective_curve_[-1] - self.objective_curve_[-1]

            self.inner_w.append(self.base_ssvm.w_history)
            self.inner_sz.append(self.base_ssvm.w_history.shape[0])
            self.inner_objective += self.base_ssvm.objective_curve_
            self.inner_primal += self.base_ssvm.primal_objective_curve_
            self.inner_staged_inference += self.base_ssvm.staged_inference_calls

            if self.verbose:
                print("Final primal objective: %f" % self.primal_objective_curve_[-1])
                print("Final cutting-plane objective: %f" % self.objective_curve_[-1])
                print("Duality gap: %f" % gap)
                print("Finished in %d iterations" % self.number_of_iterations_[-1])
                print("Time elapsed: %f s" % (self.timestamps_[-1]))
                print("Time spent by QP: %f s" % self.base_ssvm.qp_time)
                print("Time spent by inference: %f s" % self.base_ssvm.inference_time)
                print("Number of constraints: %d" % self.number_of_constraints_[-1])
                print("----------------------------------------")

            begin = 0
        else:
            begin = self.iter_done
            w = self.w_history_[-1]
            start_time = time()

        try:
            for iteration in xrange(begin, self.latent_iter):
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
                self.number_of_changes_.append(np.sum(changes))
    
                Y = Y_new

                latent_objective = self.base_ssvm.fit(X, Y, only_objective=True,
                                                      previous_w=self.w_history_[-1])
                self.latent_objective_.append(latent_objective)
                if self.verbose:
                    print("Previous Latent SSVM objective: %f" % self.latent_objective_[-1])

                if not warm_start:
                    self.base_ssvm.fit(X, Y, warm_start=False, initialize=False, save_history=True)
                else:
                    if iteration > 0:
                        self.base_ssvm.fit(X, Y, warm_start=warm_start,
                                           initialize=False, save_history=True)
                    else:
                        self.base_ssvm.fit(X, Y, warm_start=False,
                                           initialize=False, save_history=True)

                w = self.base_ssvm.w

                self.w_history_.append(w)
                self.number_of_iterations_.append(self.base_ssvm.iterations_done)
                self.timestamps_.append(time() - start_time)
                self.qp_time_.append(self.base_ssvm.qp_time)
                self.inference_time_.append(self.base_ssvm.inference_time)
                self.number_of_constraints_.append(len(self.base_ssvm.constraints_))
                self.objective_curve_.append(self.base_ssvm.objective_curve_[-1])
                self.primal_objective_curve_.append(self.base_ssvm.primal_objective_curve_[-1])
                self.inference_calls_.append(self.base_ssvm.inference_calls)
                delta = np.linalg.norm(self.w_history_[-1] - self.w_history_[-2])
                gap = self.primal_objective_curve_[-1] - self.objective_curve_[-1]
                q_delta = np.abs(self.primal_objective_curve_[-1] - self.primal_objective_curve_[-2])

                self.inner_w.append(self.base_ssvm.w_history)
                self.inner_sz.append(self.base_ssvm.w_history.shape[0])
                self.inner_objective += self.base_ssvm.objective_curve_
                self.inner_primal += self.base_ssvm.primal_objective_curve_
                self.inner_staged_inference += self.base_ssvm.staged_inference_calls

                if self.verbose:
                    print("|w-w_prev|: %f" % delta)
                    print("|Q-Q_prev|: %f" % q_delta)
                    print("Final primal objective: %f" % self.primal_objective_curve_[-1])
                    print("Final cutting-plane objective: %f" % self.objective_curve_[-1])
                    print("Duality gap: %f" % gap)
                    print("Finished in %d iterations" % self.number_of_iterations_[-1])
                    print("Time elapsed: %f s" % (self.timestamps_[-1] - self.timestamps_[-2]))
                    print("Time spent by QP: %f s" % self.base_ssvm.qp_time)
                    print("Time spent by inference: %f s" % self.base_ssvm.inference_time)
                    print("Number of constraints: %d" % self.number_of_constraints_[-1])
                    print("----------------------------------------")

                if q_delta < self.tol:
                    if self.verbose:
                        print("objective value did not change a lot, break")
                    break
        except KeyboardInterrupt:
            if self.verbose:
                print('interrupted... finishing...')
            pass

        self.number_of_changes_ = np.array(self.number_of_changes_)
        self.w_history_ = np.array(self.w_history_)
        self.number_of_iterations_ = np.array(self.number_of_iterations_)
        self.timestamps_ = np.array(self.timestamps_)
        self.qp_time_ = np.array(self.qp_time_)
        self.inference_time_ = np.array(self.inference_time_)
        self.number_of_constraints_ = np.array(self.number_of_constraints_)
        self.objective_curve_ = np.array(self.objective_curve_)
        self.primal_objective_curve_ = np.array(self.primal_objective_curve_)
        self.inference_calls_ = np.array(self.inference_calls_)
        self.latent_objective_ = np.array(self.latent_objective_)
        self.iter_done = iteration + 1

        self.inner_w = np.vstack(self.inner_w)
        self.inner_sz = np.array(self.inner_sz)
        self.inner_primal = np.array(self.inner_primal)
        self.inner_objective = np.array(self.inner_objective)
        self.inner_staged_inference = np.array(self.inner_staged_inference)

    def _get_data(self):
        # get all model data as a dict
        data = {}
        data['changes'] = self.number_of_changes_
        data['w_history'] = self.w_history_
        data['number_of_iterations'] = self.number_of_iterations_
        data['timestamps'] = self.timestamps_
        data['qp_timestamps'] = self.qp_time_
        data['inference_timestamps'] = self.inference_time_
        data['inference_calls'] = self.inference_calls_
        data['number_of_constraints'] = self.number_of_constraints_
        data['primal_objective_curve'] = self.primal_objective_curve_
        data['objective_curve'] = self.objective_curve_
        data['latent_objective'] = self.latent_objective_

        data['inner_w'] = self.inner_w
        data['inner_sz'] = self.inner_sz
        data['inner_objective'] = self.inner_objective
        data['inner_primal'] = self.inner_primal
        data['inner_staged_inference'] = self.inner_staged_inference

        return data

    def _load_data(self, data):
        # loads model data from dict
        self.number_of_changes_ = list(data['changes'])
        self.w_history_ = list(data['w_history'])
        self.number_of_iterations_  = list(data['number_of_iterations'])
        self.timestamps_ = list(data['timestamps'])
        self.qp_time_ = list(data['qp_timestamps'])
        self.inference_time_ = list(data['inference_timestamps'])
        self.number_of_constraints_ = list(data['number_of_constraints'])
        self.primal_objective_curve_ = list(data['primal_objective_curve'])
        self.objective_curve_ = list(data['objective_curve'])
        self.inference_calls_ = list(data['inference_calls'])
        self.iter_done = len(self.objective_curve_)

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

    def staged_score2(self, X, Y):
        for i in xrange(self.iter_done):
            Y_pred = self._predict_from_iter(X, i)
            losses = [self.model.loss(y, y_pred) for y, y_pred in zip(Y, Y_pred)]
            yield np.mean(losses)

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

    def staged_latent_objective(self, X, Y):
        for i in xrange(self.iter_done):
            w = self.w_history_[i]
            Y = Parallel(n_jobs=self.n_jobs, verbose=0, max_nbytes=1e8)(
                delayed(latent)(self.model, x, y, w) for x, y in zip(X, Y))
            yield self.base_ssvm.fit(X, Y, only_objective=True, previous_w=w)

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

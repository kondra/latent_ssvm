import numpy as np

from pystruct.learners import OneSlackSSVM
from time import time

from latent_crf import LatentCRF
from latent_structured_svm import LatentSSVM
from heterogenous_crf import HCRF

from data_loader import load_syntetic
from data_loader import load_msrc
from common import compute_error
from common import weak_from_hidden
from label import Label
from results import ExperimentResult

# testing with weakly labeled train set


def test_syntetic_weak(mode):
    # needs refactoring
    # Syntetic data
    # test latentSSVM on different train set sizes & on different train sets
    # mode can be 'heterogenous' or 'latent'
    results = np.zeros((18, 6))
    full_labeled = np.array([0, 2, 4, 10, 25, 100])
    train_size = 400

    for dataset in xrange(1, 19):
        X, H = load_syntetic(dataset)
        H = list(H)
        Y = weak_from_hidden(H)

        for j, nfull in enumerate(full_labeled):
            if mode == 'latent':
                crf = LatentCRF(n_states=10, n_features=10, n_edge_features=2,
                                inference_method='qpbo')
                base_clf = OneSlackSSVM(crf, max_iter=100, C=0.01, verbose=0,
                                        tol=0.1, n_jobs=4, inference_cache=100)
                clf = LatentSSVM(base_clf, latent_iter=5)
            elif mode == 'heterogenous':
                crf = HCRF(n_states=10, n_features=10, n_edge_features=2,
                           inference_method='gco')
                base_clf = OneSlackSSVM(crf, max_iter=500, C=0.1, verbose=0,
                                        tol=0.001, n_jobs=4, inference_cache=100)
                clf = LatentSSVM(base_clf, latent_iter=5, verbose=0)

            x_train = X[:train_size]
            y_train = Y[:train_size]
            h_train = H[:train_size]
            x_test = X[(train_size + 1):]
            h_test = H[(train_size + 1):]

            for i in xrange(nfull, len(h_train)):
                h_train[i] = None

            try:
                if mode == 'latent':
                    clf.fit(x_train, y_train, h_train)
                elif mode == 'heterogenous':
                    clf.fit(x_train, y_train, h_train,
                            pass_labels=True, initialize=True)
                h_pred = clf.predict_latent(x_test)

                results[dataset - 1, j] = compute_error(h_test, h_pred)

                print 'dataset=%d, nfull=%d, error=%f' % (dataset,
                                                          nfull,
                                                          results[dataset - 1, j])
            except ValueError:
                # bad QP
                print 'dataset=%d, nfull=%d: Failed' % (dataset, nfull)

    if mode == 'latent':
        np.savetxt('results/weak_labeled.csv', results, delimiter=',')
    elif mode == 'heterogenous':
        np.savetxt('results/heterogenous.csv', results, delimiter=',')

    return results


def split_test_train(X, Y, n_full, n_train):
    x_train = X[:n_train]
    y_train = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
               for y in Y[:n_full]]
    y_train += [Label(None, np.unique(y[:, 0].astype(np.int32)),
                      y[:, 1], False) for y in Y[(n_full):(n_train)]]
    y_train_full = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
                    for y in Y[:n_train]]

    x_test = X[n_train:]
    y_test = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
              for y in Y[n_train:]]

    return x_train, y_train, y_train_full, x_test, y_test


def syntetic_weak(n_full=10, n_train=200, C=0.1, dataset=1, latent_iter=15,
                  max_iter=500, inner_tol=0.001, outer_tol=0.01, min_changes=0,
                  initialize=True, alpha=0.1, n_inference_iter=5,
                  inactive_window=50, inactive_threshold=1e-5):
    crf = HCRF(n_states=10, n_features=10, n_edge_features=2, alpha=alpha,
               inference_method='gco')
    base_clf = OneSlackSSVM(crf, max_iter=max_iter, C=C, verbose=0,
                            tol=inner_tol, n_jobs=4, inference_cache=100,
                            inactive_window=50, inactive_threshold=1e-5)
    clf = LatentSSVM(base_clf, latent_iter=latent_iter, verbose=2,
                     tol=outer_tol, min_changes=min_changes, n_jobs=4)

    X, Y = load_syntetic(dataset)

    x_train, y_train, y_train_full, x_test, y_test = \
        split_test_train(X, Y, n_full, n_train)

    start = time()
    clf.fit(x_train, y_train, initialize=initialize)
    stop = time()

    train_score = clf.score(x_train, y_train_full)
    test_score = clf.score(x_test, y_test)
    time_elapsed = stop - start

    print '============================================================'
    print 'Score on train set: %f' % train_score
    print 'Score on test set: %f' % test_score
    print 'Norm of weight vector: |w|=%f' % np.linalg.norm(clf.w)
    print 'Elapsed time: %f s' % time_elapsed

    test_scores = []
    for score in clf.staged_score(x_test, y_test):
        test_scores.append(score)

    exp_data = {}
    exp_data['test_scores'] = np.array(test_scores)
    exp_data['changes'] = clf.changes_
    exp_data['w_history'] = clf.w_history_
    exp_data['delta_history'] = clf.delta_history_
    exp_data['primal_objective_curve'] = clf.primal_objective_curve_
    exp_data['objective_curve'] = clf.objective_curve_
    exp_data['timestamps'] = clf.timestamps_
    exp_data['qp_timestamps'] = clf.qp_timestamps_
    exp_data['inference_timestamps'] = clf.inference_timestamps_
    exp_data['base_iter_hitory'] = clf.base_iter_history_
    exp_data['number_of_constraints'] = clf.number_of_constraints_

    meta_data = {}
    meta_data['dataset_name'] = 'syntetic'
    meta_data['annotation_type'] = 'image-level labelling'
    meta_data['label_type'] = 'full+weak'
    meta_data['train_score'] = train_score
    meta_data['test_score'] = test_score
    meta_data['time_elapsed'] = time_elapsed

    meta_data['n_inference_iter'] = n_inference_iter
    meta_data['n_full'] = n_full
    meta_data['n_train'] = n_train
    meta_data['C'] = C
    meta_data['dataset'] = dataset
    meta_data['latent_iter'] = latent_iter
    meta_data['max_iter'] = max_iter
    meta_data['inner_tol'] = inner_tol
    meta_data['outer_tol'] = outer_tol
    meta_data['alpha'] = alpha
    meta_data['min_changes'] = min_changes
    meta_data['initialize'] = initialize
    meta_data['inactive_window'] = inactive_window
    meta_data['inactive_threshold'] = inactive_threshold

    return ExperimentResult(exp_data, meta_data)


def msrc_load(n_full, n_train):
    Xtest, Ytest = load_msrc('test')
    Ytest = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
             for y in Ytest]

    train_mask = np.genfromtxt('../data/msrc/trainmasks/trainMaskX%d.txt' % n_full)
    train_mask = train_mask[0:n_train].astype(np.bool)
    Xtrain, Ytrain_raw = load_msrc('train')
    Ytrain_full = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
                   for y in Ytrain_raw]
    Ytrain = []
    for y, f in zip(Ytrain_raw, train_mask):
        if f:
            Ytrain.append(Label(y[:, 0].astype(np.int32),
                                None, y[:, 1], True))
        else:
            Ytrain.append(Label(None, np.unique(y[:, 0].astype(np.int32)),
                                y[:, 1], False))

    return Xtrain, Ytrain, Ytrain_full, Xtest, Ytest


def msrc_weak(n_full=20, n_train=276, C=100, latent_iter=25,
              max_iter=500, inner_tol=0.001, outer_tol=0.01, min_changes=0,
              initialize=True, alpha=0.1, n_inference_iter=5,
              inactive_window=50, inactive_threshold=1e-5):
    crf = HCRF(n_states=24, n_features=2028, n_edge_features=4, alpha=alpha,
               inference_method='gco', n_iter=n_inference_iter)
    base_clf = OneSlackSSVM(crf, max_iter=max_iter, C=C, verbose=0,
                            tol=inner_tol, n_jobs=4, inference_cache=10,
                            inactive_window=50, inactive_threshold=1e-5)
    clf = LatentSSVM(base_clf, latent_iter=latent_iter, verbose=2,
                     tol=outer_tol, min_changes=min_changes, n_jobs=4)

    Xtrain, Ytrain, Ytrain_full, Xtest, Ytest = msrc_load(n_full, n_train)

    start = time()
    clf.fit(Xtrain, Ytrain, initialize=initialize)
    stop = time()

    train_score = clf.score(Xtrain, Ytrain_full)
    test_score = clf.score(Xtest, Ytest)
    time_elapsed = stop - start 

    print '============================================================'
    print 'Score on train set: %f' % train_score
    print 'Score on test set: %f' % test_score
    print 'Norm of weight vector: |w|=%f' % np.linalg.norm(clf.w)
    print 'Elapsed time: %f s' % time_elapsed

    test_scores = []
    for score in clf.staged_score(Xtest, Ytest):
        test_scores.append(score)

    exp_data = {}
    exp_data['test_scores'] = np.array(test_scores)
    exp_data['changes'] = clf.changes_
    exp_data['w_history'] = clf.w_history_
    exp_data['delta_history'] = clf.delta_history_
    exp_data['primal_objective_curve'] = clf.primal_objective_curve_
    exp_data['objective_curve'] = clf.objective_curve_
    exp_data['timestamps'] = clf.timestamps_
    exp_data['qp_timestamps'] = clf.qp_timestamps_
    exp_data['inference_timestamps'] = clf.inference_timestamps_
    exp_data['base_iter_hitory'] = clf.base_iter_history_
    exp_data['number_of_constraints'] = clf.number_of_constraints_

    meta_data = {}
    meta_data['dataset_name'] = 'msrc'
    meta_data['annotation_type'] = 'image-level labelling'
    meta_data['label_type'] = 'full+weak'
    meta_data['train_score'] = train_score
    meta_data['test_score'] = test_score
    meta_data['time_elapsed'] = time_elapsed

    meta_data['n_inference_iter'] = n_inference_iter
    meta_data['n_full'] = n_full
    meta_data['n_train'] = n_train
    meta_data['C'] = C
    meta_data['latent_iter'] = latent_iter
    meta_data['max_iter'] = max_iter
    meta_data['inner_tol'] = inner_tol
    meta_data['outer_tol'] = outer_tol
    meta_data['alpha'] = alpha
    meta_data['min_changes'] = min_changes
    meta_data['initialize'] = initialize
    meta_data['inactive_window'] = inactive_window
    meta_data['inactive_threshold'] = inactive_threshold

    return ExperimentResult(exp_data, meta_data)

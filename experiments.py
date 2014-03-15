import numpy as np

from pystruct.learners import FrankWolfeSSVM
from one_slack_ssvm import OneSlackSSVM
from time import time

from latent_structured_svm import LatentSSVM
from heterogenous_crf import HCRF

from data_loader import load_syntetic
from data_loader import load_msrc_hdf
from data_loader import load_msrc_weak_train_mask
from label import Label
from results import ExperimentResult, experiment


def split_test_train(X, Y, n_full, n_train):
    # splitting syntetic dataset
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


def load_msrc(n_full, n_train):
    # loading & splitting MSRC dataset
    MSRC_DATA_PATH = '/home/dmitry/Documents/Thesis/data/msrc/msrc.hdf5'

    Xtrain, Ytrain_raw, Xtest, Ytest = load_msrc_hdf(MSRC_DATA_PATH)
    Ytest = [Label(y[:, 0].astype(np.int32), None,
                   y[:, 1].astype(np.float64) / np.sum(y[:, 1]), True)
             for y in Ytest]

    train_mask = load_msrc_weak_train_mask(MSRC_DATA_PATH, n_full)[:n_train]
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


@experiment
def syntetic_weak(n_full=10, n_train=200, C=0.1, dataset=1, latent_iter=15,
                  max_iter=500, inner_tol=0.001, outer_tol=0.01, min_changes=0,
                  initialize=True, alpha=0.1, n_inference_iter=5,
                  inactive_window=50, inactive_threshold=1e-5,
                  warm_start=False, inference_cache=0,
                  save_inner_w=False, inference_method='gco'):
    # save parameters as meta
    meta_data = locals()

    crf = HCRF(n_states=10, n_features=10, n_edge_features=2, alpha=alpha,
               inference_method=inference_method, n_iter=n_inference_iter)
    base_clf = OneSlackSSVM(crf, verbose=2, n_jobs=4,
                            max_iter=max_iter, tol=inner_tol, C=C,
                            inference_cache=inference_cache,
                            inactive_window=inactive_window,
                            inactive_threshold=inactive_threshold)
    clf = LatentSSVM(base_clf, latent_iter=latent_iter, verbose=2,
                     tol=outer_tol, min_changes=min_changes, n_jobs=4)

    X, Y = load_syntetic(dataset)
    x_train, y_train, y_train_full, x_test, y_test = \
        split_test_train(X, Y, n_full, n_train)

    start = time()
    clf.fit(x_train, y_train,
            initialize=initialize, warm_start=warm_start,
            save_inner_w=save_inner_w)
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

    train_scores = []
    for score in clf.staged_score(x_train, y_train_full):
        train_scores.append(score)

    raw_scores = []
    for score in clf.staged_score2(x_train, y_train):
        raw_scores.append(score)

    exp_data = clf._get_data()
    exp_data['test_scores'] = np.array(test_scores)
    exp_data['train_scores'] = np.array(train_scores)
    exp_data['raw_scores'] = np.array(raw_scores)

    meta_data['dataset_name'] = 'syntetic'
    meta_data['annotation_type'] = 'image-level labelling'
    meta_data['label_type'] = 'full+weak'
    meta_data['train_score'] = train_score
    meta_data['test_score'] = test_score
    meta_data['time_elapsed'] = time_elapsed
    meta_data['iter_done'] = clf.iter_done

    return ExperimentResult(exp_data, meta_data)


@experiment
def msrc_weak(n_full=20, n_train=276, C=100, latent_iter=25,
              max_iter=500, inner_tol=0.001, outer_tol=0.01, min_changes=0,
              initialize=True, alpha=0.1, n_inference_iter=5,
              inactive_window=50, inactive_threshold=1e-5,
              warm_start=False, inference_cache=0,
              save_inner_w=False, inference_method='gco'):
    meta_data = locals()

    crf = HCRF(n_states=24, n_features=2028, n_edge_features=4, alpha=alpha,
               inference_method=inference_method, n_iter=n_inference_iter)
    base_clf = OneSlackSSVM(crf, verbose=2, n_jobs=4,
                            tol=inner_tol, max_iter=max_iter, C=C,
                            inference_cache=inference_cache,
                            inactive_window=inactive_window,
                            inactive_threshold=inactive_threshold)
    clf = LatentSSVM(base_clf, latent_iter=latent_iter, verbose=2,
                     tol=outer_tol, min_changes=min_changes, n_jobs=4)

    Xtrain, Ytrain, Ytrain_full, Xtest, Ytest = load_msrc(n_full, n_train)

    start = time()
    clf.fit(Xtrain, Ytrain,
            initialize=initialize,
            warm_start=warm_start,
            save_inner_w=save_inner_w)
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

    train_scores = []
    for score in clf.staged_score(Xtrain, Ytrain_full):
        train_scores.append(score)

    raw_scores = []
    for score in clf.staged_score2(Xtrain, Ytrain):
        raw_scores.append(score)

    exp_data = clf._get_data()
    exp_data['test_scores'] = np.array(test_scores)
    exp_data['train_scores'] = np.array(train_scores)
    exp_data['raw_scores'] = np.array(raw_scores)

    meta_data['dataset_name'] = 'msrc'
    meta_data['annotation_type'] = 'image-level labelling'
    meta_data['label_type'] = 'full+weak'
    meta_data['train_score'] = train_score
    meta_data['test_score'] = test_score
    meta_data['time_elapsed'] = time_elapsed
    meta_data['iter_done'] = clf.iter_done

    return ExperimentResult(exp_data, meta_data)


@experiment
def syntetic_full_fw(n_train=100, C=0.1, dataset=1,
                     max_iter=1000, n_inference_iter=5,
                     check_dual_every=10,
                     inference_method='gco'):
    # save parameters as meta
    meta_data = locals()

    crf = HCRF(n_states=10, n_features=10, n_edge_features=2, alpha=1,
               inference_method=inference_method, n_iter=n_inference_iter)
    clf = FrankWolfeSSVM(crf, verbose=2, n_jobs=1, check_dual_every=check_dual_every,
                         max_iter=max_iter, C=C)

    X, Y = load_syntetic(dataset)
    x_train, y_train, y_train_full, x_test, y_test = \
        split_test_train(X, Y, n_train, n_train)

    start = time()
    clf.fit(x_train, y_train)
    stop = time()

    train_score = clf.score(x_train, y_train_full)
    test_score = clf.score(x_test, y_test)
    time_elapsed = stop - start

    print '============================================================'
    print 'Score on train set: %f' % train_score
    print 'Score on test set: %f' % test_score
    print 'Elapsed time: %f s' % time_elapsed

    exp_data = {}

    exp_data['timestamps'] = clf.timestamps_
    exp_data['primal_objective'] = clf.primal_objective_curve_
    exp_data['objective'] = clf.objective_curve_

    meta_data['dataset_name'] = 'syntetic'
    meta_data['annotation_type'] = 'full'
    meta_data['label_type'] = 'full'
    meta_data['train_score'] = train_score
    meta_data['test_score'] = test_score
    meta_data['time_elapsed'] = time_elapsed

    return ExperimentResult(exp_data, meta_data)

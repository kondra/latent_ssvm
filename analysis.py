import numpy as np
import pylab as pl

from pystruct.learners import OneSlackSSVM
from latent_structured_svm import LatentSSVM
from heterogenous_crf import HCRF

from test_weak_labeled import msrc_load
from test_weak_labeled import split_test_train
from data_loader import load_syntetic

def msrc_train_score_per_iter(result, only_weak=False, plot=True):
    w_history = result.data['w_history']
    meta_data = result.meta
    n_full = meta_data['n_full']
    n_train = meta_data['n_train']
    n_inference_iter = meta_data['n_inference_iter']
    n_full = meta_data['n_full']
    n_train = meta_data['n_train']
    C = meta_data['C']
    latent_iter = meta_data['latent_iter']
    max_iter = meta_data['max_iter']
    inner_tol = meta_data['inner_tol']
    outer_tol = meta_data['outer_tol']
    alpha = meta_data['alpha']
    min_changes = meta_data['min_changes']
    initialize = meta_data['initialize']

    crf = HCRF(n_states=24, n_features=2028, n_edge_features=4, alpha=alpha,
               inference_method='gco', n_iter=n_inference_iter)
    base_clf = OneSlackSSVM(crf, max_iter=max_iter, C=C, verbose=2,
                            tol=inner_tol, n_jobs=4, inference_cache=10)
    clf = LatentSSVM(base_clf, latent_iter=latent_iter, verbose=2,
                     tol=outer_tol, min_changes=min_changes, n_jobs=4)

    Xtrain, Ytrain, Ytrain_full, Xtest, Ytest = msrc_load(n_full, n_train)

    if only_weak:
        Xtrain = [x for (i, x) in enumerate(Xtrain) if not Ytrain[i].full_labeled]
        Ytrain_full = [y for (i, y) in enumerate(Ytrain_full) if not Ytrain[i].full_labeled]

    base_clf.w = None
    clf.w_history_ = w_history
    clf.iter_done = w_history.shape[0]

    train_scores = []
    for score in clf.staged_score(Xtrain, Ytrain_full):
        train_scores.append(score)
    train_scores = np.array(train_scores)

    if plot:
        x = np.arange(0, train_scores.size)
        pl.rc('text', usetex=True)
        pl.rc('font', family='serif')
        pl.figure(figsize=(10,10), dpi=96)
        pl.title('score on train set')
        pl.plot(x, train_scores)
        pl.scatter(x, train_scores)
        pl.xlabel('iteration')
        pl.xlim([-0.5, train_scores.size + 1])

    return train_scores

def syntetic_train_score_per_iter(result, only_weak=False, plot=True):
    w_history = result.data['w_history']
    meta_data = result.meta
    n_full = meta_data['n_full']
    n_train = meta_data['n_train']
    n_inference_iter = meta_data['n_inference_iter']
    n_full = meta_data['n_full']
    n_train = meta_data['n_train']
    dataset = meta_data['dataset']
    C = meta_data['C']
    latent_iter = meta_data['latent_iter']
    max_iter = meta_data['max_iter']
    inner_tol = meta_data['inner_tol']
    outer_tol = meta_data['outer_tol']
    alpha = meta_data['alpha']
    min_changes = meta_data['min_changes']
    initialize = meta_data['initialize']

    crf = HCRF(n_states=10, n_features=10, n_edge_features=2, alpha=alpha,
               inference_method='gco', n_iter=n_inference_iter)
    base_clf = OneSlackSSVM(crf, max_iter=max_iter, C=C, verbose=0,
                            tol=inner_tol, n_jobs=4, inference_cache=100)
    clf = LatentSSVM(base_clf, latent_iter=latent_iter, verbose=2,
                     tol=outer_tol, min_changes=min_changes, n_jobs=4)

    X, Y = load_syntetic(dataset)

    Xtrain, Ytrain, Ytrain_full, Xtest, Ytest = \
        split_test_train(X, Y, n_full, n_train)

    if only_weak:
        Xtrain = [x for (i, x) in enumerate(Xtrain) if not Ytrain[i].full_labeled]
        Ytrain_full = [y for (i, y) in enumerate(Ytrain_full) if not Ytrain[i].full_labeled]

    base_clf.w = None
    clf.w_history_ = w_history
    clf.iter_done = w_history.shape[0]

    train_scores = []
    for score in clf.staged_score(Xtrain, Ytrain_full):
        train_scores.append(score)
    train_scores = np.array(train_scores)

    if plot:
        x = np.arange(0, train_scores.size)
        pl.rc('text', usetex=True)
        pl.rc('font', family='serif')
        pl.figure(figsize=(10,10), dpi=96)
        pl.title('score on train set')
        pl.plot(x, train_scores)
        pl.scatter(x, train_scores)
        pl.xlabel('iteration')
        pl.xlim([-0.5, train_scores.size + 1])

    return train_scores

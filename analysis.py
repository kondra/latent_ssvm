import numpy as np

from one_slack_ssvm import OneSlackSSVM
from latent_structured_svm import LatentSSVM
from heterogenous_crf import HCRF

from test_weak_labeled import msrc_load
from test_weak_labeled import split_test_train
from data_loader import load_syntetic

from pymongo import MongoClient

def get_all_from_mongo(dataset):
    cl = MongoClient()
    cl = cl['lSSVM']['base']
    exps = []
    for meta in cl.find():
        exps.append(meta)
    return exps

def create_model(result):
    meta = result.meta

    alpha = meta['alpha']
    n_inference_iter = meta['n_inference_iter']
    max_iter = meta['max_iter']
    C = meta['C']
    inner_tol = meta['inner_tol']
    inactive_window = meta['inactive_window']
    inactive_threshold = meta['inactive_threshold']
    latent_iter = meta['latent_iter']
    outer_tol = meta['outer_tol']
    min_changes = meta['min_changes']

    try:
        inference_cache = meta['inference_cache']
    except:
        inference_cache = 0

    crf = None
    if meta['dataset_name'] == 'syntetic':
        crf = HCRF(n_states=10, n_features=10, n_edge_features=2, alpha=alpha,
                   inference_method='gco', n_iter=n_inference_iter)
    elif meta['dataset_name'] == 'msrc':
        crf = HCRF(n_states=24, n_features=2028, n_edge_features=4, alpha=alpha,
                   inference_method='gco', n_iter=n_inference_iter)

    base_clf = OneSlackSSVM(crf, max_iter=max_iter, C=C, verbose=0,
                            tol=inner_tol, n_jobs=4, inference_cache=inference_cache,
                            inactive_window=inactive_window,
                            inactive_threshold=inactive_threshold)
    clf = LatentSSVM(base_clf, latent_iter=latent_iter, verbose=2,
                     tol=outer_tol, min_changes=min_changes, n_jobs=4)

    return clf

def load_dataset(result):
    n_train = result.meta['n_train']
    n_full = result.meta['n_full']

    Xtrain = None
    Ytrain = None
    Ytrain_full = None
    Xtest = None
    Ytest = None

    if result.meta['dataset_name'] == 'syntetic':
        dataset = result.meta['dataset']
        X, Y = load_syntetic(dataset)
        Xtrain, Ytrain, Ytrain_full, Xtest, Ytest = \
            split_test_train(X, Y, n_full, n_train)
    elif result.meta['dataset_name'] == 'msrc':
        Xtrain, Ytrain, Ytrain_full, Xtest, Ytest = \
            msrc_load(n_full, n_train)

    return Xtrain, Ytrain, Ytrain_full, Xtest, Ytest

def compute_score_per_inner_iter(result, score_types=['train', 'test', 'raw']):
    clf = create_model(result)

    Xtrain, Ytrain, Ytrain_full, Xtest, Ytest = \
        load_dataset(result)

    train_scores = []
    test_scores = []
    raw_scores = []

    for i in xrange(result.data['inner_w'].shape[0]):
        print('%d of %d' % (i, result.data['inner_w'].shape[0]))
        clf.w = result.data['inner_w'][i,:]
        if 'train' in score_types:
            train_scores.append(clf.score(Xtrain, Ytrain_full))
        if 'test' in score_types:
            test_scores.append(clf.score(Xtest, Ytest))
        if 'raw' in score_types:
            raw_scores.append(clf.score(Xtrain, Ytrain))

    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    raw_scores = np.array(raw_scores)

    if 'train' in score_types:
        result.data['inner_train_scores'] = train_scores
    if 'test' in score_types:
        result.data['inner_test_scores'] = test_scores
    if 'raw' in score_types:
        result.data['inner_raw_scores'] = raw_scores

    result.update_data()
    
    return result

def compute_score_per_iter(result, score_types=['raw']):
    # score_types should be a list of strings, possible values are:
    # 'raw', 'train', 'test'

    clf = create_model(result)

    Xtrain, Ytrain, Ytrain_full, Xtest, Ytest = \
        load_dataset(result)

    clf.base_ssvm.w = None
    clf.w_history_ = result.data['w_history']
    clf.iter_done = w_history.shape[0]

    if 'train' in score_types:
        result.data['train_scores'] = np.array([s for s in clf.staged_score(Xtrain, Ytrain_full)])
    if 'test' in score_types:
        result.data['test_scores'] = np.array([s for s in clf.staged_score(Xtest, Ytest)])
    if 'raw' in score_types:
        result.data['raw_scores'] = np.array([s for s in clf.staged_score(Xtrain, Ytrain)])

    result.update_data()

    return result

import numpy as np
import pylab as pl
import cPickle

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

# testing with weakly labeled train set


def test_syntetic_weak(mode):
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


def syntetic_weak():
    #heterogenous model
    models_basedir = 'models/syntetic/'
    results_basedir = 'results/syntetic/'
    prefix = 'NEWareas_v3_'
    n_full = 10
    n_train = 100

    crf = HCRF(n_states=10, n_features=10, n_edge_features=2,
               inference_method='gco')
    base_clf = OneSlackSSVM(crf, max_iter=500, C=0.1, verbose=0,
                            tol=0.001, n_jobs=4, inference_cache=100)
    clf = LatentSSVM(base_clf, latent_iter=15, verbose=2, tol=0.01,
                     min_changes=10, n_jobs=4)

    X, Y = load_syntetic(1)

    x_train, y_train, y_train_full, x_test, y_test = split_test_train(X, Y,
                                                                      n_full,
                                                                      n_train)

    start = time()
    clf.fit(x_train, y_train, initialize=True)
    stop = time()

    np.savetxt(models_basedir + prefix + 'syntetic_weak.csv', clf.w)
    with open(models_basedir + prefix + 'syntetic_weak' + '.pickle', 'w') as f:
        cPickle.dump(clf, f)

    print 'Score on train set: %f' % clf.score(x_train, y_train_full)
    print 'Score on test set: %f' % clf.score(x_test, y_test)
    print 'Norm of weight vector: |w|=%f' % np.linalg.norm(clf.w)
    print 'Elapsed time: %f s' % (stop - start)

    test_error = []
    for score in clf.staged_score(x_test, y_test):
        test_error.append(score)

    np.savetxt(results_basedir + prefix + 'error_per_iter', np.array(test_error))
    np.savetxt(results_basedir + prefix + 'deltas_per_iter', clf.w_deltas)
    np.savetxt(results_basedir + prefix + 'changes_per_iter', clf.changes_count)


def msrc_weak():
    #heterogenous model
    basedir = '../data/msrc/trainmasks/'
    models_basedir = 'models/msrc/'
    results_basedir = 'results/msrc/'
    prefix = ''

    n_train = 20
    train_mask = np.genfromtxt(basedir + 'trainMaskX%d.txt' % n_train)
    train_mask = train_mask[0:276].astype(np.bool)

    crf = HCRF(n_states=24, n_features=2028, n_edge_features=4,
               inference_method='gco')
    base_clf = OneSlackSSVM(crf, max_iter=100, C=0.1, verbose=0,
                            tol=0.1, n_jobs=4, inference_cache=10)
    clf = LatentSSVM(base_clf, latent_iter=20, verbose=2, tol=0.01)

    Xtest, Htest = load_msrc('test')
    Xtrain, Htrain = load_msrc('train')

    Ytrain = weak_from_hidden(Htrain)
    Ytest = weak_from_hidden(Htest)

    is_full = train_mask
    for i in xrange(len(is_full)):
        if is_full[i] == False:
            # remove full-labels
            Htrain[i][:, 0] = 0

    start = time()
    clf.fit(Xtrain, Ytrain, Htrain, is_full, pass_labels=True, initialize=True)
    stop = time()

#    np.savetxt(models_basedir + prefix + 'msrc_weak.csv', clf.w)
#    with open(models_basedir + prefix + 'msrc_weak' + '.pickle', 'w') as f:
#        cPickle.dump(clf, f)

    print 'Score on test set: %f' % clf.score(Xtest, Htest)
    print 'Norm of weight vector: |w|=%f' % np.linalg.norm(clf.w)
    print 'Elapsed time: %f s' % (stop - start)

    test_error = []
    for score in clf.staged_score(Xtest, Htest):
        test_error.append(score)

    np.savetxt(results_basedir + prefix + 'error_per_iter', np.array(test_error))
    np.savetxt(results_basedir + prefix + 'deltas_per_iter', clf.w_deltas)
    np.savetxt(results_basedir + prefix + 'changes_per_iter', clf.changes_count)

    return clf


if __name__ == '__main__':
    syntetic_weak()
#    clf = msrc_weak()
#    test_syntetic_weak('heterogenous')

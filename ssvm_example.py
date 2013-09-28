import numpy as np

import cPickle

from sklearn.cross_validation import train_test_split
from edge_crf import EdgeCRF
from pystruct.learners import OneSlackSSVM

from time import time

from data_loader import load_syntetic
from data_loader import load_msrc
from common import compute_error

# testing with full labeled train sets


def syntetic():
    # train model on a single set
    models_basedir = 'models/syntetic/'
    crf = EdgeCRF(n_states=10, n_features=10, n_edge_features=2,
                  inference_method='gco')
    clf = OneSlackSSVM(crf, max_iter=10000, C=0.01, verbose=2,
                       tol=0.1, n_jobs=4, inference_cache=100)

    X, Y = load_syntetic(1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=100,
                                                        random_state=179)

    start = time()
    clf.fit(x_train, y_train)
    stop = time()

    np.savetxt(models_basedir + 'syntetic_full.csv', clf.w)
    with open(models_basedir + 'syntetic_full' + '.pickle', 'w') as f:
        cPickle.dump(clf, f)

    y_pred = clf.predict(x_test)

    print 'Error on test set: %f' % compute_error(y_test, y_pred)
    print 'Score on test set: %f' % clf.score(x_test, y_test)
    print 'Score on train set: %f' % clf.score(x_train, y_train)
    print 'Norm of weight vector: |w|=%f' % np.linalg.norm(clf.w)
    print 'Elapsed time: %f s' % (stop - start)

    return clf


def syntetic_test():
    # test model on different train set size & on different train sets
    results = np.zeros((18, 5))
    full_labeled = np.array([2, 4, 10, 25, 100])
    train_size = 400

    for dataset in xrange(1, 19):
        X, Y = load_syntetic(dataset)

        for j, nfull in enumerate(full_labeled):
            crf = EdgeCRF(n_states=10, n_features=10, n_edge_features=2,
                          inference_method='qpbo')
            clf = OneSlackSSVM(crf, max_iter=10000, C=0.01, verbose=0,
                               tol=0.1, n_jobs=4, inference_cache=100)

            x_train = X[:nfull]
            y_train = Y[:nfull]
            x_test = X[(train_size + 1):]
            y_test = Y[(train_size + 1):]

            try:
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)

                results[dataset - 1, j] = compute_error(y_test, y_pred)

                print 'dataset=%d, nfull=%d, error=%f' % (dataset, nfull,
                                                          results[dataset - 1, j])
            except ValueError:
                print 'dataset=%d, nfull=%d: Failed' % (dataset, nfull)

    np.savetxt('results/syntetic/full_labeled.txt', results)

def remove_areas(Y):
    Y = [y[:, 0] for y in Y]
    return Y


def msrc():
    models_basedir = 'models/msrc/'
    crf = EdgeCRF(n_states=24, n_features=2028, n_edge_features=4,
                  inference_method='gco')
    clf = OneSlackSSVM(crf, max_iter=10000, C=0.01, verbose=2,
                       tol=0.1, n_jobs=4,
                       inference_cache=100)

    X, Y = load_msrc('train')
    Y = remove_areas(Y)

    start = time()
    clf.fit(X, Y)
    stop = time()

    np.savetxt(models_basedir + 'msrc_full.csv', clf.w)
    with open(models_basedir + 'msrc_full' + '.pickle', 'w') as f:
        cPickle.dump(clf, f)

    X, Y = load_msrc('test')
    Y = remove_areas(Y)

    Y_pred = clf.predict(X)

    print 'Error on test set: %f' % compute_error(Y, Y_pred)
    print 'Score on test set: %f' % clf.score(X, Y)
    print 'Norm of weight vector: |w|=%f' % np.linalg.norm(clf.w)
    print 'Elapsed time: %f s' % (stop - start)

    return clf


def msrc_test():
    # test model on different train set sizes
    basedir = '../data/msrc/trainmasks/'
    models_basedir = 'models/msrc/'
    quality = []

    Xtest, Ytest = load_msrc('test')
    Ytest = remove_areas(Ytest)
    Xtrain, Ytrain = load_msrc('train')
    Ytrain = remove_areas(Ytrain)

    for n_train in [20, 40, 80, 160, 276]:
        crf = EdgeCRF(n_states=24, n_features=2028, n_edge_features=4,
                      inference_method='gco')
        clf = OneSlackSSVM(crf, max_iter=1000, C=0.01, verbose=0,
                           tol=0.1, n_jobs=4, inference_cache=100)

        if n_train != 276:
            train_mask = np.genfromtxt(basedir + 'trainMaskX%d.txt' % n_train)
            train_mask = train_mask[:277].astype(np.bool)
        else:
            train_mask = np.ones(276).astype(np.bool)

        curX = []
        curY = []
        for (s, x, y) in zip(train_mask, Xtrain, Ytrain):
            if s:
                curX.append(x)
                curY.append(y)

        start = time()
        clf.fit(curX, curY)
        stop = time()

        np.savetxt(models_basedir + 'test_model_%d.csv' % n_train, clf.w)
        with open(models_basedir + 'test_model_%d' % n_train + '.pickle', 'w') as f:
            cPickle.dump(clf, f)

        Ypred = clf.predict(Xtest)

        q = 1 - compute_error(Ytest, Ypred)

        print 'n_train=%d, quality=%f, time=%f' % (n_train, q, stop - start)
        quality.append(q)

    np.savetxt('results/msrc/msrc_full.txt', quality)


if __name__ == '__main__':
    clf = syntetic()
#    clf = msrc()

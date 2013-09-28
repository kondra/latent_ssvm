import numpy as np

from sklearn.cross_validation import train_test_split
from edge_crf import EdgeCRF
from pystruct.learners import OneSlackSSVM

from time import time

from common import load_syntetic
from common import load_msrc
from common import compute_error


def syntetic():
    crf = EdgeCRF(n_states=10, n_features=10, n_edge_features=2,
                  inference_method='gco')
    clf = OneSlackSSVM(crf, max_iter=10000, C=0.01, verbose=2,
                       tol=0.1, show_loss_every=50, n_jobs=4,
                       inference_cache=100)

    X, Y = load_syntetic(1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=100)

    start = time()
    clf.fit(x_train, y_train)
    stop = time()

    y_pred = clf.predict(x_test)

    print 'Error on test set: %f' % compute_error(y_test, y_pred)
    print 'Score on test set: %f' % clf.score(x_test, y_test)
    print 'Score on train set: %f' % clf.score(x_train, y_train)
    print 'Norm of weight vector: |w|=%f' % np.linalg.norm(clf.w)
    print 'Elapsed time: %f s' % (stop - start)


def remove_areas(Y):
    Y = [y[:, 0] for y in Y]
    return Y


def msrc():
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

    X, Y = load_msrc('test')
    Y = remove_areas(Y)

    Y_pred = clf.predict(X)

    print 'Error on test set: %f' % compute_error(Y, Y_pred)
    print 'Score on test set: %f' % clf.score(X, Y)
    print 'Norm of weight vector: |w|=%f' % np.linalg.norm(clf.w)
    print 'Elapsed time: %f s' % (stop - start)


def msrc_test():
    basedir = '../data/msrc/trainmasks/'
    quality = []

    Xtest, Ytest = load_msrc('test')
    Ytest = remove_areas(Ytest)
    Xtrain, Ytrain = load_msrc('train')
    Ytrain = remove_areas(Ytrain)

    Ys = []

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

        clf.fit(curX, curY)

        Ypred = clf.predict(Xtest)
        Ys.append(Ypred)

        q = 1 - compute_error(Ytest, Ypred)

        print 'n_train=%d, quality=%f' % (n_train, q)
        quality.append(q)

    np.savetxt('results/msrc/msrc_full.txt', quality)

    return Ys


if __name__ == '__main__':
#    Ys = msrc_test()
    msrc()

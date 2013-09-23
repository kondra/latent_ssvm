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
                       tol=0.1, show_loss_every=5, n_jobs=4,
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
                       tol=0.1, show_loss_every=5, n_jobs=4,
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


if __name__ == '__main__':
    msrc()

import numpy as np

from sklearn.cross_validation import train_test_split
from edge_crf import EdgeCRF
from pystruct.learners import OneSlackSSVM

from time import time

base = '../data/'


def compute_error(Y, Y_pred):
    err = 0.0
    N = len(Y)
    for i in xrange(N):
        err += np.sum(Y[i] != Y_pred[i])

    err /= N
    err /= 400

    return err


def load_data():
    unary_filename = base + 'unary10_e1.txt'
    pairwise_filename = base + 'pairwise10_e1.txt'
    label_filename = base + 'labels10_e1.txt'

    unary = np.genfromtxt(unary_filename)
    pairwise = np.genfromtxt(pairwise_filename)
    label = np.genfromtxt(label_filename)

    X_structured = []
    Y = []
    edges = pairwise[0:760, 1:3].astype(np.int32)
    edges = edges - 1

    for i in xrange(800):
        node_features = unary[(i * 400):((i + 1) * 400), 2:]
        edge_features = pairwise[(i * 760):((i + 1) * 760), 3:]
        X_structured.append((node_features, edges, edge_features))
        y = label[(i * 400):((i + 1) * 400), 2].astype(np.int32)
        Y.append(y - 1)

    return X_structured, Y


if __name__ == '__main__':
    crf = EdgeCRF(n_states=10, n_features=10, n_edge_features=2,
                  inference_method='qpbo',
                  class_weight=np.repeat(0.1, 10))
    clf = OneSlackSSVM(crf, max_iter=10000, C=0.01, verbose=2,
                       tol=0.1, show_loss_every=5, n_jobs=4,
                       inference_cache=100)

    X, Y = load_data()

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

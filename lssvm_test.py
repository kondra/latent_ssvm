import numpy as np

from latent_crf import LatentCRF
from edge_crf import EdgeCRF
from latent_structured_svm import LatentSSVM

from pystruct.learners import OneSlackSSVM
from time import time

from common import load_data
from common import compute_error


def latent_model():
    crf = LatentCRF(n_states=10, n_features=10, n_edge_features=2,
                    inference_method='qpbo')
    base_clf = OneSlackSSVM(crf, max_iter=100, C=0.01, verbose=0,
                            tol=0.1, n_jobs=4, inference_cache=100)
    clf = LatentSSVM(base_clf, latent_iter=5)

    return clf


def ssvm_model():
    crf = EdgeCRF(n_states=10, n_features=10, n_edge_features=2,
                  inference_method='qpbo')
    clf = OneSlackSSVM(crf, max_iter=10000, C=0.01, verbose=0,
                       tol=0.1, show_loss_every=5, n_jobs=4,
                       inference_cache=100)

    return clf


def weak_from_hidden(H):
    Y = []
    for h in H:
        Y.append(np.unique(h))
    return Y


def test_latent(full_labeled, train_size):
    results = np.zeros((18, 7))
    timestamps = np.zeros((18, 7))

    for dataset in xrange(1, 19):
        X, H = load_data(dataset)
        Y = weak_from_hidden(H)

        for j, nfull in enumerate(full_labeled):
            clf = ssvm_model()

            x_train = X[:train_size]
            y_train = Y[:train_size]
            h_train = H[:train_size]
            x_test = X[(train_size + 1):]
            h_test = H[(train_size + 1):]

            for i in xrange(nfull, len(h_train)):
                h_train[i] = None

            try:
                start = time()
                clf.fit(x_train, y_train, h_train)
                stop = time()
                h_pred = clf.predict_latent(x_test)

                results[dataset - 1, j] = compute_error(h_test, h_pred)
                timestamps[dataset - 1, j] = stop - start

                print 'dataset=%d, nfull=%d, \
                       error=%f, time=%f' % (dataset, nfull,
                                             results[dataset - 1, j],
                                             timestamps[dataset - 1, j])
            except ValueError:
                print 'dataset=%d, nfull=%d: Failed' % (dataset, nfull)

    return results, timestamps


def test_ssvm(full_labeled, train_size):
    results = np.zeros((18, 7))
    timestamps = np.zeros((18, 7))

    for dataset in xrange(1, 19):
        X, Y = load_data(dataset)

        for j, nfull in enumerate(full_labeled):
            clf = ssvm_model()

            x_train = X[:nfull]
            y_train = Y[:nfull]
            x_test = X[(train_size + 1):]
            y_test = Y[(train_size + 1):]

            try:
                start = time()
                clf.fit(x_train, y_train)
                stop = time()
                y_pred = clf.predict(x_test)

                results[dataset - 1, j] = compute_error(y_test, y_pred)
                timestamps[dataset - 1, j] = stop - start

                print 'dataset=%d, nfull=%d, \
                       error=%f, time=%f' % (dataset, nfull,
                                             results[dataset - 1, j],
                                             timestamps[dataset - 1, j])
            except ValueError:
                print 'dataset=%d, nfull=%d: Failed' % (dataset, nfull)

    return results, timestamps


if __name__ == '__main__':
    full_labeled = np.array([0, 2, 4, 10, 25, 100, 400])
    train_size = 400

    results, timestamps = test_ssvm(full_labeled, train_size)

    np.savetxt('ssvm_quality.csv', results, delimiter=',')
    np.savetxt('ssvm_timestamps.csv', timestamps, delimiter=',')

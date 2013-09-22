import numpy as np

from latent_crf import LatentCRF
from edge_crf import EdgeCRF
from latent_structured_svm import LatentSSVM
from heterogenous_crf import HCRF

from pystruct.learners import OneSlackSSVM

from common import load_data
from common import compute_error
from common import weak_from_hidden


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


def heterogenous_model():
    crf = HCRF(n_states=10, n_features=10, n_edge_features=2,
               inference_method='gco')
    base_clf = OneSlackSSVM(crf, max_iter=500, C=0.1, verbose=0,
                            tol=0.001, n_jobs=4, inference_cache=100)
    clf = LatentSSVM(base_clf, latent_iter=5, verbose=0)

    return clf


def test_weak_labeled(model_func):
    results = np.zeros((18, 6))
    full_labeled = np.array([0, 2, 4, 10, 25, 100])
    train_size = 400

    for dataset in xrange(1, 19):
        X, H = load_data(dataset)
        Y = weak_from_hidden(H)

        for j, nfull in enumerate(full_labeled):
            clf = model_func()

            x_train = X[:train_size]
            y_train = Y[:train_size]
            h_train = H[:train_size]
            x_test = X[(train_size + 1):]
            h_test = H[(train_size + 1):]

            for i in xrange(nfull, len(h_train)):
                h_train[i] = None

            try:
#                clf.fit(x_train, y_train, h_train)
                clf.fit(x_train, y_train, h_train,
                        pass_labels=True, initialize=True)
                h_pred = clf.predict_latent(x_test)

                results[dataset - 1, j] = compute_error(h_test, h_pred)

                print 'dataset=%d, nfull=%d, error=%f' % (dataset,
                                                          nfull,
                                                          results[dataset - 1, j])
            except ValueError:
                print 'dataset=%d, nfull=%d: Failed' % (dataset, nfull)

    return results


def test_full_labeled():
    results = np.zeros((18, 5))
    full_labeled = np.array([2, 4, 10, 25, 100])
    train_size = 400

    for dataset in xrange(1, 19):
        X, Y = load_data(dataset)

        for j, nfull in enumerate(full_labeled):
            clf = ssvm_model()

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

    return results


if __name__ == '__main__':
    results = test_weak_labeled(heterogenous_model)
    np.savetxt('results/heterogenous.csv', results, delimiter=',')

#    results = test_full_labeled()
#    np.savetxt('results/full_labeled.csv', results, delimiter=',')

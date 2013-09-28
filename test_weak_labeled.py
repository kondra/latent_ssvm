import numpy as np
import pylab as np
import cPickle

from pystruct.learners import OneSlackSSVM
from time import time

from latent_crf import LatentCRF
from latent_structured_svm import LatentSSVM
from heterogenous_crf import HCRF

from data_loader import load_syntetic
from common import compute_error
from common import weak_from_hidden

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


def syntetic_weak():
    #heterogenous model
    models_basedir = 'models/syntetic/'
    results_basedir = 'results/syntetic/'
    crf = HCRF(n_states=10, n_features=10, n_edge_features=2,
               inference_method='gco')
    base_clf = OneSlackSSVM(crf, max_iter=500, C=0.1, verbose=0,
                            tol=0.001, n_jobs=1, inference_cache=100)
    clf = LatentSSVM(base_clf, latent_iter=50, verbose=2, tol=0.01)

    X, H = load_syntetic(1)
    X = list(X)
    H = list(H)
    Y = weak_from_hidden(H)

    x_train = X[:400]
    y_train = Y[:400]
    h_train = H[:400]
    x_test = X[401:]
    y_test = Y[401:]
    h_test = H[401:]

    for i in xrange(10, len(h_train)):
        h_train[i] = None

    start = time()
    clf.fit(x_train, y_train, h_train, pass_labels=True, initialize=True)
    stop = time()

    np.savetxt(models_basedir + 'high_c_on_first_syntetic_weak.csv', clf.w)
    with open(models_basedir + 'high_c_on_first_syntetic_weak' + '.pickle', 'w') as f:
        cPickle.dump(clf, f)

    h_pred = clf.predict_latent(x_test)

    print 'Error on test set: %f' % compute_error(h_test, h_pred)
    print 'Norm of weight vector: |w|=%f' % np.linalg.norm(clf.w)
    print 'Elapsed time: %f s' % (stop - start)

    test_error = []
    for h_pred in clf.staged_predict_latent(x_test):
        test_error.append(compute_error(h_test, h_pred))

    np.savetxt(results_basedir + 'high_c_on_first_error_per_iter', np.array(test_error))
    np.savetxt(results_basedir + 'high_c_on_first_deltas_per_iter', clf.w_deltas)
    np.savetxt(results_basedir + 'high_c_on_first_changes_per_iter', clf.changes_count)


if __name__ == '__main__':
    syntetic_weak()
#    test_syntetic_weak('heterogenous')

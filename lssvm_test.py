import numpy as np

from latent_crf import LatentCRF
from latent_structured_svm import LatentSSVM
from pystruct.learners import OneSlackSSVM
from time import time

from common import load_data
from common import compute_error


if __name__ == '__main__':
    results = np.zeros((18, 7))
    timestamps = np.zeros((18, 7))

    for dataset in xrange(1, 18):
        for j, nfull in enumerate([0, 2, 4, 10, 25, 100, 400]):
            crf = LatentCRF(n_states=10, n_features=10, n_edge_features=2,
                            inference_method='qpbo')
            base_clf = OneSlackSSVM(crf, max_iter=100, C=0.01, verbose=0,
                                    tol=0.1, n_jobs=4, inference_cache=100)
            clf = LatentSSVM(base_clf, latent_iter=5)

            X, H = load_data(dataset)

            Y = []
            for h in H:
                Y.append(np.unique(h))

            x_train = X[:400]
            y_train = Y[:400]
            h_train = H[:400]
            x_test = X[401:]
            y_test = Y[401:]
            h_test = H[401:]

            for i in xrange(nfull, len(h_train)):
                h_train[i] = None

            start = time()
            clf.fit(x_train, y_train, h_train)
            stop = time()

            h_pred = clf.predict_latent(x_test)

            results[dataset - 1, j] = compute_error(h_test, h_pred)
            timestamps[dataset - 1, j] = stop - start

            print 'Dataset: %d, nfull=%d, error=%f, time=%f' % (dataset, nfull, results[dataset - 1, j], timestamps[dataset - 1, j])

    np.savetxt('quality.csv', results, delimiter=',')
    np.savetxt('timestamps.csv', timestamps, delimiter=',')

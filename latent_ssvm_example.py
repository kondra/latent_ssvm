import numpy as np

#from latent_crf import LatentCRF
from heterogenous_crf import HCRF
from latent_structured_svm import LatentSSVM
from pystruct.learners import OneSlackSSVM
from time import time

from common import load_data
from common import compute_error


if __name__ == '__main__':
    crf = HCRF(n_states=10, n_features=10, n_edge_features=2,
               inference_method='gco')
    base_clf = OneSlackSSVM(crf, max_iter=500, C=0.1, verbose=2,
                            tol=0.001, n_jobs=4, inference_cache=100)
    clf = LatentSSVM(base_clf, latent_iter=5, verbose=2)

    X, H = load_data(1)

    Y = []
    for h in H:
        Y.append(np.unique(h))

    x_train = X[:400]
    y_train = Y[:400]
    h_train = H[:400]
    x_test = X[401:]
    y_test = Y[401:]
    h_test = H[401:]

    for i in xrange(2, len(h_train)):
        h_train[i] = None

    start = time()
    clf.fit(x_train, y_train, h_train, pass_labels=True, initialize=True)
    stop = time()

    h_pred = clf.predict_latent(x_test)

    print 'Error on test set: %f' % compute_error(h_test, h_pred)
    print 'Norm of weight vector: |w|=%f' % np.linalg.norm(clf.w)
    print 'Elapsed time: %f s' % (stop - start)

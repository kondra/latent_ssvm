import numpy as np

from sklearn.cross_validation import train_test_split
from edge_crf import EdgeCRF
from pystruct.learners import OneSlackSSVM

from time import time

from common import load_data
from common import compute_error

if __name__ == '__main__':
    crf = EdgeCRF(n_states=10, n_features=10, n_edge_features=2,
                  inference_method='gco')
    clf = OneSlackSSVM(crf, max_iter=10000, C=0.01, verbose=2,
                       tol=0.1, show_loss_every=5, n_jobs=4,
                       inference_cache=100)

    X, Y = load_data(1)

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

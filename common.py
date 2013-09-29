import numpy as np

def compute_error(Y, Y_pred):
    err = 0.0
    N = len(Y)
    for i in xrange(N):
        err += np.sum(Y[i] != Y_pred[i]) / float(Y[i].size)

    err /= N

    return err


def weak_from_hidden(H):
    Y = []
    for h in H:
        Y.append(np.unique(h[:, 0].astype(np.int32)))
    return Y


import numpy as np

#a lot of hardcoded values here

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


import numpy as np

from data_loader import load_syntetic as load_syntetic_dataset
from data_loader import load_msrc_hdf, load_msrc_weak_train_mask
from label import Label


def load_binary_syntetic(dataset, n_train):
    # splitting syntetic dataset
    X, Y = load_syntetic_dataset(dataset)

    Xnew = []
    Ynew = []
    for x, y in zip(X, Y):
        Xnew.append((x[0], x[1], np.ones((x[2].shape[0], 1))))
        y_ = y[:, 0].astype(np.int32)
        labels = np.unique(y_)
        y[y_ == labels[0], 0] = 0
        for l in labels[1:]:
            if l == 0:
                continue
            y[y_ == l, 0] = 1
        Ynew.append(y)

    X = Xnew
    Y = Ynew

    x_train = X[:n_train]
    y_train = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
               for y in Y[:n_train]]

    x_test = X[n_train:]
    y_test = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
              for y in Y[n_train:]]

    return x_train, y_train, x_test, y_test


def load_syntetic(dataset, n_full, n_train):
    # splitting syntetic dataset
    X, Y = load_syntetic_dataset(dataset)
    x_train = X[:n_train]
    y_train = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
               for y in Y[:n_full]]
    y_train += [Label(None, np.unique(y[:, 0].astype(np.int32)), y[:, 1], False)
                for y in Y[(n_full):(n_train)]]
    y_train_full = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
                    for y in Y[:n_train]]

    x_test = X[n_train:]
    y_test = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
              for y in Y[n_train:]]

    return x_train, y_train, y_train_full, x_test, y_test


def load_msrc(n_full, n_train):
    # loading & splitting MSRC dataset
    MSRC_DATA_PATH = '/home/dmitry/Documents/Thesis/data/msrc/msrc.hdf5'

    x_train, y_train_raw, x_test, y_test = load_msrc_hdf(MSRC_DATA_PATH)
    y_test = [Label(y[:, 0].astype(np.int32), None,
                   y[:, 1].astype(np.float64) / np.sum(y[:, 1]), True)
             for y in y_test]

    if n_train != n_full:
        train_mask = load_msrc_weak_train_mask(MSRC_DATA_PATH, n_full)[:n_train]
    else:
        train_mask = [True for i in xrange(len(y_train_raw))]
    y_train_full = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
                   for y in y_train_raw]
    y_train = []
    for y, f in zip(y_train_raw, train_mask):
        if f:
            y_train.append(Label(y[:, 0].astype(np.int32),
                                None, y[:, 1], True))
        else:
            y_train.append(Label(None, np.unique(y[:, 0].astype(np.int32)),
                                y[:, 1], False))

    return x_train, y_train, y_train_full, x_test, y_test



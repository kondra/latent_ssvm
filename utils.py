import numpy as np

from data_loader import load_syntetic as load_syntetic_dataset
from data_loader import load_msrc_hdf, load_msrc_weak_train_mask
from label import Label


def load_syntetic(dataset, n_full, n_train):
    # splitting syntetic dataset
    X, Y = load_syntetic_dataset(dataset)
    x_train = X[:n_train]
    y_train = [Label(y[:, 0].astype(np.int32), None, y[:, 1], True)
               for y in Y[:n_full]]
    y_train += [Label(None, np.unique(y[:, 0].astype(np.int32)),
                      y[:, 1], False) for y in Y[(n_full):(n_train)]]
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

    train_mask = load_msrc_weak_train_mask(MSRC_DATA_PATH, n_full)[:n_train]
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



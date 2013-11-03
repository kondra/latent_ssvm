import numpy as np
import scipy.sparse as sps

base = '../data/'


def load_msrc(dataset):
    # fast loader, you should use this
    if dataset == 'train':
        filename = base + 'msrc/train_data.npz'
        npz = np.load(filename)
        unary = npz['Xunary_train']
        unary = [sps.csr_matrix(x) for x in unary]
        return zip(unary,
                   npz['edges_train'],
                   npz['Xpair_train']), list(npz['Y_train'])
    if dataset == 'test':
        filename = base + 'msrc/test_data.npz'
        npz = np.load(filename)
        unary = npz['Xunary_test']
        unary = [sps.csr_matrix(x) for x in unary]
        return zip(unary,
                   npz['edges_test'],
                   npz['Xpair_test']), list(npz['Y_test'])


def load_syntetic(dataset):
    # fast loader
    filename = base + 'syntetic/features%d.npz' % dataset
    npz = np.load(filename)
    # add superpixel area, which is one because pixel == superpixel in this dataset
    Y = [np.vstack([y, np.ones(y.size)]).T for y in list(npz['Y'])]
    return list(npz['X']), Y

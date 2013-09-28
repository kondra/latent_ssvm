import numpy as np

base = '../data/'


def load_msrc(dataset):
    # fast loader, you should use this
    if dataset == 'train':
        filename = base + 'msrc/train_data.npz'
        npz = np.load(filename)
        return zip(npz['Xunary_train'],
                   npz['edges_train'],
                   npz['Xpair_train']), npz['Y_train']
    if dataset == 'test':
        filename = base + 'msrc/test_data.npz'
        npz = np.load(filename)
        return zip(npz['Xunary_test'],
                   npz['edges_test'],
                   npz['Xpair_test']), npz['Y_test']


def load_syntetic(dataset):
    # fast loader
    filename = base + 'syntetic/features%d.npz' % dataset
    npz = np.load(filename)
    return npz['X'], npz['Y']

import numpy as np
import scipy.sparse as sps
import h5py

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

def load_msrc_weak_train_mask(filename, n):
    f = h5py.File(filename, 'r', libver='latest')
    name = 'train_mask_%d' % n
    mask = np.empty(f[name].shape, dtype=np.int32)
    f[name].read_direct(mask)
    f.close()
    return mask

def load_msrc_hdf(filename):
    # hdf5 loader, the best
    f = h5py.File(filename, 'r', libver='latest')
    g_u = f['features']['unaries']
    data = np.empty(g_u['data'].shape, dtype=np.float32)
    indices = np.empty(g_u['indices'].shape, dtype=np.int32)
    indptr = np.empty(g_u['indptr'].shape, dtype=np.int32)
    g_u['data'].read_direct(data)
    g_u['indices'].read_direct(indices)
    g_u['indptr'].read_direct(indptr)
    width = g_u.attrs['width']
    height = g_u.attrs['height']
    raw_unaries = sps.csr_matrix((data, indices, indptr), shape=(height, width))
    del data
    del indices
    del indptr

    g_f = f['features']
    raw_labels = np.empty(g_f['labels'].shape, dtype=np.int32)
    g_f['labels'].read_direct(raw_labels)
    raw_areas = np.empty(g_f['areas'].shape, dtype=np.int32)
    g_f['areas'].read_direct(raw_areas)

    raw_labels_all = np.vstack((raw_labels, raw_areas)).T

    spnum = np.empty(g_f['spnum'].shape, dtype=np.int32)
    g_f['spnum'].read_direct(spnum)
    unaries = []
    labels = []
    pos = 0
    for n in spnum:
        unaries.append(raw_unaries[pos:pos+n, :])
        labels.append(raw_labels_all[pos:pos+n, :])
        pos += n
    del raw_unaries
    del raw_labels
    del raw_labels_all
    del raw_areas

    edges_num = np.empty(g_f['edges_num'].shape, dtype=np.int32)
    g_f['edges_num'].read_direct(edges_num)
    raw_edges = np.empty(g_f['edges'].shape, dtype=np.int32)
    g_f['edges'].read_direct(raw_edges)
    raw_pairwise = np.empty(g_f['pairwise'].shape, dtype=np.float32)
    g_f['pairwise'].read_direct(raw_pairwise)

    edges = []
    pairwise = []

    pos = 0
    for n in edges_num:
        edges.append(raw_edges[pos:pos+n, :])
        pairwise.append(raw_pairwise[pos:pos+n, :])
        pos += n
    del raw_edges
    del raw_pairwise

    X = zip(unaries, edges, pairwise)
    Y = labels

    Xtrain = X[:276]
    Ytrain = Y[:276]
    Xtest = X[276:]
    Ytest = Y[276:]

    f.close()

    return Xtrain, Ytrain, Xtest, Ytest

def save_msrc_hdf():
    npz_test = np.load(base + 'msrc/test_data.npz')
    npz_train = np.load(base + 'msrc/train_data.npz')
    unaries_test = list(npz_test['Xunary_test'])
    unaries_train = list(npz_train['Xunary_train'])
    unaries = unaries_train + unaries_test

    spnum = np.array([x.shape[0] for x in unaries], dtype=np.int32)
    unaries = np.vstack(unaries)
    unaries = sps.csr_matrix(unaries)

    edges_test = list(npz_test['edges_test'])
    edges_train = list(npz_train['edges_train'])
    edges = edges_train + edges_test

    edges_num = np.array([e.shape[0] for e in edges], dtype=np.int32)
    edges = np.vstack(edges)

    pairwise_test = list(npz_test['Xpair_test'])
    pairwise_train = list(npz_train['Xpair_train'])
    pairwise = pairwise_train + pairwise_test

    pairwise = np.vstack(pairwise).astype(np.float32)

    labels_test = list(npz_test['Y_test'])
    labels_train = list(npz_train['Y_train'])
    labels = labels_train + labels_test
    labels = np.vstack(labels)
    areas = labels[:, 1]
    labels = labels[:, 0]

    train_mask_20 = np.genfromtxt('../data/msrc/trainmasks/trainMaskX20.txt', dtype=np.int32)
    train_mask_40 = np.genfromtxt('../data/msrc/trainmasks/trainMaskX40.txt', dtype=np.int32)
    train_mask_80 = np.genfromtxt('../data/msrc/trainmasks/trainMaskX80.txt', dtype=np.int32)
    train_mask_160 = np.genfromtxt('../data/msrc/trainmasks/trainMaskX160.txt', dtype=np.int32)

    test_mask = np.genfromtxt('../data/msrc/features/testMask.txt', dtype=np.int32)
    train_mask = np.genfromtxt('../data/msrc/features/trainMask.txt', dtype=np.int32)
    valid_mask = np.genfromtxt('../data/msrc/features/validMask.txt', dtype=np.int32)

    mapping = []
    with open('../data/msrc/features/names.txt', 'r') as f:
        for line in f:
            mapping.append(line.split(' ')[-1].strip())
    mapping = np.array(mapping)

    f = h5py.File('msrc.hdf5', 'w', libver='latest')
    grp0 = f.create_group('features')

    grp = grp0.create_group('unaries')
    grp.create_dataset('data', data=unaries.data)
    grp.create_dataset('indices', data=unaries.indices)
    grp.create_dataset('indptr', data=unaries.indptr)
    grp.attrs['height'] = unaries.shape[0]
    grp.attrs['width'] = unaries.shape[1]

    grp0.create_dataset('spnum', data=spnum)
    grp0.create_dataset('edges_num', data=edges_num)
    grp0.create_dataset('edges', data=edges)
    grp0.create_dataset('pairwise', data=pairwise)
    grp0.create_dataset('labels', data=labels)
    grp0.create_dataset('areas', data=areas)

    f.create_dataset('train_mask_20', data=train_mask_20)
    f.create_dataset('train_mask_40', data=train_mask_40)
    f.create_dataset('train_mask_80', data=train_mask_80)
    f.create_dataset('train_mask_160', data=train_mask_160)
    f.create_dataset('train_mask', data=train_mask)
    f.create_dataset('valid_mask', data=valid_mask)
    f.create_dataset('test_mask', data=test_mask)
    f.create_dataset('mapping', data=mapping)

    f.close()


def load_syntetic(dataset):
    # fast loader
    filename = base + 'syntetic/features%d.npz' % dataset
    npz = np.load(filename)
    # add superpixel area, which is one because pixel == superpixel in this dataset
    Y = [np.vstack([y, np.ones(y.size)]).T for y in list(npz['Y'])]
    return list(npz['X']), Y

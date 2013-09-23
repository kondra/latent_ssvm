import numpy as np

#a lot of hardcoded values here

base = '../data/'


def load_syntetic_data(dataset):
    # load and transfrom from csv files
    unary_filename = base + 'syntetic/txt/unary10_e%d.txt' % dataset
    pairwise_filename = base + 'syntetic/txt/pairwise10_e%d.txt' % dataset
    label_filename = base + 'syntetic/txt/labels10_e%d.txt' % dataset

    unary = np.genfromtxt(unary_filename)
    pairwise = np.genfromtxt(pairwise_filename)
    label = np.genfromtxt(label_filename)

    X_structured = []
    Y = []
    edges = pairwise[0:760, 1:3].astype(np.int32) - 1

    for i in xrange(800):
        node_features = unary[(i * 400):((i + 1) * 400), 2:]
        edge_features = pairwise[(i * 760):((i + 1) * 760), 3:]
        X_structured.append((node_features, edges, edge_features))
        y = label[(i * 400):((i + 1) * 400), 2].astype(np.int32)
        Y.append(y - 1)

    return X_structured, Y


def split_msrc_data(begin, end):
    # genfromtxt eats too much memory, split data to six chunks
    msrc_base = base + 'msrc/features/'
    filename = msrc_base + 'unaryFixed.txt'

    fout = open(msrc_base + 'unary%d-%d' % (begin, end), 'w')
    fin = open(filename, 'r')
    for l in fin:
        pos = l.find(' ')
        n = int(l[0:pos])
        if n < begin:
            continue
        if n == end:
            break
        fout.write(l)

    fout.close()
    fin.close()


def transfrom_splitted():
    # depends on gc, it may overflow
    x = np.genfromtxt('unary1-100')
    np.savez('unary_1-100', unary=x)
    x = np.genfromtxt('unary200-300')
    np.savez('unary_200-300', unary=x)
    x = np.genfromtxt('unary300-400')
    np.savez('unary_300-400', unary=x)
    x = np.genfromtxt('unary400-500')
    np.savez('unary_400-500', unary=x)
    x = np.genfromtxt('unary500-600')
    np.savez('unary_500-600', unary=x)


def save_train_test_msrc():
    #numpy doesn't load too big file (more than 2GB)
    #1-276 - train
    #277-532 - test
    import os
    os.chdir('/mnt/rec/stuff/thesis/msrc_data/')

    x1 = np.load('unary_1-100.npz')['unary'].astype(np.float32)
    x2 = np.load('unary_100-200.npz')['unary'].astype(np.float32)
    x3 = np.load('unary_200-300.npz')['unary'].astype(np.float32)

    X = np.vstack([x1, x2, x3])

    train_mask = X[:, 0] <= 276

    X = X[train_mask, :]

    np.savez('unary_train', unary=X)

    x4 = np.load('unary_300-400.npz')['unary'].astype(np.float32)
    x5 = np.load('unary_400-500.npz')['unary'].astype(np.float32)
    x6 = np.load('unary_500-600.npz')['unary'].astype(np.float32)

    X = np.vstack([x3, x4, x5, x6])

    test_mask = (X[:, 0] <= 532) & (X[:, 0] > 276)

    X = X[test_mask, :]

    np.savez('unary_test', unary=X)


def load_msrc_data():
    # load and transfrom
    msrc_base = base + 'msrc/features/'
    msrc_base2 = '/mnt/rec/stuff/thesis/msrc_data/'
    unary_train_filename = msrc_base2 + 'unary_train.npz'
    unary_test_filename = msrc_base2 + 'unary_test.npz'
    pairwise_filename = msrc_base + 'pairwiseFixed.txt'
    labels_filename = msrc_base + 'labels.txt'

    unary_train = np.load(unary_train_filename)['unary']
    unary_test = np.load(unary_test_filename)['unary']
    pairwise = np.genfromtxt(pairwise_filename)
    labels = np.genfromtxt(labels_filename)
    labels = labels[:, 0:4]

    Xunary_test = []
    Xpair_test = []
    edges_test = []
    Y_test = []

    Xunary_train = []
    Xpair_train = []
    edges_train = []
    Y_train = []

    N_train = 276
    N_test = 532

    cpos = 0
    for i in xrange(1, N_train + 1):
        ppos = cpos
        while cpos < labels.shape[0] and labels[cpos, 0] == i:
            cpos += 1
        Y_train.append(labels[ppos:cpos, 2:4].astype(np.int32) - 1)
        Xunary_train.append(unary_train[ppos:cpos, 2:])

    pbegin = cpos
    for i in xrange(N_train + 1, N_test + 1):
        ppos = cpos
        while cpos < labels.shape[0] and labels[cpos, 0] == i:
            cpos += 1
        Y_test.append(labels[ppos:cpos, 2:4].astype(np.int32) - 1)
        Xunary_test.append(unary_test[(ppos-pbegin):(cpos-pbegin), 2:])

    cpos = 0
    for i in xrange(1, N_train + 1):
        ppos = cpos
        while cpos < pairwise.shape[0] and pairwise[cpos, 0] == i:
            cpos += 1
        edges_train.append(pairwise[ppos:cpos, 1:3].astype(np.int32) - 1)
        Xpair_train.append(pairwise[ppos:cpos, 3:])

    for i in xrange(N_train + 1, N_test + 1):
        ppos = cpos
        while cpos < pairwise.shape[0] and pairwise[cpos, 0] == i:
            cpos += 1
        edges_test.append(pairwise[ppos:cpos, 1:3].astype(np.int32) - 1)
        Xpair_test.append(pairwise[ppos:cpos, 3:])

    X_train = (Xunary_train, edges_train, Xpair_train)
    X_test = (Xunary_test, edges_test, Xpair_test)

    return X_train, Y_train, X_test, Y_test


def load_msrc(dataset):
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
    filename = base + 'syntetic/features%d.npz' % dataset
    npz = np.load(filename)
    return npz['X'], npz['Y']


def save_msrc_to_npz():
    X_train, Y_train, X_test, Y_test = load_msrc_data()
    filename_train = base + 'msrc/train_data'
    filename_test = base + 'msrc/test_data'
    np.savez(filename_train, Xunary_train=X_train[0],
             edges_train=X_train[1], Xpair_train=X_train[2],
             Y_train=Y_train)
    np.savez(filename_test, Xunary_test=X_test[0],
             edges_test=X_test[1], Xpair_test=X_test[2],
             Y_test=Y_test)


def save_syntetic_to_npz():
    for i in xrange(1, 21):
        X, Y = load_syntetic_data(i)
        filename = base + 'features%d' % i
        np.savez(filename, X=X, Y=Y)


def compute_error(Y, Y_pred):
    err = 0.0
    N = len(Y)
    for i in xrange(N):
        err += np.sum(Y[i] != Y_pred[i])

    err /= N
    err /= Y[0].size

    return err


def weak_from_hidden(H):
    Y = []
    for h in H:
        Y.append(np.unique(h))
    return Y

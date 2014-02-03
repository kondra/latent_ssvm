import numpy as np
import cPickle
from time import time

from pystruct.models.base import StructuredModel

from one_slack_ssvm import OneSlackSSVM
from latent_structured_svm import LatentSSVM

def generate_sample(w, max_iter=1000, temp=0.125, a=2):
    x = np.random.uniform(0,1,size=(5,3))
    t = np.random.randint(1,4,size=5)

    for it in xrange(max_iter):
        for i in xrange(5):
            v = np.zeros(3)
            for k in [1,2,3]:
                s = 0
                for j in xrange(5):
                    if j == i:
                        continue
                    for l in [1,2,3]:
                        s += w[k-1,l-1] * (t[j] == l)
                v[k-1] = 1.0/temp * (x[i,k-1] + s)
            v = v / np.sum(v)
            z = np.random.uniform(0,1)
            for k in [1,2,3]:
                if z < v[k-1]:
                    t[i] = k
                    break

        for i in xrange(5):
            for k in [1,2,3]:
                if t[i] != k:
                    x[i,k-1] = np.random.uniform(0,1)
                else:
                    x[i,k-1] = np.random.beta(a, 1)
        
    return x, t


def generate_dataset(n=500):
    X = []
    T = []

    w = np.random.standard_normal(size=(3,3))
    w = (w + w.T) / 2

    for i in xrange(n):
        x, t = generate_sample(w)
        X.append(x)
        T.append(t)

    return X, T


def compute_energy(x, wu, wp, t):
    E = 0
    for i in xrange(5):
        for k in [1,2,3]:
            E += (t[i] == k) * wu[k-1] * x[i,k-1]
    for i in xrange(5):
        for j in xrange(5):
            if i >= j:
                continue
            for k in [1,2,3]:
                for l in [1,2,3]:
                    E += wp[k-1,l-1] * (t[i]==k) * (t[j]==l)
    return E


def infer_labels(x, wu, wp, z=None, y=None):
    t_max = []
    E_max = -1000000000
    for t0 in [1,2,3]:
        for t1 in [1,2,3]:
            for t2 in [1,2,3]:
                for t3 in [1,2,3]:
                    for t4 in [1,2,3]:
                        t = [t0,t1,t2,t3,t4]
                        if z is not None:
                            if not np.all(np.bincount(t) == z):
                                continue
                        E = compute_energy(x,wu,wp,t)
                        if y is not None:
                            if y.full_labeled:
                                E += np.sum(t!=y.full)
                            else:
                                E += np.sum(np.abs(np.bincount(t) - y.weak))
                        if E > E_max:
                            t_max = t
                            E_max = E
    
    return t_max


class Label(object):
    def __init__(self, full, weak, full_labeled):
        self.full = full
        self.weak = weak
        self.full_labeled = full_labeled
        if self.full is None and self.weak is None:
            raise ValueError("You should specify full or weak labels")
        if self.weak is None:
            self.weak = np.bincount(self.full)
        if self.full is None:
            self.full = generate_st_weak(self.weak)
            assert(np.all(self.weak == np.bincount(self.full)))

    def __eq__(self, other):
        return np.all(self.full == other.full)


def generate_st_weak(z):
    y = []
    for i,k in enumerate(z):
        if k == 0:
            continue
        for j in xrange(k):
            y.append(i)
    return np.array(y, dtype=np.int64)


def load_simple_dataset(filename='simple_data.pkl'):
    f = open(filename, 'rb')
    X,Y = cPickle.load(f)
    f.close()

    x_train = []
    y_train = []
    y_train_full = []
    x_test = []
    y_test = []

    x_train = X[:400]
    y_train_full = [Label(y, None, True) for y in Y[:400]]
    x_test = X[400:]
    y_test = [Label(y, None, True) for y in Y[400:]]
    y_train = y_train_full[:100]
    y_train += [Label(None, np.bincount(y.full), False) for y in y_train_full[100:]]
#    y_train = [Label(None, np.bincount(y.full), False) for y in y_train_full]

    for y in y_train:
        assert(np.all(np.bincount(y.full) == y.weak))

    return x_train, y_train_full, y_train, x_test, y_test


class SimpleMRF(StructuredModel):
    def __init__(self):
        self.size_psi = 9+3
        self.inference_calls = 0

    def latent(self, x, y, w):   
#        if y.full_labeled:
#            return y
        wu = w[:3]
        wp = np.reshape(w[3:], (3,3))
        return Label(infer_labels(x, wu, wp, y.weak), y.weak, False) 

    def psi(self, x, y):
        t = y.full
        psi = np.zeros(12)
        for k in xrange(3):
            for i in xrange(5):
                psi[k] += (t[i] == k+1) * x[i,k]
        for k in xrange(3):
            for l in xrange(3):
                for i in xrange(5):
                    for j in xrange(5):
                        if i >= j:
                            continue
                        psi[3+3*k+l] += (t[i]==k+1) * (t[j]==l+1)

        return psi

    def loss(self, y, y_hat):
        return np.sum(y.full != y_hat.full)

    def loss_augmented_inference(self, x, y, w, relaxed=False, return_energy=False):
        wu = w[:3]
        wp = np.reshape(w[3:], (3,3))
        return Label(infer_labels(x, wu, wp, None, y), None, y.full_labeled)

    def inference(self, x, w, relaxed=False, return_energy=False):
        wu = w[:3]
        wp = np.reshape(w[3:], (3,3))
        return Label(infer_labels(x, wu, wp), None, True)


def test_simple_dataset(max_iter=1000, C=0.001, latent_iter=10, min_changes=0):
    crf = SimpleMRF()
    base_clf = OneSlackSSVM(crf, max_iter=max_iter, C=C, verbose=2,
                            n_jobs=4, tol=1e-3)
    clf = LatentSSVM(base_clf, latent_iter=latent_iter, verbose=2, min_changes=min_changes,
                     n_jobs=4, tol=1e-3)

    x_train, y_train_full, y_train, x_test, y_test = load_simple_dataset()

    start = time()
    clf.fit(x_train, y_train)
    stop = time()

    return clf

if __name__ == '__main__':
    clf = test_simple_dataset()

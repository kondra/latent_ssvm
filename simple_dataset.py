import numpy as np
import cPickle

from pystruct.models.base import StructuredModel

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
            if i == j:
                continue
            for k in [1,2,3]:
                for l in [1,2,3]:
                    E += wp[k-1,l-1] * (t[i]==k) * (t[j]==l)
    return E


def infer_labels(x, wu, wp, z=None):
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
                        if E > E_max:
                            t_max = t
                            E_max = E
    
    return t_max, E_max


class Label(object):
    def __init__(self, full, weak, full_labeled):
        self.full = full
        self.weak = weak
        self.full_labeled = full_labeled
        if self.full is None and self.weak is None:
            raise ValueError("You should specify full or weak labels")
        if self.weak is None:
            self.weak = np.bincount(y)
        if self.full is None:
            self.full = generate_st_weak(self.weak)

    def __eq__(self, other):
        return np.all(self.full == other.full)


def generate_st_weak(z):
    y = np.zeros(5)
    for i,k in enumerate(z):
        pos = np.random.randint(0, 5, k)
        if pos.size == 0:
            continue
        y[pos] = i
    return y


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
    y_train += [Label(None, np.bincount(y), False) for y in y_train_full[100:]]

    for y in y_train:
        assert(np.all(np.bincount(y.full) == y.weak))


class SimpleMRF(StructuredModel):
    def __init__(self):
        pass

    def latent(self, x, y, w):   
        if y.full_labeled:
            return y
        wu = w[:3]
        wp = np.reshape(w[3:], (3,3))
        return infer_labels(x, wu, wp, y.weak)

    def psi(self, x, y):
        pass

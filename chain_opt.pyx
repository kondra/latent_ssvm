# distutils: language = c++

import numpy as np
cimport numpy as np

cimport cython

from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.vector cimport vector


def optimize_chain_fast(
    np.ndarray[np.int64_t, ndim=1, mode='c'] chain,
    np.ndarray[np.float64_t, ndim=2, mode='c'] unary_cost,
    np.ndarray[np.float64_t, ndim=3, mode='c'] pairwise_cost,
    map[pair[int,int],int] edge_index):

    cdef int n_nodes = chain.shape[0]
    cdef int n_states = unary_cost.shape[1]

    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] p = np.zeros((n_states, n_nodes), dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=2, mode='c'] track = np.zeros((n_states, n_nodes), dtype=np.int32)

    p[:,0] = unary_cost[0,:]
    track[:,0] = -1

    cdef int i, k
    cdef int edge

    for i in range(1, n_nodes):
        p[:,i] = unary_cost[i,:]
        edge = edge_index[(chain[i - 1], chain[i])]
        p_cost = pairwise_cost[edge]
        for k in range(n_states):
            p[k,i] += np.min(p[:,i - 1] + p_cost[:,k])
            track[k,i] = np.argmin(p[:,i - 1] + p_cost[:,k])

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] x = np.zeros(n_nodes, dtype=np.int32)
    cdef int current = np.argmin(p[:,n_nodes - 1])
    for i in range(n_nodes - 1, -1, -1):
        x[i] = current
        current = track[current,i]

    return x, np.min(p[:,n_nodes - 1])

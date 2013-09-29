import numpy as np
import pylab as pl

import os


def plot_syntetic_full_weak():
    # full + weak plotter for syntetic data
    # on different full-labeled training set sizes
    os.chdir("results/syntetic")
    weak_results = np.genfromtxt('weak_labeled.csv', delimiter=',')
    full_results = np.genfromtxt('full_labeled.csv', delimiter=',')

    weak_results[0, 2] = 0.5
    full_results[0, 1] = 0.5

    y = np.mean(weak_results[:17, :6], axis=0)
    y_min = np.min(weak_results[:17, :6], axis=0)
    y_max = np.max(weak_results[:17, :6], axis=0)
    x = np.array([0, 2, 4, 10, 25, 100])

    full_y = np.mean(full_results[:17, :5], axis=0)
    full_y_min = np.min(full_results[:17, :5], axis=0)
    full_y_max = np.max(full_results[:17, :5], axis=0)

    pl.errorbar([0, 1, 2, 3, 4, 5], 1 - y, yerr=[y_max - y, y - y_min],
                label='+weak')
    pl.errorbar([1, 2, 3, 4, 5], 1 - full_y,
                yerr=[full_y_max - full_y, full_y - full_y_min], label='full')
    pl.xticks(np.arange(0, 6), x)
    pl.title('C=0.01')
    pl.xlabel('number of fully-labeled examples')
    pl.ylabel('hamming loss')
    pl.ylim([0, 1])
    pl.xlim([-0.1, 5.1])
    pl.legend(loc='lower right')


def plot_syntetic_heterogenous():
    # plot results from heterogenous lssvm trained on weak labeled data
    os.chdir("results/syntetic")
    results = np.genfromtxt('heterogenous')
    results = np.reshape(results, (-1, 6))

    y = np.mean(results, axis=0)
    y_min = np.min(results, axis=0)
    y_max = np.max(results, axis=0)
    x = np.array([0, 2, 4, 10, 25, 100])

    pl.errorbar([0, 1, 2, 3, 4, 5], 1 - y,
                yerr=[y_max - y, y - y_min], label='+weak')
    pl.xticks(np.arange(0, 6), x)
    pl.title('C=0.1')
    pl.xlabel('number of fully-labeled examples')
    pl.ylabel('hamming loss')
    pl.ylim([0, 1])
    pl.xlim([-0.1, 5.1])


def plot_syntetic_heterogenous_per_iter():
    os.chdir("results/syntetic")

    errors = np.genfromtxt('high_c_on_first_error_per_iter')
    deltas = np.genfromtxt('high_c_on_first_deltas_per_iter')
    changes= np.genfromtxt('high_c_on_first_changes_per_iter')
    x = np.arange(0, errors.size)

    pl.figure()
    pl.plot(x, errors)
    pl.scatter(x, errors)
    pl.title('Syntetic, C=0.1, full=10')
    pl.xlabel('latent SSVM iteration')
    pl.ylabel('error on test set')

    pl.figure()
    pl.plot(x, deltas)
    pl.scatter(x, deltas)
    pl.title('Syntetic, C=0.1')
    pl.xlabel('latent SSVM iteration')
    pl.ylabel('difference of ssvm weight vectors')

    pl.figure()
    pl.plot(x, changes)
    pl.scatter(x, changes)
    pl.title('Syntetic, C=0.1')
    pl.xlabel('latent SSVM iteration')
    pl.ylabel('changes in inferred latent labelling')


def plot_msrc_full():
    # plot results of ssvm trained on full labeled msrc data
    os.chdir("results/msrc")
    y = np.genfromtxt('msrc_full.txt')
    x = np.array([0, 20, 40, 80, 160, 276])

    pl.plot([1, 2, 3, 4, 5], y)
    pl.scatter([1, 2, 3, 4, 5], y, marker='o')
    pl.xticks(np.arange(0, 6), x)
    pl.title('C=0.01')
    pl.xlabel('number of fully-labeled examples')
    pl.ylabel('hamming loss')
    pl.ylim([0, 1])
    pl.xlim([-0.1, 5.1])

if __name__ == '__main__':
    plot_syntetic_heterogenous_per_iter()

    pl.show()

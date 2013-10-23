import numpy as np
import pylab as pl

import os

from results import ExperimentResult


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


def plot_heterogenous_per_iter(result):
    scores =  result.data['test_scores']
    deltas =  result.data['delta_history']
    changes = result.data['changes']
    objective_curve = result.data['objective_curve']
    primal_objective_curve = result.data['primal_objective_curve']
    x = np.arange(0, scores.size)

    pl.rc('text', usetex=True)
    pl.rc('font', family='serif')

    pl.figure(figsize=(10,10), dpi=96)

    pl.subplot(2, 2, 1)
    pl.title('score on test set')
    pl.plot(x, scores)
    pl.scatter(x, scores)
    pl.xlabel('iteration')
    pl.xlim([-0.5, scores.size + 1])

    pl.subplot(2, 2, 2)
    pl.title(r"\|w-w_{prev}\|_2")
    pl.plot(x, deltas)
    pl.scatter(x, deltas)
    pl.xlabel('iteration')
    pl.xlim([-0.5, scores.size + 1])

    pl.subplot(2, 2, 3)
    pl.title('changes in inferred latent labelling')
    pl.plot(x, changes)
    pl.scatter(x, changes)
    pl.xlabel('iteration')
    pl.xlim([-0.5, scores.size + 1])

    pl.subplot(2, 2, 4)
    pl.title('objective')
    pl.plot(x[1:], primal_objective_curve[1:-1], label='primal')
    pl.scatter(x[1:], primal_objective_curve[1:-1])
    pl.plot(x[1:], objective_curve[1:-1], c='r', label='cutting-plane')
    pl.scatter(x[1:], objective_curve[1:-1], c='r')
    pl.xlabel('iteration')
    pl.legend(loc='upper right')
    pl.xlim([-0.5, scores.size + 1])

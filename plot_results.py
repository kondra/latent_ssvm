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
    scores =  list(result.data['test_scores'])
    scores.append(result.meta['test_score'])
    scores = np.array(scores)
    train_scores =  list(result.data['train_scores'])
    train_scores.append(result.meta['train_score'])
    train_scores = np.array(train_scores)
    changes = result.data['changes']
    objective_curve = result.data['objective_curve']
    primal_objective_curve = result.data['primal_objective_curve']
    x = np.arange(0, scores.size)

    pl.rc('text', usetex=True)
    pl.rc('font', family='serif')

    pl.figure(figsize=(5, 5), dpi=96)
    pl.title('score')
    pl.plot(x, scores, label='test')
    pl.plot(x, train_scores, c='r', label='train')
    pl.scatter(x, scores)
    pl.scatter(x, train_scores, c='r')
    pl.xlabel('iteration')
    pl.xlim([-1, scores.size])
    pl.legend(loc='lower right')

    pl.figure(figsize=(5,5), dpi=96)
    pl.title('objective')
    pl.plot(x[1:], primal_objective_curve[1:], label='primal')
    pl.scatter(x[1:], primal_objective_curve[1:])
    pl.plot(x[1:], objective_curve[1:], c='r', label='cutting-plane')
    pl.scatter(x[1:], objective_curve[1:], c='r')
    pl.xlabel('iteration')
    pl.legend(loc='upper right')
    pl.xlim([-1, scores.size])

    pl.figure(figsize=(5,5), dpi=96)
    pl.title('changes in inferred latent labelling')
    pl.plot(x[1:], changes)
    pl.scatter(x[1:], changes)
    pl.xlabel('iteration')
    pl.xlim([-1, scores.size])

def plot_inner_scores(result):
    n_iter = result.data['w_history'].shape[0]
    sizes = result.data['inner_sz']
    test_scores = result.data['inner_test_scores']
    train_scores = result.data['inner_train_scores']
    acc = 0
    pl.rc('text', usetex=True)
    pl.rc('font', family='serif')
    pl.figure(figsize=(5, 5 * n_iter))
    for i in xrange(0, n_iter):
        pl.subplot(n_iter,1,i+1)
        pl.plot(test_scores[acc:acc+sizes[i]], label='test')
        pl.plot(train_scores[acc:acc+sizes[i]], c='r', label='train')
        pl.legend(loc='lower right')
        pl.ylabel('score')
        pl.xlabel('iteration')
        acc += sizes[i]

def plot_inner_objectives(result):
    n_iter = result.data['w_history'].shape[0]
    sizes = result.data['inner_sz']
    inner_primal = result.data['inner_primal']
    inner_objective = result.data['inner_objective']
    acc = 0
    pl.rc('text', usetex=True)
    pl.rc('font', family='serif')
    pl.figure(figsize=(5, 5 * n_iter))
    for i in xrange(0, n_iter):
        pl.subplot(n_iter,1,i+1)
        pl.plot(inner_primal[acc:acc+sizes[i]], label='primal')
        pl.plot(inner_objective[acc:acc+sizes[i]], c='r', label='cutting-plane')
        pl.legend(loc='upper right')
        pl.xlabel('iteration')
        acc += sizes[i]

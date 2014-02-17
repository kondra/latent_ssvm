import numpy as np
import pylab as pl

import os

import results

# old legacy stuff

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
    pl.title('Syntetic')
    pl.xlabel('number of fully-labeled objects')
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
    pl.ylabel('hamming loss')
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
#    pl.rc('text', usetex=True)
#    pl.rc('font', family='serif')
    pl.figure(figsize=(5, 5 * n_iter))
    for i in xrange(0, n_iter):
        pl.subplot(n_iter,1,i+1)
        pl.plot(test_scores[acc:acc+sizes[i]], label='test')
        pl.plot(train_scores[acc:acc+sizes[i]], c='r', label='train')
        pl.legend(loc='lower right')
        pl.ylabel('score')
        pl.xlabel('iteration')
        acc += sizes[i]

# good plotting utils

def plot_objectives(result, first_iter=1, save_dir=None):
    objective = result.data['objective_curve'][first_iter:]
    primal_objective = result.data['primal_objective_curve'][first_iter:]
    ind = np.arange(first_iter, objective.shape[0] + first_iter)

    pl.figure(figsize=(5,5), dpi=96)
    pl.plot(ind, primal_objective, label='primal')
    pl.plot(ind, objective, c='r', label='cutting-plane')
    pl.title('objective')
    pl.xlabel('iteration')
    pl.legend(loc='upper right')
    pl.xticks(ind, ind)

    if save_dir is not None:
        pl.savefig(save_dir + '/objectives.png')

def plot_changes(result, save_dir=None):
    changes = result.data['changes']
    ind = np.arange(changes.shape[0])
    pl.figure(figsize=(5,5), dpi=96)

    pl.plot(ind, changes)
    pl.xticks(ind, ind)
    pl.title('changes in inferred latent labelling')
    pl.xlabel('iteration')

    if save_dir is not None:
        pl.savefig(save_dir + '/changes.png')

def plot_scores(result, save_dir=None):
    test_scores =  result.data['test_scores']
    train_scores =  result.data['train_scores']
    ind = np.arange(test_scores.shape[0])

    pl.figure(figsize=(5, 5), dpi=96)
    pl.title('score')
    pl.plot(ind, test_scores, label='test')
    pl.plot(ind, train_scores, c='r', label='train')
    pl.ylabel('hamming loss')
    pl.xlabel('iteration')
    pl.xticks(ind, ind)
    pl.legend(loc='lower right')

    if save_dir is not None:
        pl.savefig(save_dir + '/scores.png')

def plot_inner_objectives(result, save_dir=None):
    n_iter = result.data['w_history'].shape[0]
    for i in xrange(0, n_iter):
        objectives = get_objective_per_iter(result, i)
        pl.figure(figsize=(5, 5))
        pl.plot(np.log(objectives['primal']), label='primal')
        pl.plot(np.log(objectives['cutting-plane']), c='r', label='cutting-plane')
        pl.legend(loc='upper right')
        pl.xlabel('iteration')
        pl.ylabel('log Objective')
        pl.title('SSVM objectives iteration %d' % i)

        if save_dir is not None:
            pl.savefig(save_dir + ('/inner_objective_%d.png' % i))

def plot_w_norm(result, first_iter=1, save_dir=None):
    w_norms = [0.5 * np.sum(w ** 2) for w in result.data['w_history'][first_iter:,:]]
    ind = np.arange(first_iter, len(w_norms) + first_iter)
    pl.figure(figsize=(5,5))
    pl.plot(ind, w_norms)
    pl.xticks(ind, ind)
    pl.xlabel('iteration')
    pl.title('0.5 |w|^2')

    if save_dir is not None:
        pl.savefig(save_dir + '/w_norm.png')

def plot_latent_objective(result, first_iter=1, norm=False, save_dir=None):
    objective = result.data['latent_objective'][first_iter:]
    ind = np.arange(first_iter, objective.shape[0] + first_iter)
    pl.figure(figsize=(5,5))
    pl.plot(ind, objective, label='objective')
    if norm:
        w_norms = [0.5 * np.sum(w ** 2) for w in result.data['w_history'][first_iter:,:]]
        pl.plot(ind, w_norms, c='r', label='w norm')
        pl.legend(loc='right')
    pl.xticks(ind, ind)
    pl.xlabel('iteration')
    pl.title('Latent SSVM objective')

    if save_dir is not None:
        if norm:
            pl.savefig(save_dir + '/latent_objective_with_norm.png')
        else:
            pl.savefig(save_dir + '/latent_objective.png')

def plot_raw_scores(result, first_iter=1, save_dir=None):
    pl.figure(figsize=(5,5))
    score = result.data['raw_scores'][first_iter:]
    ind = np.arange(first_iter, score.shape[0] + first_iter)
    pl.plot(ind, score)
    pl.xticks(ind, ind)
    pl.xlabel('iteration')
    pl.ylabel('kappa+delta')
    pl.title('Latent SSVM score (Kappa + Delta) on train set')

    if save_dir is not None:
        pl.savefig(save_dir + '/raw_scores.png')

def plot_all(result, save=False):
    save_dir = None
    if save:
        save_dir = os.path.join(results.working_directory, result.id, 'figures')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    plot_scores(result, save_dir=save_dir)
    plot_raw_scores(result, save_dir=save_dir)
    plot_latent_objective(result, save_dir=save_dir)
    plot_w_norm(result, save_dir=save_dir)
    plot_objectives(result, save_dir=save_dir)
    plot_changes(result, save_dir=save_dir)
    plot_inner_objectives(result, save_dir=save_dir)

# auxilary utils

def get_objective_per_iter(result, iteration):
    sizes = result.data['inner_sz'] + 1

    begin = np.sum(sizes[:iteration])
    end = begin + sizes[iteration]

    return {'cutting-plane' : result.data['inner_objective'][begin:end],
            'primal' : result.data['inner_primal'][begin:end]}

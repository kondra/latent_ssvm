from common import compute_error
from common import weak_from_hidden

def test_syntetic_weak(mode):
    # needs refactoring; does not work
    # Syntetic data
    # test latentSSVM on different train set sizes & on different train sets
    # mode can be 'heterogenous' or 'latent'
    results = np.zeros((18, 6))
    full_labeled = np.array([0, 2, 4, 10, 25, 100])
    train_size = 400

    for dataset in xrange(1, 19):
        X, H = load_syntetic(dataset)
        H = list(H)
        Y = weak_from_hidden(H)

        for j, nfull in enumerate(full_labeled):
            if mode == 'latent':
                crf = LatentCRF(n_states=10, n_features=10, n_edge_features=2,
                                inference_method='qpbo')
                base_clf = OneSlackSSVM(crf, max_iter=100, C=0.01, verbose=0,
                                        tol=0.1, n_jobs=4, inference_cache=100)
                clf = LatentSSVM(base_clf, latent_iter=5)
            elif mode == 'heterogenous':
                crf = HCRF(n_states=10, n_features=10, n_edge_features=2,
                           inference_method='gco')
                base_clf = OneSlackSSVM(crf, max_iter=500, C=0.1, verbose=0,
                                        tol=0.001, n_jobs=4, inference_cache=100)
                clf = LatentSSVM(base_clf, latent_iter=5, verbose=0)

            x_train = X[:train_size]
            y_train = Y[:train_size]
            h_train = H[:train_size]
            x_test = X[(train_size + 1):]
            h_test = H[(train_size + 1):]

            for i in xrange(nfull, len(h_train)):
                h_train[i] = None

            try:
                if mode == 'latent':
                    clf.fit(x_train, y_train, h_train)
                elif mode == 'heterogenous':
                    clf.fit(x_train, y_train, h_train,
                            pass_labels=True, initialize=True)
                h_pred = clf.predict_latent(x_test)

                results[dataset - 1, j] = compute_error(h_test, h_pred)

                print 'dataset=%d, nfull=%d, error=%f' % (dataset,
                                                          nfull,
                                                          results[dataset - 1, j])
            except ValueError:
                # bad QP
                print 'dataset=%d, nfull=%d: Failed' % (dataset, nfull)

    if mode == 'latent':
        np.savetxt('results/weak_labeled.csv', results, delimiter=',')
    elif mode == 'heterogenous':
        np.savetxt('results/heterogenous.csv', results, delimiter=',')

    return results

def msrc_train_score_per_iter(result, only_weak=False, plot=True):
    w_history = result.data['w_history']
    meta_data = result.meta
    n_full = meta_data['n_full']
    n_train = meta_data['n_train']
    n_inference_iter = meta_data['n_inference_iter']
    C = meta_data['C']
    latent_iter = meta_data['latent_iter']
    max_iter = meta_data['max_iter']
    inner_tol = meta_data['inner_tol']
    outer_tol = meta_data['outer_tol']
    alpha = meta_data['alpha']
    min_changes = meta_data['min_changes']
    initialize = meta_data['initialize']

    crf = HCRF(n_states=24, n_features=2028, n_edge_features=4, alpha=alpha,
               inference_method='gco', n_iter=n_inference_iter)
    base_clf = OneSlackSSVM(crf, max_iter=max_iter, C=C, verbose=2,
                            tol=inner_tol, n_jobs=4, inference_cache=10)
    clf = LatentSSVM(base_clf, latent_iter=latent_iter, verbose=2,
                     tol=outer_tol, min_changes=min_changes, n_jobs=4)

    Xtrain, Ytrain, Ytrain_full, Xtest, Ytest = msrc_load(n_full, n_train)

    if only_weak:
        Xtrain = [x for (i, x) in enumerate(Xtrain) if not Ytrain[i].full_labeled]
        Ytrain_full = [y for (i, y) in enumerate(Ytrain_full) if not Ytrain[i].full_labeled]

    base_clf.w = None
    clf.w_history_ = w_history
    clf.iter_done = w_history.shape[0]

    train_scores = []
    for score in clf.staged_score(Xtrain, Ytrain_full):
        train_scores.append(score)
    train_scores = np.array(train_scores)

    if plot:
        x = np.arange(0, train_scores.size)
        pl.rc('text', usetex=True)
        pl.rc('font', family='serif')
        pl.figure(figsize=(10,10), dpi=96)
        pl.title('score on train set')
        pl.plot(x, train_scores)
        pl.scatter(x, train_scores)
        pl.xlabel('iteration')
        pl.xlim([-0.5, train_scores.size + 1])

    return train_scores

def syntetic_train_score_per_iter(result, only_weak=False, plot=True):
    w_history = result.data['w_history']
    meta_data = result.meta
    n_full = meta_data['n_full']
    n_train = meta_data['n_train']
    n_inference_iter = meta_data['n_inference_iter']
    dataset = meta_data['dataset']
    C = meta_data['C']
    latent_iter = meta_data['latent_iter']
    max_iter = meta_data['max_iter']
    inner_tol = meta_data['inner_tol']
    outer_tol = meta_data['outer_tol']
    alpha = meta_data['alpha']
    min_changes = meta_data['min_changes']
    initialize = meta_data['initialize']

    crf = HCRF(n_states=10, n_features=10, n_edge_features=2, alpha=alpha,
               inference_method='gco', n_iter=n_inference_iter)
    base_clf = OneSlackSSVM(crf, max_iter=max_iter, C=C, verbose=0,
                            tol=inner_tol, n_jobs=4, inference_cache=100)
    clf = LatentSSVM(base_clf, latent_iter=latent_iter, verbose=2,
                     tol=outer_tol, min_changes=min_changes, n_jobs=4)

    X, Y = load_syntetic(dataset)

    Xtrain, Ytrain, Ytrain_full, Xtest, Ytest = \
        split_test_train(X, Y, n_full, n_train)

    if only_weak:
        Xtrain = [x for (i, x) in enumerate(Xtrain) if not Ytrain[i].full_labeled]
        Ytrain_full = [y for (i, y) in enumerate(Ytrain_full) if not Ytrain[i].full_labeled]

    base_clf.w = None
    clf.w_history_ = w_history
    clf.iter_done = w_history.shape[0]

    train_scores = []
    for score in clf.staged_score(Xtrain, Ytrain_full):
        train_scores.append(score)
    train_scores = np.array(train_scores)

    if plot:
        x = np.arange(0, train_scores.size)
        pl.rc('text', usetex=True)
        pl.rc('font', family='serif')
        pl.figure(figsize=(10,10), dpi=96)
        pl.title('score on train set')
        pl.plot(x, train_scores)
        pl.scatter(x, train_scores)
        pl.xlabel('iteration')
        pl.xlim([-0.5, train_scores.size + 1])

    return train_scores


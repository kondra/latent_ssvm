from test_weak_labeled import msrc_weak

result = msrc_weak('msrc lower regularization, alpha=1', max_iter=10000, latent_iter=3, inner_tol=0.1,
                   min_changes=20, outer_tol=0.1,
                   initialize=True, C=1000, alpha=1, n_inference_iter=5)

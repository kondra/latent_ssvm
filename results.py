# experiment result wrapper

from git import Repo
import uuid

repo = Repo('~/Documents/Thesis/latent_ssvm')

class ExperimentResult(object):
    def __init__(self, test_scores, changes, w_history, delta_history,
                 primal_objective_curve, objective_curve, timestamps, **kwargs):
        # generated data like scores per iteration, model parameters
        # stored in hdf5 file
        self.test_scores = test_scores
        self.changes = changes
        self.w_history = w_history
        self.delta_history = delta_history
        self.primal_objective_curve = primal_objective_curve
        self.objective_curve = objective_curve
        self.timestamps = timestamps

        # meta information, comments, parameters
        # this will be stored in mongodb
        self.meta = kwargs
        self.meta['commit_hash'] = repo.head.commit.hexsha
        self.meta['name'] = ''
        self.meta['comment'] = ''
        # unique experiment identifier
        self.meta['id'] = uuid.uuid1().hex

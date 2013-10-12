# experiment result wrapper

import uuid
import h5py

from git import Repo
from pymongo import MongoClient


def save(result, name, comment):
    result.name = name
    result.comment = comment
    result.save_meta()
    result.save_data()
    return result.id


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
        repo = Repo('~/Documents/Thesis/latent_ssvm')
        self.meta['commit_hash'] = repo.head.commit.hexsha
        self.meta['name'] = ''
        self.meta['comment'] = ''
        # unique experiment identifier
        self.meta['id'] = uuid.uuid1().hex

    def save_meta(self):
        client = MongoClient()
        client['lSSVM']['base'].insert(self.meta)
        client.disconnect()

    def save_data(self):
        f = h5py.File('/home/dmitry/Documents/Thesis/latent_ssvm/notebooks/experiment_data.hdf5', 'a')
        grp = f[self.meta['dataset_name']].create_group(self.meta['id'])
        grp.create_dataset("test_scores", data=self.test_scores)
        grp.create_dataset("changes", data=self.changes)
        grp.create_dataset("w_history", data=self.w_history)
        grp.create_dataset("delta_history", data=self.delta_history)
        grp.create_dataset("primal_objective_curve", data=self.primal_objective_curve)
        grp.create_dataset("objective_curve", data=self.objective_curve)
        grp.create_dataset("timestamps", data=self.timestamps)
        f.close()
        return grp.id.id

    @property
    def name(self):
        return self.meta['name']

    @name.setter
    def name(self, name_):
        self.meta['name'] = name_

    @property
    def comment(self):
        return self.meta['comment']

    @comment.setter
    def comment(self, comment_):
        self.meta['comment'] = comment_

    @property
    def id(self):
        return self.meta['id']

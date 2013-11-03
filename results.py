# experiment result wrapper

import numpy as np

import uuid
import h5py

from git import Repo
from pymongo import MongoClient


path_to_repo = '~/Documents/Thesis/latent_ssvm'
path_to_datafile = '/home/dmitry/Documents/Thesis/latent_ssvm/notebooks/experiment_data.hdf5'


class experiment(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, description, *args, **kwargs):
        result = None
        try:
            result = self.f(*args, **kwargs)
        except:
            raise
        result.save(description)
        return result


class ExperimentResult(object):
    def __init__(self, data, meta, is_new=True):
        # meta information, comments, parameters
        # this will be stored in mongodb
        self.meta = meta
        if is_new:
            repo = Repo(path_to_repo)
            self.meta['commit_hash'] = repo.head.commit.hexsha
            # unique experiment identifier
            self.meta['id'] = uuid.uuid1().hex
        # generated data like scores per iteration, model parameters
        # stored in hdf5 file
        self.data = data

    def save_data(self):
        f = h5py.File(path_to_datafile, 'a', libver='latest')
        grp = f[self.meta['dataset_name']].create_group(self.meta['id'])
        for k in self.data.keys():
            grp.create_dataset(k, data=self.data[k])
        f.close()
        return grp.id.id

    def save_meta(self):
        client = MongoClient()
        client['lSSVM']['base'].insert(self.meta)
        client.disconnect()

    def save(self, description=''):
        self.description = description
        self.save_meta()
        self.save_data()
        return self.id

    @staticmethod
    def load(exp_id):
        client = MongoClient()
        meta = client['lSSVM']['base'].find_one({'id' : exp_id})
        f = h5py.File(path_to_datafile, 'r', libver='latest')
        grp = f[meta[u'dataset_name']][exp_id]
        data = {}
        for k in grp.keys():
            data[k] = np.empty(grp[k].shape)
            grp[k].read_direct(data[k])
        f.close()
        return ExperimentResult(data, meta, is_new=False)

    @property
    def description(self):
        return self.meta['description']

    @description.setter
    def description(self, description_):
        self.meta['description'] = description_

    @property
    def id(self):
        return self.meta['id']

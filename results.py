# experiment result wrapper

import numpy as np

import uuid
import h5py
import sys
import os
import json
import tempfile
import logging
import logging.config

from git import Repo
from pymongo import MongoClient
from datetime import datetime
from time import time


path_to_repo = '/home/dmitry/Documents/Thesis/latent_ssvm'
working_directory = '/home/dmitry/ExtDocuments/Thesis/experiments'
tmp_directory = '/home/dmitry/ExtDocuments/Thesis/tmp'


class experiment(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, description, *args, **kwargs):
        result = None
        f = tempfile.NamedTemporaryFile(suffix='.log',
                                        dir=tmp_directory,
                                        delete=False)
        f.close()
        logging.config.dictConfig(
            {
                'version': 1,
                'formatters': {
                    'default': {
                        'format': '%(asctime)s %(levelname)s:%(name)s:%(message)s'
                    },
                },
                'handlers': {
                    'file': {
                        '()': logging.FileHandler,
                        'level': 'DEBUG',
                        'filename': f.name,
                        'mode': 'w',
                        'formatter': 'default',
                    },
                    'default': {
                        'level':'DEBUG',    
                        'class':'logging.StreamHandler',
                        'formatter': 'default',
                    }, 
                },
                'root': {
                    'handlers': ['file'],
                    'level': 'DEBUG',
                }
            }
        )
        try:
            result = self.f(*args, **kwargs)
        except:
            raise
        result.description = description
        result.save(f.name)
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
            self.meta['now'] = datetime.now().ctime()
            self.meta['timestamp'] = time()
        # generated data like scores per iteration, model parameters
        # stored in hdf5 file
        self.data = data

    def save_data(self):
        f = h5py.File(self.path_to_datafile, 'a', libver='latest')
        for k in self.data.keys():
            f.create_dataset(k, data=self.data[k])
        f.close()

    def update_data(self):
        f = h5py.File(self.path_to_datafile, 'a', libver='latest')
        for k in self.data.keys():
            if k not in f.keys():
                f.create_dataset(k, data=self.data[k])
        f.close()

    def save_meta(self):
        with open(self.path_to_metafile, 'w') as f:
            if '_id' in self.meta:
                del self.meta['_id']
            json.dump(self.meta, f, indent=0)
        try:
            with MongoClient('localhost', 27018) as client:
                client['lSSVM']['base'].insert(self.meta)
        except:
            print('could not connect to mongo; use only file backend')

    def save(self, logfile_name=None):
        self.exp_directory = os.path.join(working_directory, self.id)
        os.mkdir(self.exp_directory)
        if logfile_name is not None:
            os.rename(logfile_name, os.path.join(self.exp_directory, 'learning.log'))
        self.path_to_datafile = os.path.join(self.exp_directory, 'data.hdf5')
        self.path_to_metafile = os.path.join(self.exp_directory, 'meta.json')

        self.save_meta()
        self.save_data()
        return self.id

    @staticmethod
    def load(exp_id):
        path_to_datafile = os.path.join(working_directory, exp_id, 'data.hdf5')
        path_to_metafile = os.path.join(working_directory, exp_id, 'meta.json')
        if not os.path.exists(path_to_datafile):
            return self.load_old(exp_id)
        with open(path_to_metafile, 'r') as f:
            meta = json.load(f)
        with h5py.File(path_to_datafile, 'r', libver='latest') as f:
            data = {}
            for k in f.keys():
                data[k] = np.empty(f[k].shape)
                f[k].read_direct(data[k])
        return ExperimentResult(data, meta, is_new=False)

    @staticmethod
    def load_old(exp_id):
        client = MongoClient('localhost', 27018)
        meta = client['lSSVM']['base'].find_one({'id' : exp_id})

        path_to_datafile = '/home/dmitry/ExtDocuments/Thesis/experiment_data.hdf5'
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
        if 'description' in self.meta:
            return self.meta['description']
        else:
            return ''

    @description.setter
    def description(self, description_):
        self.meta['description'] = description_

    @property
    def id(self):
        return self.meta['id']


def resave_all():
    client = MongoClient('localhost', 27018)
    for r in client['lSSVM']['base'].find():
        exp_id = r['id']
        if exp_id == 'c15fdd7e336a11e3bd04002522e71f59':
            continue
        if exp_id == '43014e3037d011e3aeb1002522e71f59':
            continue
        if exp_id == '6e727b5c904511e3b6fa002522e71f59':
            continue
        path = os.path.join(working_directory, exp_id)
        if os.path.exists(path):
            continue
        print('processing experiment ' + exp_id)
        result = ExperimentResult.load_old(exp_id)
        result.save()

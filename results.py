# experiment result wrapper


class ExperimentResult(object):
    def __init__(self, name, test_scores, changes_count, w_deltas, ws, **kwargs):
        self.name = name
        self.test_scores = test_scores
        self.changes_count = changes_count
        self.w_deltas = w_deltas
        self.ws = ws
        self.args = kwargs

# experiment result wrapper


class ExperimentResult(object):
    def __init__(self, name, test_scores, changes, w_history, delta_history,
                 primal_objective_curve, objective_curve, timestamps, **kwargs):
        self.name = name
        self.test_scores = test_scores
        self.changes = changes
        self.w_history = w_history
        self.delta_history = delta_history
        self.primal_objective_curve = primal_objective_curve
        self.objective_curve = objective_curve
        self.args = kwargs

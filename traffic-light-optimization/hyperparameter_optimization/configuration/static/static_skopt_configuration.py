
import skopt

from hyperparameter_optimization.configuration.skopt_configuration import SkoptConfiguration

DURATION_MIN = 1
DURATION_MAX = 300


class StaticSkoptConfiguration(SkoptConfiguration):

    def _add_optimizable_phases_parameters(self, traffic_light_id, phase_name):
        return [
            skopt.space.Integer(
                DURATION_MIN, DURATION_MAX,
                name=traffic_light_id + '.' + phase_name + '.' + 'duration')
        ]

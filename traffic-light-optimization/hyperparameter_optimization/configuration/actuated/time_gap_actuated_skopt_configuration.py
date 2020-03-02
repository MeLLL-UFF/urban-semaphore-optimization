
import skopt

from hyperparameter_optimization.configuration.skopt_configuration import SkoptConfiguration

MIN_DUR_MIN = 1
MIN_DUR_MAX = 300
MAX_DUR_MIN = 1
MAX_DUR_MAX = 300
MAX_GAP_MIN = 0
MAX_GAP_MAX = 20
DETECTOR_GAP_MIN = 0
DETECTOR_GAP_MAX = 20

class TimeGapActuatedSkoptConfiguration(SkoptConfiguration):

    def _add_traffic_lights_parameters(self, traffic_light_id):
        return [
            skopt.space.Real(
                MAX_GAP_MIN, MAX_GAP_MAX,
                name=traffic_light_id + '.' + 'max-gap'),
            skopt.space.Real(
                DETECTOR_GAP_MIN, DETECTOR_GAP_MAX,
                name=traffic_light_id + '.' + 'detector-gap')
        ]

    def _add_optimizable_phases_parameters(self, traffic_light_id, phase_name):
        return [
            skopt.space.Integer(
                MIN_DUR_MIN, MIN_DUR_MAX,
                name=traffic_light_id + '.' + phase_name + '.' + 'minDur'),
            skopt.space.Integer(
                MAX_DUR_MIN, MAX_DUR_MAX,
                name=traffic_light_id + '.' + phase_name + '.' + 'maxDur')
        ]

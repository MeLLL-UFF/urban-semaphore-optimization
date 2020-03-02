
import skopt

from hyperparameter_optimization.configuration.skopt_configuration import SkoptConfiguration

MIN_DUR_MIN = 1
MIN_DUR_MAX = 300
MAX_DUR_MIN = 1
MAX_DUR_MAX = 300
MIN_TIME_LOSS_MIN = 0
MIN_TIME_LOSS_MAX = 20
DETECTOR_RANGE_MIN = 0
DETECTOR_RANGE_MAX = 1000

class TimeLossActuatedSkoptConfiguration(SkoptConfiguration):

    def _add_traffic_lights_parameters(self, traffic_light_id):
        return [
            skopt.space.Real(
                MIN_TIME_LOSS_MIN, MIN_TIME_LOSS_MAX,
                name=traffic_light_id + '.' + 'minTimeLoss'),
            skopt.space.Real(
                DETECTOR_RANGE_MIN, DETECTOR_RANGE_MAX,
                name=traffic_light_id + '.' + 'detectorRange')
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

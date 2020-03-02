
import skopt

from hyperparameter_optimization.configuration.skopt_configuration import SkoptConfiguration

GAMMA_MIN = 0
GAMMA_MAX = 1
EPSILON_MIN = 0
EPSILON_MAX = 1
LEARNING_RATE_MIN = 0
LEARNING_RATE_MAX = 1
BATCH_SIZE_OPTIONS = [16, 32, 64, 128]
MEMORY_PALACE_OPTIONS = [True, False]

class ReinforcementLearningSkoptConfiguration(SkoptConfiguration):

    def _add_traffic_lights_parameters(self, traffic_light_id):
        return [
            skopt.space.Real(
                GAMMA_MIN, GAMMA_MAX,
                name=traffic_light_id + '.' + 'gamma'),
            skopt.space.Real(
                EPSILON_MIN, EPSILON_MAX,
                name=traffic_light_id + '.' + 'epsilon'),
            skopt.space.Real(
                LEARNING_RATE_MIN, LEARNING_RATE_MAX,
                name=traffic_light_id + '.' + 'learning_rate'),
            skopt.space.Categorical(
                BATCH_SIZE_OPTIONS,
                name=traffic_light_id + '.' + 'batch_size'),
            skopt.space.Categorical(
                MEMORY_PALACE_OPTIONS,
                name=traffic_light_id + '.' + 'memory_palace')
        ]

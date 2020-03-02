
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

class SameHyperparameterReinforcementLearningSkoptConfiguration(SkoptConfiguration):

    def _add_general_parameters(self):
        return [
            skopt.space.Real(
                GAMMA_MIN, GAMMA_MAX,
                name='gamma'),
            skopt.space.Real(
                EPSILON_MIN, EPSILON_MAX,
                name='epsilon'),
            skopt.space.Real(
                LEARNING_RATE_MIN, LEARNING_RATE_MAX,
                name='learning_rate'),
            skopt.space.Categorical(
                BATCH_SIZE_OPTIONS,
                name='batch_size'),
            skopt.space.Categorical(
                MEMORY_PALACE_OPTIONS,
                name='memory_palace')
        ]

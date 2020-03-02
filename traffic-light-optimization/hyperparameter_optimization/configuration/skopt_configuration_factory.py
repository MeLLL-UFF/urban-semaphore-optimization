
import skopt

from hyperparameter_optimization.skopt_objective import time_loss_objective, depart_delay_objective
from hyperparameter_optimization.configuration.off.off_skopt_configuration import OffSkoptConfiguration
from hyperparameter_optimization.configuration.static.static_skopt_configuration import StaticSkoptConfiguration
from hyperparameter_optimization.configuration.actuated.time_gap_actuated_skopt_configuration \
    import TimeGapActuatedSkoptConfiguration
from hyperparameter_optimization.configuration.actuated.time_loss_actuated_skopt_configuration \
    import TimeLossActuatedSkoptConfiguration
from hyperparameter_optimization.configuration.sotl.sotl_skopt_configuration import SotlSkoptConfiguration
from hyperparameter_optimization.configuration.reinforcement_learning.\
    reinforcement_learning_skopt_configuration import ReinforcementLearningSkoptConfiguration
from experiment.strategy import OFF, STATIC, TIME_GAP_ACTUATED, TIME_LOSS_ACTUATED, SOTL_WAVE, SOTL_REQUEST, \
    SOTL_PLATOON, SOTL_PHASE, SOTL_MARCHING, REINFORCEMENT_LEARNING
from experiment.objective import TIME_LOSS, DEPART_DELAY


skopt_configuration_instances = {

    OFF: OffSkoptConfiguration,
    STATIC: StaticSkoptConfiguration,
    TIME_GAP_ACTUATED: TimeGapActuatedSkoptConfiguration,
    TIME_LOSS_ACTUATED: TimeLossActuatedSkoptConfiguration,
    SOTL_MARCHING: SotlSkoptConfiguration,
    SOTL_PHASE: SotlSkoptConfiguration,
    SOTL_PLATOON: SotlSkoptConfiguration,
    SOTL_REQUEST: SotlSkoptConfiguration,
    SOTL_WAVE: SotlSkoptConfiguration,
    REINFORCEMENT_LEARNING: ReinforcementLearningSkoptConfiguration
}

def get_skopt_configuration(scenario, strategy):
    return skopt_configuration_instances[strategy]().build_tls_space(scenario)

skopt_objective_configuration_instances = {

    TIME_LOSS: time_loss_objective,
    DEPART_DELAY: depart_delay_objective
}

def skopt_objective_function(scenario, strategy, objective):

    strategy_space = get_skopt_configuration(scenario, strategy)
    objective_function = skopt_objective_configuration_instances[objective]

    @skopt.utils.use_named_args(strategy_space)
    def custom_objective(**params):
        return objective_function(params)

    return custom_objective



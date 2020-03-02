
from city.simulation.simulation import Simulation
from city.simulation.reinforcement_learning_simulation import ReinforcementLearningSimulation

from experiment.strategy import  REINFORCEMENT_LEARNING

DEFAULT = 'DEFAULT'

simulation_instances = {

    DEFAULT: Simulation,
    REINFORCEMENT_LEARNING: ReinforcementLearningSimulation
}

def get_simulation(strategy):
    return simulation_instances.get(strategy, simulation_instances[DEFAULT])
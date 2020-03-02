
from city.traffic_light_system.traffic_light.traffic_light import TrafficLight
from city.traffic_light_system.traffic_light.off.off_traffic_light import OffTrafficLight
from city.traffic_light_system.traffic_light.reinforcement_learning.reinforcement_learning_traffic_light import ReinforcementLearningTrafficLight

from experiment.strategy import OFF, REINFORCEMENT_LEARNING

DEFAULT = 'DEFAULT'

traffic_light_configurer_instances = {

    DEFAULT: TrafficLight,
    OFF: OffTrafficLight,
    REINFORCEMENT_LEARNING: ReinforcementLearningTrafficLight
}

def get_traffic_light(strategy):
    return traffic_light_configurer_instances.get(strategy, traffic_light_configurer_instances[DEFAULT])
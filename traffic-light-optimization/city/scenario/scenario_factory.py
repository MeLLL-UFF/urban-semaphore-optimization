
from city.scenario.inga_small.inga_small_scenario import INGA_SMALL, IngaSmallScenario


scenario_instances = {
    INGA_SMALL: IngaSmallScenario
}

def get_scenario(scenario_name):
    return scenario_instances[scenario_name]()
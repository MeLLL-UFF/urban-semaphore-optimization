

class Experiment:

    def __init__(self, scenario, traffic_configuration, objective, strategy):

        self.scenario = scenario
        self.traffic_configuration = traffic_configuration
        self.objective = objective
        self.strategy = strategy

        scenario_directions = 'x'.join([str(value['cars_total_current'])
                                        for key, value in self.scenario.traffic_configuration.items()])

        self.path = self.scenario.name + '/' + self.objective + '/' + self.strategy + '/' + scenario_directions
        self.name = self.strategy + '__' + self.objective + '__' + self.scenario.name + '__' + scenario_directions

    def __repr__(self):
        return self.name
import copy


class TrafficLightConfigurer:

    def __init__(self):
        self.traffic_light_parameters = None
        self.network_definition = None

    def set_parameters(self, traffic_light_parameters):
        self.traffic_light_parameters = traffic_light_parameters

    def set_network_definition(self, network_definition):
        self.network_definition = copy.deepcopy(network_definition)

    def configure_traffic_light(self, tlLogic):
        pass

    def configure_optimizable_phase(self, tlLogic, phase):
        pass

    def configure_non_optimizable_phase(self, tlLogic, phase):
        pass

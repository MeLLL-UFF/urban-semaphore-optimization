
NUMBER = 'number'

class FlowConfigurer:

    def __init__(self):
        self.flow_scenario = None

    def set_scenario(self, scenario):
        self.scenario = scenario

    def configure_flow(self, flow):

        flow_id = flow.get('id')

        total_of_flow_vehicles = self.scenario.get_total_of_flow_vehicles(flow_id)

        flow.set(NUMBER, str(total_of_flow_vehicles))

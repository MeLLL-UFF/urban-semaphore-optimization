
from city.traffic_light_system.traffic_light.configurer.traffic_light_configurer import TrafficLightConfigurer

TYPE = 'type'
DURATION = 'duration'

class StaticTrafficLightConfigurer(TrafficLightConfigurer):

    def configure_traffic_light(self, tlLogic):
        tlLogic.set(TYPE, 'static')

    def configure_optimizable_phase(self, tlLogic, phase):

        key = tlLogic.get('id') + '.' + phase.get('name') + '.' + DURATION
        duration = str(self.traffic_light_parameters[key])

        phase.set(DURATION, duration)


instance = StaticTrafficLightConfigurer()
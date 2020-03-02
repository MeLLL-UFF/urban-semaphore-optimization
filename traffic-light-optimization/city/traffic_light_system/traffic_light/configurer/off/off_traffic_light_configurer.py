
from city.traffic_light_system.traffic_light.configurer.traffic_light_configurer import TrafficLightConfigurer

TYPE = 'type'

class OffTrafficLightConfigurer(TrafficLightConfigurer):

    def configure_traffic_light(self, tlLogic):
        pass
        # This way is currently not working
        #tlLogic.set(TYPE, 'off')

instance = OffTrafficLightConfigurer()
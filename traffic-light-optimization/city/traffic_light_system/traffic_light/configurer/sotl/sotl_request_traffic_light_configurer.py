
from city.traffic_light_system.traffic_light.configurer.sotl.base.sotl_traffic_light_configurer \
    import SotlTrafficLightConfigurer, TYPE

class SotlRequestTrafficLightConfigurer(SotlTrafficLightConfigurer):

    def configure_traffic_light(self, tlLogic):
        tlLogic.set(TYPE, 'sotl_request')

instance = SotlRequestTrafficLightConfigurer()

import traci

from city.traffic_light_system.traffic_light.traffic_light_factory import get_traffic_light


class TrafficLightSystem:

    def __init__(self, strategy, parameters=None, **kwargs):

        self.traffic_lights = {}

        traffic_light_ids = traci.trafficlight.getIDList()
        for id in traffic_light_ids:
            TrafficLight = get_traffic_light(strategy)
            self.traffic_lights[id] = TrafficLight(id, parameters, **kwargs)

    def step(self):

        for id, traffic_light in self.traffic_lights.items():
            traffic_light.step()

    def stop(self):

        for id, traffic_light in self.traffic_lights.items():
            traffic_light.stop()


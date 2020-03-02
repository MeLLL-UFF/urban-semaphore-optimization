
import traci

from city.traffic_light_system.traffic_light.traffic_light import TrafficLight

class OffTrafficLight(TrafficLight):

    def __init__(self, id, parameters, **kwargs):

        super().__init__(id, parameters)

        traci.trafficlight.setProgram(self.id, 'off')
import os
import sys

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from tqdm import tqdm
import traci

from city.traffic_light_system.traffic_light_system import TrafficLightSystem
from definitions import get_tripinfo_file_path, get_sumo_configuration_file_path
from utils.sumo_util import get_sumo_binary

class Simulation:

    def __init__(self, experiment, parameters, **kwargs):

        self._experiment = experiment
        self.execution = kwargs.get('execution', '')

        self._start_traci()

        self._traffic_light_system = TrafficLightSystem(experiment.strategy, parameters,
                                                        experiment=experiment, **kwargs)

    def run(self):

        cars_total = self._experiment.scenario.cars_total
        cars_arrived_progress_bar = tqdm(total=cars_total, desc="Number of arrived cars")

        cars_remaining = cars_total
        """execute the TraCI control loop"""
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            step_arrived_cars = traci.simulation.getArrivedNumber()
            cars_remaining -= step_arrived_cars

            if step_arrived_cars > 0:
                cars_arrived_progress_bar.update(step_arrived_cars)
            else:
                cars_arrived_progress_bar.refresh()

            self._traffic_light_system.step()

        cars_arrived_progress_bar.refresh()
        self._traffic_light_system.stop()

    def shutdown(self):
        traci.close()
        sys.stdout.flush()

    def _start_traci(self):

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([
            get_sumo_binary(),
            '-c', os.path.abspath(get_sumo_configuration_file_path(self._experiment.scenario)),
            '--tripinfo-output', get_tripinfo_file_path(self._experiment, self.execution),
            '--time-to-teleport', '900',
            '--collision.stoptime', '10',
            '--lanechange.duration', '1.0']
        )
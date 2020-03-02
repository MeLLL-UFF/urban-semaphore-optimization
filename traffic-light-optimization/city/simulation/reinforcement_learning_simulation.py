
from tqdm import tqdm

from city.simulation.simulation import Simulation


class ReinforcementLearningSimulation(Simulation):

    def __init__(self, experiment, parameters, **kwargs):

        self._experiment = experiment
        self._parameters = parameters
        self._kwargs = kwargs

        self.episodes = parameters.get('episodes', 25)

    def run(self):

        execution = self._kwargs.get('execution', '')

        for episode in tqdm(range(1, self.episodes + 1), desc='Reinforcement Learning episode'):

            self._kwargs['execution'] = str(execution) + '_' + str(episode)

            simulation = Simulation(self._experiment, self._parameters, **self._kwargs)
            simulation.run()
            simulation.shutdown()

    def shutdown(self):
        pass
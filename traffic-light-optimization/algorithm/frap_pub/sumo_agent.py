
from algorithm.frap_pub.agent import Agent


class SumoAgent(Agent):
    
    # Let SumoEnv take care of the traffic signalization by not overriding it
    def choose_action(self, step, state, *args, **kwargs):

        return 'no_op'

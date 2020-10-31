
from algorithm.frap.internal.frap_pub.transfer_dqn_agent import TransferDQNAgent
from algorithm.frap.internal.frap_pub.planning_only_agent import PlanningOnlyAgent


class TransferDQNWithPlanningAgent(TransferDQNAgent, PlanningOnlyAgent):
    
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, 
                 cnt_round, best_round=None, bar_round=None, mode='train',
                 *args, **kwargs):

        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, cnt_round, best_round, bar_round,
                         mode, **kwargs)

        self.action_sampling_size = self.dic_agent_conf["ACTION_SAMPLING_SIZE"]
        self.action_sampling_policy = self.dic_agent_conf["ACTION_SAMPLING_POLICY"]

        self.planning_sample_only = self.dic_agent_conf["PLANNING_SAMPLE_ONLY"]

    def choose_action(self, step, state, *args, **kwargs):
        
        kwargs.update(
            {
                'cnt_round': self.cnt_round
            }
        )

        action = PlanningOnlyAgent.choose_action(self, step, state, *args, **kwargs)

        if self.planning_sample_only:
            action = super().choose_action(step, state)

        return action

    def _choose_action(self, initial_step, one_state, intersection_index, previous_action, rng, possible_actions=None,
                       env=None, *args, **kwargs):

        if self.action_sampling_policy == 'best':
            q_values = self.q_network.predict(self.convert_state_to_input(one_state))
            sorted_q_values = sorted(enumerate(q_values[0]), key=lambda x: x[1], reverse=True)
            
            possible_actions = list(list(zip(*sorted_q_values[0: self.action_sampling_size]))[0])

        elif self.action_sampling_policy == 'random':
            all_actions = range(0, self.phases)

            possible_actions = rng.choice(all_actions, 3, replace=False)

        else:
            raise ValueError('Incorrect action sampling policy')

        action, rewards = PlanningOnlyAgent._choose_action(self, initial_step, one_state, intersection_index, previous_action,
                                                  rng, possible_actions, env, **kwargs)

        return action, rewards

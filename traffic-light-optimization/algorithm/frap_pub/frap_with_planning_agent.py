
from algorithm.frap_pub.frap_agent import FrapAgent
from algorithm.frap_pub.planning_only_agent import PlanningOnlyAgent


class FrapWithPlanningAgent(FrapAgent, PlanningOnlyAgent):
    
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

        intersection_index = kwargs.get('intersection_index', None)

        if self.mode == 'train':
            action = PlanningOnlyAgent.choose_action(self, step, state, *args, **kwargs)

            if self.planning_sample_only:
                action = FrapAgent.choose_action(self, step, state, intersection_index=intersection_index)
        elif self.mode == 'test':
            action = FrapAgent.choose_action(self, step, state, intersection_index=intersection_index)
        else:
            raise ValueError("Action not available for mode " + str(self.mode))

        return action

    def _choose_action(self, initial_step, one_state, original_step, intersection_index, rng, planning_iterations,
                       previous_planning_actions, possible_actions=None, env=None, *args, **kwargs):

        intersection_phases = self.phases[intersection_index]

        phase_indices = [self.unique_phases.index(intersection_phase) for intersection_phase in intersection_phases]

        if self.action_sampling_policy == 'best':
            q_values = self.q_network.predict(self.convert_state_to_input(one_state))
            sorted_q_values = sorted(zip(phase_indices, q_values[0][phase_indices]), key=lambda x: x[1], reverse=True)
            
            possible_actions = list(list(zip(*sorted_q_values[0: self.action_sampling_size]))[0])

        elif self.action_sampling_policy == 'random':
            sample_size = min(self.action_sampling_size, len(phase_indices))

            possible_actions = rng.choice(phase_indices, sample_size, replace=False)

        else:
            raise ValueError('Incorrect action sampling policy')

        action, rewards = PlanningOnlyAgent._choose_action(
            self, initial_step, one_state, original_step, intersection_index, rng, planning_iterations,
            previous_planning_actions, possible_actions, env, **kwargs)

        return action, rewards

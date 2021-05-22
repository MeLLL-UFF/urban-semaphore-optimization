import os
import copy
import pickle
from functools import partial
import statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from utils import xml_util

from algorithm.frap_pub.agent import Agent
from algorithm.frap_pub.definitions import ROOT_DIR


class PlanningOnlyAgent(Agent):
    
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, mode='test',
                 *args, **kwargs):

        if len(dic_traffic_env_conf['INTERSECTION_ID']) > 1:
            raise NotImplementedError("Planning supports one intersection only at this time")

        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, mode)

        self.env = None
        self.dic_exp_conf = dic_exp_conf

        self.phases = self.dic_traffic_env_conf['PHASE']
        self.planning_iterations = self.dic_agent_conf["PLANNING_ITERATIONS"]
        self.pick_action_and_keep_with_it = self.dic_agent_conf["PICK_ACTION_AND_KEEP_WITH_IT"]

        self.tiebreak_policy = self.dic_agent_conf["TIEBREAK_POLICY"]

        self.previous_actions = [None]*len(self.phases)

        xml_util.register_copyreg()

    def set_simulation_environment(self, env):
        self.env = env

    def choose_action(self, step, state, *args, **kwargs):

        intersection_index = kwargs.get('intersection_index', None)

        if intersection_index is None:
            raise ValueError('intersection_index must be declared')

        if self.mode == 'replay':
            action = self.replay_action(step, intersection_index)
        else:

            rng = np.random.Generator(np.random.MT19937(23423))

            previous_planning_actions = []
            if self.previous_actions[intersection_index] is not None:
                previous_planning_actions.append(self.previous_actions[intersection_index])

            action, _ = self._choose_action(step, state, step, intersection_index, rng,
                                            self.planning_iterations, previous_planning_actions)

            self.previous_actions[intersection_index] = action

        return action

    def replay_action(self, step, intersection_index):

        intersection_id = self.dic_traffic_env_conf['INTERSECTION_ID'][intersection_index]

        filename = 'inter' + '_' + intersection_id + '_' + 'actions' + '.pkl'

        records_path = self.dic_path['PATH_TO_WORK_DIRECTORY']
        file = os.path.join(ROOT_DIR, records_path, 'test_round', 'round_0', filename)

        with open(file, 'rb') as handle:
            data = pickle.load(handle)

        step_data = data[step]

        assert step_data['time'] == step

        action = step_data['action']

        return action

    def load_network(self, *args, **kwargs):
        pass

    def _choose_action(self, initial_step, one_state, original_step, intersection_index, rng, planning_iterations,
                       previous_planning_actions, possible_actions=None, env=None, *args, **kwargs):

        if possible_actions is None:
            possible_actions = range(0, len(self.phases[intersection_index]))

        if env is None:
            env = self.env

        save_state_filepath = env.save_state()

        # mutable objects need deep copy in the target function
        simulation_possibility_kwargs = {
            'env': env,
            'initial_step': initial_step,
            'one_state': one_state,  # deep copy needed
            'original_step': original_step,
            'intersection_index': intersection_index,
            'save_state_filepath': save_state_filepath,
            'rng_state': rng.bit_generator.state,  # deep copy needed
            'planning_iterations': planning_iterations,
            'previous_planning_actions': previous_planning_actions,  # deep copy needed
            'possible_actions': possible_actions,
        }

        simulation_possibility_kwargs.update(
            **kwargs
        )

        with ThreadPoolExecutor(max_workers=len(possible_actions)) as executor:
            possible_future_rewards = executor.map(
                partial(self._run_simulation_possibility, **simulation_possibility_kwargs),
                possible_actions
            )

        possible_future_rewards = list(possible_future_rewards)

        mean_rewards = [statistics.mean(future_rewards) for future_rewards in possible_future_rewards]

        best_actions = np.flatnonzero(mean_rewards == np.max(mean_rewards))

        if len(best_actions) > 1:
            if self.tiebreak_policy == 'random':
                action = rng.choice(best_actions)

            elif self.tiebreak_policy == 'maintain':
                if previous_planning_actions and previous_planning_actions[-1] in best_actions:
                    action = previous_planning_actions[-1]
                else:
                    action = rng.choice(best_actions)

            elif self.tiebreak_policy == 'change':
                if previous_planning_actions and previous_planning_actions[-1] in best_actions:
                    index = np.argwhere(best_actions == previous_planning_actions[-1])[0]
                    best_actions = np.delete(best_actions, index)

                action = rng.choice(best_actions)
            else:
                raise ValueError('Invalid tiebreak_policy: ' + str(self.tiebreak_policy))
        else:
            action = best_actions[0]

        rewards = possible_future_rewards[action]

        os.remove(save_state_filepath)

        return action, rewards

    def _run_simulation_possibility(
            self,
            action,
            env,
            initial_step,
            one_state,
            original_step,
            intersection_index,
            save_state_filepath,
            rng_state,
            planning_iterations,
            previous_planning_actions,
            possible_actions,
            **kwargs):

        try:

            one_state = copy.deepcopy(one_state)
            rng_state = copy.deepcopy(rng_state)
            previous_planning_actions = copy.deepcopy(previous_planning_actions)

            env = copy.deepcopy(env)

            env.external_configurations['SUMOCFG_PARAMETERS'].update(
                {
                    '--begin': initial_step,
                    '--load-state': save_state_filepath
                }
            )

            execution_name = 'planning_for_step' + '_' + str(original_step) + '__' + \
                             'previous_phases' + '_' + (
                                 '-'.join(str(x) for x in previous_planning_actions)
                                 if len(previous_planning_actions) else str(None)) + '__' + \
                             'initial_step' + '_' + str(initial_step) + '__' + \
                             'phase' + '_' + str(action)

            write_mode = False
            if self.mode == 'train':
                env.path_to_log = self.env.path_to_log + '__' + execution_name
                if not os.path.exists(ROOT_DIR + '/' + env.path_to_log):
                    os.makedirs(ROOT_DIR + '/' + env.path_to_log)
                write_mode = True

            env.write_mode = write_mode

            _, next_action = env.reset_for_planning(execution_name)
            rewards = []

            if self.dic_agent_conf["PICK_ACTION_AND_KEEP_WITH_IT"]:

                test_run_counts = planning_iterations * self.dic_traffic_env_conf["MIN_ACTION_TIME"]

                done = False
                step = 0
                while not done and step < test_run_counts:

                    action_list = ['no_op']*len(next_action)
                    action_list[intersection_index] = action

                    next_state, reward, done, steps_iterated, next_action = env.step(action_list)

                    one_state = next_state[intersection_index]
                    rewards.append(reward[0])
                    step += steps_iterated

            else:

                action_list = ['no_op']*len(next_action)
                action_list[intersection_index] = action

                next_state, reward, done, steps_iterated, _ = env.step(action_list)

                one_state = next_state[intersection_index]
                rewards.append(reward[0])

                previous_planning_actions.append(action)

                planning_iterations -= 1
                if planning_iterations > 0 or done:

                    rng = np.random.Generator(np.random.MT19937(23423))
                    rng.bit_generator.state = rng_state

                    _, future_rewards = self._choose_action(
                        initial_step + steps_iterated,
                        one_state,
                        original_step,
                        intersection_index,
                        rng,
                        planning_iterations,
                        previous_planning_actions,
                        possible_actions,
                        env,
                        **kwargs
                    )

                    rewards.extend(future_rewards)

            if self.mode == 'train':
                env.save_log()
            env.end_sumo()

        except Exception as e:
            print(e)
            raise e

        return rewards

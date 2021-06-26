import os
import copy
import pickle
from functools import partial
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from utils import xml_util
from algorithm.frap_pub.synchronization_util import save_log_lock

from algorithm.frap_pub.agent import Agent
from algorithm.frap_pub.definitions import ROOT_DIR


class PlanningOnlyAgent(Agent):
    
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, mode='test',
                 *args, **kwargs):

        if len(dic_traffic_env_conf['INTERSECTION_ID']) > 1:
            raise NotImplementedError("Planning supports one intersection only at this time")

        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, mode)

        self.rng = np.random.Generator(np.random.MT19937(23423))

        self.env = None
        self.dic_exp_conf = dic_exp_conf

        self.phases = self.dic_traffic_env_conf['PHASE']
        self.action_sampling_size = self.dic_agent_conf["ACTION_SAMPLING_SIZE"]
        self.planning_iterations = self.dic_agent_conf["PLANNING_ITERATIONS"]

        context = mp.get_context('spawn')
        self.process_pool_executor = ProcessPoolExecutor(
            max_workers=max(pow(self.action_sampling_size, self.planning_iterations), 16), mp_context=context)

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

            action = self._choose_action(step, state, intersection_index)

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

    def shutdown(self):
        self.process_pool_executor.shutdown()

    def _choose_action(self, step, state, intersection_index):

        previous_planning_actions = []
        if self.previous_actions[intersection_index] is not None:
            previous_planning_actions.append(self.previous_actions[intersection_index])

        save_state_filepath = self.env.save_state()

        states = [state]
        planning_step_list = [step]
        previous_planning_actions_list = [[]]
        save_state_filepath_list = [save_state_filepath]

        env = copy.deepcopy(self.env)

        env.path_to_log = env.path_to_log + '__' + 'planning'
        if not os.path.exists(ROOT_DIR + '/' + env.path_to_log):
            os.makedirs(ROOT_DIR + '/' + env.path_to_log)

        envs = [env]

        possible_future_rewards = np.array([])

        remaining_planning_iterations = self.planning_iterations

        simulation_possibility_kwargs = {
            'original_step': step,
            'intersection_index': intersection_index,
        }

        while remaining_planning_iterations > 0:

            save_state_filepath_to_remove_list = save_state_filepath_list

            possible_actions = []

            for state in states:
                possible_actions.extend(self._get_possible_actions(state, intersection_index))

            save_state_filepath_list = self._expand_data_list(save_state_filepath_list)

            envs = self._expand_data_list(envs, deepcopy=True)

            planning_step_list = self._expand_data_list(planning_step_list)

            previous_planning_actions_list = self._expand_data_list(previous_planning_actions_list, deepcopy=True)

            states, rewards, steps_iterated_list, save_state_filepath_list, envs = list(zip(*list(
                self.process_pool_executor.map(
                    partial(PlanningOnlyAgent._run_simulation_possibility, **simulation_possibility_kwargs),
                    possible_actions, planning_step_list, previous_planning_actions_list, save_state_filepath_list, envs
            ))))

            for save_state_filepath_to_remove in save_state_filepath_to_remove_list:
                os.remove(save_state_filepath_to_remove)

            if len(possible_future_rewards) != 0:
                for i in range(len(possible_future_rewards) - 1, -1, -1):
                    possible_future_rewards = \
                        np.concatenate([possible_future_rewards[:i],
                                        [possible_future_rewards[i]] * self.action_sampling_size,
                                        possible_future_rewards[i+1:]],
                                       axis=None)
            else:
                possible_future_rewards = np.array([0] * self.action_sampling_size)

            possible_future_rewards = np.add(possible_future_rewards, rewards)

            if len(previous_planning_actions_list) != 0:
                for i in range(len(previous_planning_actions_list) - 1, -1, -1):
                    previous_planning_actions_list[i].append(possible_actions[i])

            save_state_filepath_list = list(save_state_filepath_list)
            envs = list(envs)

            for i in range(len(planning_step_list) - 1, -1, -1):
                planning_step_list[i] += steps_iterated_list[i]

            remaining_planning_iterations -= 1

        for save_state_filepath_to_remove in save_state_filepath_list:
            os.remove(save_state_filepath_to_remove)

        best_actions = np.flatnonzero(possible_future_rewards == np.max(possible_future_rewards))

        if len(best_actions) > 1:
            if self.tiebreak_policy == 'random':
                action = self.rng.choice(best_actions)

            elif self.tiebreak_policy == 'maintain':
                if previous_planning_actions and previous_planning_actions[-1] in best_actions:
                    action = previous_planning_actions[-1]
                else:
                    action = self.rng.choice(best_actions)

            elif self.tiebreak_policy == 'change':
                if previous_planning_actions and previous_planning_actions[-1] in best_actions:
                    index = np.argwhere(best_actions == previous_planning_actions[-1])[0]
                    best_actions = np.delete(best_actions, index)

                action = self.rng.choice(best_actions)
            else:
                raise ValueError('Invalid tiebreak_policy: ' + str(self.tiebreak_policy))
        else:
            action = best_actions[0]

        return action

    def _expand_data_list(self, data, deepcopy=False):
        for i in range(len(data) - 1, -1, -1):
            if deepcopy:
                data = data[:i] + [copy.deepcopy(data[i]) for _ in range(self.action_sampling_size)] + data[i + 1:]
            else:
                data = data[:i] + [data[i]] * self.action_sampling_size + data[i + 1:]

        return data

    def _get_possible_actions(self, state, intersection_index):
        return range(0, len(self.phases[intersection_index]))

    @staticmethod
    def _run_simulation_possibility(
            action,
            planning_step,
            previous_planning_actions,
            save_state_filepath,
            env,
            original_step,
            intersection_index):

        try:

            xml_util.register_copyreg()

            env.external_configurations['SUMOCFG_PARAMETERS'].update(
                {
                    '--begin': planning_step,
                    '--load-state': save_state_filepath
                }
            )

            execution_name = 'planning_for_step' + '_' + str(original_step) + '__' + \
                             'previous_phases' + '_' + (
                                 '-'.join(str(x) for x in previous_planning_actions)
                                 if len(previous_planning_actions) else str(None)) + '__' + \
                             'planning_step' + '_' + str(planning_step) + '__' + \
                             'phase' + '_' + str(action)

            _, next_action = env.reset_for_planning(execution_name)

            action_list = ['no_op']*len(next_action)
            action_list[intersection_index] = action

            next_state, reward, done, steps_iterated, _ = env.step(action_list)
            next_state = next_state[0]
            reward = reward[0]

            save_state_filepath = env.save_state()

            save_log_lock.acquire()
            env.save_log()
            save_log_lock.release()
            env.end_sumo()

        except Exception as e:
            print(e)
            raise e

        return next_state, reward, steps_iterated, save_state_filepath, env

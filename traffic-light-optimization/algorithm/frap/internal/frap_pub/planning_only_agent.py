import os
import copy
from functools import partial
import statistics
import itertools

import numpy as np

from utils.process_util import NoDaemonPool

from algorithm.frap.internal.frap_pub.agent import Agent


class PlanningOnlyAgent(Agent):
    
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, env):

        if dic_traffic_env_conf["NUM_INTERSECTIONS"] > 1:
            raise NotImplementedError("Planning Only supports one intersection only at this time")

        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path)

        self.env = env
        self.dic_exp_conf = dic_exp_conf

        self.phases = self.dic_traffic_env_conf['PHASE']
        self.planning_iterations = self.dic_agent_conf["PLANNING_ITERATIONS"]
        self.pick_action_and_keep_with_it = self.dic_agent_conf["PICK_ACTION_AND_KEEP_WITH_IT"]

    def choose_action(self, initial_step, one_state):

        if self.pick_action_and_keep_with_it:
            possibilities = [[index]*self.planning_iterations for index, _ in enumerate(self.phases)]
        else:
            possibilities = list(itertools.product(range(0, len(self.phases)), repeat=self.planning_iterations))

        save_state_filepath = self.env.save_state()

        kwargs = {
            'initial_step': initial_step,
            'path_to_log': copy.deepcopy(self.env.path_to_log),
            'path_to_work_directory': copy.deepcopy(self.env.path_to_work_directory),
            'dic_traffic_env_conf': copy.deepcopy(self.env.dic_traffic_env_conf),
            'dic_path': copy.deepcopy(self.env.dic_path),
            'dic_exp_conf': copy.deepcopy(self.dic_exp_conf),
            'external_configurations': copy.deepcopy(self.env.external_configurations),
            'save_state_filepath': save_state_filepath,
            'env_mode': self.env.mode
        }
        
        with NoDaemonPool(processes=16) as pool:
            rewards = pool.map(
                partial(PlanningOnlyAgent._run_simulation_possibility, **kwargs),
                possibilities
            )

        index = np.random.choice(np.flatnonzero(rewards == np.max(rewards)))

        possibility = possibilities[index]

        os.remove(save_state_filepath)

        return possibility[0]

    @staticmethod
    def _run_simulation_possibility(
            possibility,
            initial_step, 
            path_to_log,
            path_to_work_directory,
            dic_traffic_env_conf,
            dic_path,
            dic_exp_conf,
            external_configurations,
            save_state_filepath, 
            env_mode):

        try:

            external_configurations['SUMOCFG_PARAMETERS'].pop('--log', None)
            external_configurations['SUMOCFG_PARAMETERS'].pop('--duration-log.statistics', None)

            external_configurations['SUMOCFG_PARAMETERS'].update(
                {
                    '--begin': initial_step,
                    '--load-state': save_state_filepath
                }
            )

            from algorithm.frap.internal.frap_pub.config import DIC_ENVS

            env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                path_to_log=path_to_log,
                path_to_work_directory=path_to_work_directory,
                dic_traffic_env_conf=dic_traffic_env_conf,
                dic_path=dic_path,
                external_configurations=external_configurations,
                mode=env_mode,
                sumo_output_enabled=False)

            done = False
            execution_name = 'planning_for' + '_' + '_' + 'possibility' + '_' + str(possibility) + '_' + \
                'initial_step' + '_' + str(initial_step)

            state, next_action = env.reset(execution_name)

            min_action_time = dic_traffic_env_conf["MIN_ACTION_TIME"]
            test_run_counts = min(len(possibility), dic_exp_conf["TEST_RUN_COUNTS"] - initial_step)

            possibility_iterator = iter(possibility)

            step = 0
            stop_cnt = 0
            rewards = []
            while not done and step < test_run_counts:
                action_list = [None]*len(next_action)

                new_actions_needed = np.where(np.array(next_action) == None)[0]
                for index in new_actions_needed:

                    action = next(possibility_iterator)

                    action_list[index] = action

                next_state, reward, done, steps_iterated, next_action, _ = env.step(action_list)

                rewards.append(reward[0])
                step += steps_iterated
                stop_cnt += steps_iterated

            env.end_sumo()

            mean_reward = statistics.mean(rewards)
        except Exception as e:
            print(e)
            raise e

        return mean_reward

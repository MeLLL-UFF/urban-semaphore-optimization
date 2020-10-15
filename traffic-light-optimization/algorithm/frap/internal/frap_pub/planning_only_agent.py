import os
import copy
from functools import partial
import statistics

import numpy as np

from utils.process_util import NoDaemonPool

from algorithm.frap.internal.frap_pub.agent import Agent


class PlanningOnlyAgent(Agent):
    
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, env):

        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path)

        self.env = env
        self.dic_exp_conf = dic_exp_conf

        self.phases = self.dic_traffic_env_conf['PHASE']

    def choose_action(self, step_num, one_state):

        initial_step = step_num

        save_state_filepath = self.env.save_state()

        test_run_counts = self.dic_agent_conf["PLANNING_TIME_MULTIPLIER"] * self.dic_traffic_env_conf["MIN_ACTION_TIME"]

        kwargs = {
            'initial_step': initial_step,
            'path_to_log': copy.deepcopy(self.env.path_to_log),
            'path_to_work_directory': copy.deepcopy(self.env.path_to_work_directory),
            'dic_agent_conf': copy.deepcopy(self.dic_agent_conf),
            'dic_traffic_env_conf': copy.deepcopy(self.env.dic_traffic_env_conf),
            'dic_path': copy.deepcopy(self.env.dic_path),
            'dic_exp_conf': copy.deepcopy(self.dic_exp_conf),
            'external_configurations': copy.deepcopy(self.env.external_configurations),
            'save_state_filepath': save_state_filepath,
            'env_mode': self.env.mode,
            'test_run_counts': test_run_counts
        }
        
        with NoDaemonPool(processes=1) as pool:
            rewards = pool.map(
                partial(PlanningOnlyAgent._run_simulation_possibility, **kwargs),
                range(0, len(self.phases))
            )

        index = np.random.choice(np.flatnonzero(rewards == np.max(rewards)))

        os.remove(save_state_filepath)

        return index

    @staticmethod
    def _run_simulation_possibility(
            phase_index,
            initial_step, 
            path_to_log,
            path_to_work_directory,
            dic_agent_conf,
            dic_traffic_env_conf,
            dic_path,
            dic_exp_conf,
            external_configurations,
            save_state_filepath, 
            env_mode,
            test_run_counts):
            
        
        external_configurations['SUMOCFG_PARAMETERS'].update(
            {
                '--begin': initial_step,
                '--load-state': save_state_filepath
            }
        )

        dic_exp_conf["TEST_RUN_COUNTS"] = test_run_counts

        from algorithm.frap.internal.frap_pub.config import DIC_AGENTS, DIC_ENVS
        
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
            path_to_log=path_to_log,
            path_to_work_directory=path_to_work_directory,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            external_configurations=external_configurations,
            mode=env_mode,
            sumo_output_enabled=False)

        agent_name = dic_exp_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            dic_exp_conf=dic_exp_conf,
            env=env
        )

        done = False
        execution_name = 'planning_for' + '_' + '_' + 'phase' + '_' + str(phase_index) + '_' + \
            'initial_step' + '_' + str(initial_step)
            
        state = env.reset(execution_name)
        
        dic_agent_conf["PLANNING_TIME_MULTIPLIER"]


        test_run_counts = dic_exp_conf["TEST_RUN_COUNTS"]
        if not dic_agent_conf["PICK_ACTION_AND_KEEP_WITH_IT"]:
            if dic_agent_conf["PLANNING_TIME_MULTIPLIER"] > 2:
                test_run_counts = 2 * dic_traffic_env_conf["MIN_ACTION_TIME"]
        min_action_time = dic_traffic_env_conf["MIN_ACTION_TIME"]
        
        step_num = 0
        stop_cnt = 0
        rewards = []
        while step_num < int(test_run_counts / min_action_time):
            action_list = []
            for one_state in state:

                if dic_agent_conf["PICK_ACTION_AND_KEEP_WITH_IT"] or step_num == 0:
                    action = phase_index
                else:
                    agent.dic_agent_conf["PLANNING_TIME_MULTIPLIER"] -= 1
                    action = agent.choose_action(initial_step + (step_num * min_action_time), one_state)

                action_list.append(action)

            next_state, reward, done, _ = env.step(action_list)

            state = next_state
            rewards.append(reward[0])
            step_num += 1
            stop_cnt += 1

        env.end_sumo()

        mean_reward = statistics.mean(rewards)

        return mean_reward

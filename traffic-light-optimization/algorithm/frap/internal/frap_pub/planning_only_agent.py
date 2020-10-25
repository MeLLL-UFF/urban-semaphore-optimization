import os
import copy
import uuid
from functools import partial
import statistics

import numpy as np

from utils.process_util import NoDaemonPool

from algorithm.frap.internal.frap_pub.agent import Agent

class PlanningOnlyAgent(Agent):
    
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, mode='test', tiebreak_policy='random',
                 *args, **kwargs):

        # tiebreak_policy='random', 'maintain', 'change'

        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, mode)

        self.env = None
        self.dic_exp_conf = dic_exp_conf

        self.phases = self.dic_traffic_env_conf['PHASE']
        self.planning_iterations = self.dic_agent_conf["PLANNING_ITERATIONS"]
        self.pick_action_and_keep_with_it = self.dic_agent_conf["PICK_ACTION_AND_KEEP_WITH_IT"]

        self.previous_action = None
        self.current_action = None

        self.tiebreak_policy = tiebreak_policy

    def set_simulation_environment(self, env):
        self.env = env
    
    def choose_action(self, initial_step, one_state):
        
        self.previous_action = self.current_action

        rng = np.random.Generator(np.random.MT19937(23423))

        action, _ = self._choose_action(initial_step, rng, self.previous_action)

        self.current_action = action

        return action
    
    def _choose_action(self, initial_step, rng, previous_action):

        save_state_filepath = self.env.save_state()

        test_run_counts = self.planning_iterations * self.dic_traffic_env_conf["MIN_ACTION_TIME"]

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
            'agent_mode': self.mode,
            'test_run_counts': test_run_counts,
            'rng': rng,
            'tiebreak_policy': self.tiebreak_policy
        }
        
        with NoDaemonPool(processes=len(self.phases)) as pool:
            possible_future_rewards = pool.map(
                partial(PlanningOnlyAgent._run_simulation_possibility, **kwargs),
                range(0, len(self.phases))
            )

        mean_rewards = [statistics.mean(future_rewards) for future_rewards in possible_future_rewards]

        best_actions = np.flatnonzero(mean_rewards == np.max(mean_rewards))

        if self.tiebreak_policy == 'random':
            action = rng.choice(best_actions)

        elif self.tiebreak_policy == 'maintain':
            if previous_action in best_actions:
                action = previous_action
            else:
                action = rng.choice(best_actions)

        elif self.tiebreak_policy == 'change':
            if len(best_actions) > 1 and previous_action in best_actions:
                index = np.argwhere(best_actions == previous_action)[0]
                best_actions = np.delete(best_actions, index)

            action = rng.choice(best_actions)
        else:
            raise ValueError('Invalid tiebreak_policy: ' + str(tiebreak_policy))

        rewards = possible_future_rewards[action]

        os.remove(save_state_filepath)

        return action, rewards

    @staticmethod
    def _run_simulation_possibility(
            action,
            initial_step, 
            path_to_log,
            path_to_work_directory,
            dic_agent_conf,
            dic_traffic_env_conf,
            dic_path,
            dic_exp_conf,
            external_configurations,
            save_state_filepath,
            agent_mode, 
            env_mode,
            test_run_counts,
            rng,
            tiebreak_policy):
            
        try:
            external_configurations['SUMOCFG_PARAMETERS'].pop('--log', None)
            external_configurations['SUMOCFG_PARAMETERS'].pop('--duration-log.statistics', None)

            external_configurations['SUMOCFG_PARAMETERS'].update(
                {
                    '--begin': initial_step,
                    '--load-state': save_state_filepath
                }
            )

            from algorithm.frap.internal.frap_pub.config import DIC_AGENTS, DIC_ENVS
            
            env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                path_to_log=path_to_log,
                path_to_work_directory=path_to_work_directory,
                dic_traffic_env_conf=dic_traffic_env_conf,
                dic_path=dic_path,
                external_configurations=external_configurations,
                mode=env_mode,
                sumo_output_enabled=False)

            execution_name = 'planning_for' + '_' + 'phase' + '_' + str(action) + '_' + \
                'initial_step' + '_' + str(initial_step) + '_' + str(uuid.uuid4())
                
            state, next_action = env.reset(execution_name)
            
            rewards = []

            if dic_agent_conf["PICK_ACTION_AND_KEEP_WITH_IT"]:

                done = False
                step = 0
                stop_cnt = 0
                while not done and step < test_run_counts:

                    action_list = [None]*len(next_action)

                    new_actions_needed = np.where(np.array(next_action) == None)[0]
                    for index in new_actions_needed:

                        action_list[index] = action

                    _, reward, done, steps_iterated, next_action, _ = env.step(action_list)

                    rewards.append(reward[0])
                    step += steps_iterated
                    stop_cnt += steps_iterated

            else:

                action_list = [None]*len(next_action)

                new_actions_needed = np.where(np.array(next_action) == None)[0]
                for index in new_actions_needed:

                    action_list[index] = action

                _, reward, _, steps_iterated, _, _ = env.step(action_list)

                rewards.append(reward[0])

                previous_action = action


                dic_agent_conf["PLANNING_ITERATIONS"] -= 1

                if dic_agent_conf["PLANNING_ITERATIONS"] > 0:

                    agent_name = dic_exp_conf["MODEL_NAME"]
                    agent = DIC_AGENTS[agent_name](
                        dic_agent_conf=dic_agent_conf,
                        dic_traffic_env_conf=dic_traffic_env_conf,
                        dic_path=dic_path,
                        dic_exp_conf=dic_exp_conf,
                        mode=agent_mode,
                        tiebreak_policy=tiebreak_policy
                    )
                    agent.set_simulation_environment(env)

                    _, future_rewards = agent._choose_action(initial_step + steps_iterated, rng, previous_action)

                    rewards.extend(future_rewards)

            if agent_mode == 'train':
                env.bulk_log()
            env.end_sumo()

        except Exception as e:
            print(e)
            raise e
        
        return rewards

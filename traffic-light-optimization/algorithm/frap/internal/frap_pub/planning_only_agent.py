import os
import copy
from functools import partial
import statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from utils import xml_util

from algorithm.frap.internal.frap_pub.agent import Agent
from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR


class PlanningOnlyAgent(Agent):
    
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, mode='test',
                 *args, **kwargs):

        if dic_traffic_env_conf["NUM_INTERSECTIONS"] > 1:
            raise NotImplementedError("Planning supports one intersection only at this time")

        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, mode)

        self.env = None
        self.dic_exp_conf = dic_exp_conf

        self.phases = self.dic_traffic_env_conf['PHASE']
        self.planning_iterations = self.dic_agent_conf["PLANNING_ITERATIONS"]
        self.pick_action_and_keep_with_it = self.dic_agent_conf["PICK_ACTION_AND_KEEP_WITH_IT"]

        self.previous_action = None
        self.current_action = None

        self.tiebreak_policy = self.dic_agent_conf["TIEBREAK_POLICY"]

        xml_util.register_copyreg()

    def set_simulation_environment(self, env):
        self.env = env

    def choose_action(self, step, state, *args, **kwargs):
        
        rng = np.random.Generator(np.random.MT19937(23423))
        
        self.previous_action = self.current_action

        intersection_index = kwargs.get('intersection_index', None)

        if intersection_index is None:
            raise ValueError('intersection_index must be declared')

        action, _ = self._choose_action(step, state, intersection_index, self.previous_action, rng, self.planning_iterations)

        self.current_action = action

        return action
    
    def _choose_action(self, initial_step, one_state, intersection_index, previous_action, rng, planning_iterations, 
                       possible_actions=None, env=None, *args, **kwargs):

        if possible_actions is None:
            possible_actions = range(0, len(self.phases))

        if env is None:
            env = self.env

        save_state_filepath = env.save_state()

        simulation_possibility_kwargs = {
            'initial_step': initial_step,
            'one_state': copy.deepcopy(one_state),
            'intersection_index': intersection_index,
            'save_state_filepath': save_state_filepath,
            'rng_state': copy.deepcopy(rng.bit_generator.state),
            'planning_iterations': planning_iterations,
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
            raise ValueError('Invalid tiebreak_policy: ' + str(self.tiebreak_policy))

        rewards = possible_future_rewards[action]

        os.remove(save_state_filepath)

        return action, rewards

    def _run_simulation_possibility(
            self,
            action,
            initial_step,
            one_state,
            intersection_index,
            save_state_filepath,
            rng_state,
            planning_iterations,
            possible_actions,
            **kwargs):
            
        try:

            env = copy.deepcopy(self.env)

            env.external_configurations['SUMOCFG_PARAMETERS'].pop('--log', None)
            env.external_configurations['SUMOCFG_PARAMETERS'].pop('--duration-log.statistics', None)

            env.external_configurations['SUMOCFG_PARAMETERS'].update(
                {
                    '--begin': initial_step,
                    '--load-state': save_state_filepath
                }
            )

            execution_name = 'planning_for' + '_' + 'phase' + '_' + str(action) + '_' + \
                'initial_step' + '_' + str(initial_step)

            write_mode = False
            if self.mode == 'train':
                env.path_to_log += '__' + execution_name
                if not os.path.exists(ROOT_DIR + '/' + env.path_to_log):
                    os.makedirs(ROOT_DIR + '/' + env.path_to_log)
                write_mode = True

            env.write_mode = write_mode
            env.sumo_output_enabled = False

            _, next_action = env.reset(execution_name)
            
            rewards = []

            if self.dic_agent_conf["PICK_ACTION_AND_KEEP_WITH_IT"]:

                test_run_counts = planning_iterations * self.dic_traffic_env_conf["MIN_ACTION_TIME"]

                done = False
                step = 0
                while not done and step < test_run_counts:

                    action_list = ['no_op']*len(next_action)
                    action_list[intersection_index] = action

                    next_state, reward, done, steps_iterated, next_action, _ = env.step(action_list)

                    one_state = next_state[intersection_index]
                    rewards.append(reward[0])
                    step += steps_iterated

            else:

                action_list = ['no_op']*len(next_action)
                action_list[intersection_index] = action

                next_state, reward, done, steps_iterated, _, _ = env.step(action_list)

                one_state = next_state[intersection_index]
                rewards.append(reward[0])

                previous_action = action

                planning_iterations -= 1
                if planning_iterations > 0 or done:

                    rng = np.random.Generator(np.random.MT19937(23423))
                    rng.bit_generator.state = rng_state

                    _, future_rewards = self._choose_action(
                        initial_step + steps_iterated, 
                        one_state,
                        intersection_index,
                        previous_action, 
                        rng,
                        planning_iterations,
                        possible_actions,
                        env,
                        **kwargs
                    )

                    rewards.extend(future_rewards)

            if self.mode == 'train':
                env.bulk_log()
            env.end_sumo()

        except Exception as e:
            print(e)
            raise e
        
        return rewards

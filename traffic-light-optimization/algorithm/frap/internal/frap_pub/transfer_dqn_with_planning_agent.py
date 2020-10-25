import os
import copy
from functools import partial
import statistics
import itertools

import numpy as np

from utils.process_util import NoDaemonPool

from algorithm.frap.internal.frap_pub.transfer_dqn_agent import TransferDQNAgent
from algorithm.frap.internal.frap_pub.planning_only_agent import PlanningOnlyAgent


class TransferDQNWithPlanningAgent(TransferDQNAgent):
    
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, 
                 cnt_round, best_round=None, bar_round=None, mode='test', 
                 *args, **kwargs):

        if dic_traffic_env_conf["NUM_INTERSECTIONS"] > 1:
            raise NotImplementedError("Planning supports one intersection only at this time")
        
        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, best_round, bar_round, mode)

        self.planning_component = PlanningOnlyAgent(self, dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, mode)

        self.env = None
        self.dic_exp_conf = dic_exp_conf

        self.phases = self.dic_traffic_env_conf['PHASE']
        self.planning_iterations = self.dic_agent_conf["PLANNING_ITERATIONS"]
        self.pick_action_and_keep_with_it = self.dic_agent_conf["PICK_ACTION_AND_KEEP_WITH_IT"]

        self.sample_only = self.dic_agent_conf["SAMPLE_ONLY"]

    def set_simulation_environment(self, env):
        self.env = env
        self.planning_component.set_simulation_environment(env)

    def choose_action(self, step_num, one_state):

        action = self.planning_component.choose_action(step_num, one_state)

        if self.sample_only:
            action = super().choose_action(step_num, one_state)

        return action

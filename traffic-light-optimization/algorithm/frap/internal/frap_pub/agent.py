import json
import os
import shutil


class Agent:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, dic_exp_conf, mode='test', 
                 *args, **kwargs):

        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.mode = mode

    def choose_action(self, step, state, *args, **kwargs):

        raise NotImplementedError

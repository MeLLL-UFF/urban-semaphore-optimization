import json
import os
import shutil


class Agent:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, mode='test', 
                 *args, **kwargs):

        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.mode = mode

    def choose_action(self):

        raise NotImplementedError

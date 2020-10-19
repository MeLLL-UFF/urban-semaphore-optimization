import os
import copy

import numpy as np

from algorithm.frap.internal.frap_pub.config import DIC_AGENTS, DIC_ENVS

from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR


class Generator:

    def __init__(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, best_round=None,
                 external_configurations={}):


        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.external_configurations = external_configurations

        # every generator's output
        # generator for pretraining
        if self.dic_exp_conf["PRETRAIN"]:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"], "train_round",
                                            "round_" + str(self.cnt_round), "generator_" + str(self.cnt_gen))
            if not os.path.exists(ROOT_DIR + '/' + self.path_to_log):
                os.makedirs(ROOT_DIR + '/' + self.path_to_log)

            self.agent_name = self.dic_exp_conf["PRETRAIN_MODEL_NAME"]
            self.agent = DIC_AGENTS[self.agent_name](
                dic_agent_conf=self.dic_agent_conf,
                dic_sumo_env_conf=self.dic_sumo_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round,
                best_round=best_round
            )

        else:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                            "round_"+str(self.cnt_round), "generator_"+str(self.cnt_gen))
            if not os.path.exists(ROOT_DIR + '/' + self.path_to_log):
                os.makedirs(ROOT_DIR + '/' + self.path_to_log)

            self.agent_name = self.dic_exp_conf["MODEL_NAME"]
            self.agent = DIC_AGENTS[self.agent_name](
                dic_agent_conf=self.dic_agent_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=self.cnt_round,
                best_round=best_round
            )

        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                              path_to_log = self.path_to_log,
                              path_to_work_directory = self.dic_path["PATH_TO_WORK_DIRECTORY"],
                              dic_traffic_env_conf = self.dic_traffic_env_conf,
                              dic_path=self.dic_path,
                              external_configurations=self.external_configurations,
                              mode='train')

    def generate(self):

        done = False

        execution_name = 'train' + '_' + \
                         'generator' + '_' + str(self.cnt_gen) + '_' + \
                         'round' + '_' + str(self.cnt_round)
        state, next_action = self.env.reset(execution_name)
        step = 0
        stop_cnt = 0
        while not done and step < self.dic_exp_conf["RUN_COUNTS"]:
            action_list = [None]*len(next_action)
            
            new_actions_needed = np.where(np.array(next_action) == None)[0]
            for index in new_actions_needed:
                
                one_state = state[index]

                action = self.agent.choose_action(step, one_state)

                action_list[index] = action

            next_state, reward, done, steps_iterated, next_action, _ = self.env.step(action_list)

            state = next_state
            step += steps_iterated
            stop_cnt += steps_iterated
        self.env.bulk_log()
        self.env.end_sumo()

        if self.dic_traffic_env_conf["DONE_ENABLE"]:
            run_cnt_log = open(os.path.join(ROOT_DIR, self.dic_path["PATH_TO_WORK_DIRECTORY"],
                                            "generator_stop_cnt_log.txt"), "a")
            run_cnt_log.write("%s, %10s, %d\n"%("generator", "round_"+str(self.cnt_round), stop_cnt))
            run_cnt_log.close()

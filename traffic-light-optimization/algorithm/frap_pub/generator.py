import os
import copy

import numpy as np

from algorithm.frap_pub.config import DIC_AGENTS, DIC_ENVS
from algorithm.frap_pub.construct_sample import ConstructSample
from algorithm.frap_pub.updater import Updater
from algorithm.frap_pub.definitions import ROOT_DIR
from algorithm.frap_pub import synchronization_util


class Generator:

    def __init__(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 best_round=None, bar_round=None, external_configurations=None):

        if external_configurations is None:
            external_configurations = {}

        self.cnt_round = cnt_round
        self.cnt_gen = cnt_gen
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.best_round = best_round
        self.bar_round = bar_round
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
                dic_exp_conf=self.dic_exp_conf,
                cnt_round=self.cnt_round,
                best_round=best_round,
                mode='train'
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
                dic_exp_conf=self.dic_exp_conf,
                cnt_round=self.cnt_round,
                best_round=best_round,
                mode='train'
            )

        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                              path_to_log=self.path_to_log,
                              path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
                              dic_traffic_env_conf=self.dic_traffic_env_conf,
                              dic_path=self.dic_path,
                              external_configurations=self.external_configurations,
                              mode='train')
        
        if self.agent_name == 'PlanningOnly' or self.agent_name == 'FrapWithPlanning':
            self.agent.set_simulation_environment(self.env)

        if self.agent_name in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE_BETWEEN_STEPS"]:
            self.updater = Updater(
                cnt_round=self.cnt_round,
                dic_agent_conf=self.dic_agent_conf,
                dic_exp_conf=self.dic_exp_conf,
                dic_traffic_env_conf=self.dic_traffic_env_conf,
                dic_path=self.dic_path,
                best_round=self.best_round,
                bar_round=self.bar_round,
                agent=self.agent
            )

    def generate(self):

        done = False

        execution_name = 'train' + '_' + \
                         'generator' + '_' + str(self.cnt_gen) + '_' + \
                         'round' + '_' + str(self.cnt_round)

        if self.agent_name in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE_BETWEEN_STEPS"]:
            make_reward_start_index = 0
            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(ROOT_DIR + '/' + train_round):
                os.makedirs(ROOT_DIR + '/' + train_round)
            cs = ConstructSample(path_to_samples=train_round, cnt_round=self.cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)

        state, next_action = self.env.reset(execution_name)
        step = 0
        stop_cnt = 0
        while not done and step < self.dic_exp_conf["RUN_COUNTS"]:
            action_list = [None]*len(next_action)
            
            new_actions_needed = np.where(np.array(next_action) == None)[0]
            for index in new_actions_needed:
                
                one_state = state[index]

                action = self.agent.choose_action(step, one_state, intersection_index=index)

                action_list[index] = action

            next_state, reward, done, steps_iterated, next_action, _ = self.env.step(action_list)

            if self.agent_name in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE_BETWEEN_STEPS"]:
                if step > self.dic_agent_conf['UPDATE_START'] and step % self.dic_agent_conf['UPDATE_PERIOD'] == 0:

                    self.env.bulk_log()

                    # synchronize here
                    i = synchronization_util.network_update_begin_barrier.wait()

                    if i == 0:
                        cs.make_reward(start_index=make_reward_start_index)
                        make_reward_start_index = step

                        self.updater.load_sample()
                        self.updater.update_network()

                    # synchronize here
                    synchronization_util.network_update_end_barrier.wait()

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

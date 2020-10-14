import os
import copy
from algorithm.frap.internal.frap_pub.config import DIC_AGENTS, DIC_ENVS

from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR


class Planner:

    def __init__(self, cnt_round, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 external_configurations={}):

        self.cnt_round = cnt_round
        self.dic_exp_conf = dic_exp_conf
        self.dic_path = dic_path
        self.dic_agent_conf = copy.deepcopy(dic_agent_conf)
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.external_configurations = external_configurations

        self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                        "round_"+str(self.cnt_round))
        if not os.path.exists(ROOT_DIR + '/' + self.path_to_log):
            os.makedirs(ROOT_DIR + '/' + self.path_to_log)


        self.env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](
                        path_to_log = self.path_to_log,
                        path_to_work_directory = self.dic_path["PATH_TO_WORK_DIRECTORY"],
                        dic_traffic_env_conf = self.dic_traffic_env_conf,
                        dic_path=self.dic_path,
                        external_configurations=self.external_configurations,
                        mode='test')

        self.agent_name = self.dic_exp_conf["MODEL_NAME"]
        self.agent = DIC_AGENTS[self.agent_name](
            dic_agent_conf=self.dic_agent_conf,
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
            dic_exp_conf=self.dic_exp_conf,
            env=self.env
        )

    def plan(self):

        done = False
        execution_name = 'test' + '_' + 'round' + '_' + str(self.cnt_round)
        state = self.env.reset(execution_name)
        step_num = 0
        stop_cnt = 0

        test_run_counts = self.dic_exp_conf["TEST_RUN_COUNTS"] 
        min_action_time = self.dic_traffic_env_conf["MIN_ACTION_TIME"]

        while not done and step_num < int(test_run_counts / min_action_time):
            action_list = []
            for one_state in state:

                action = self.agent.choose_action(step_num * min_action_time, one_state)

                action_list.append(action)

            next_state, reward, done, _ = self.env.step(action_list)

            state = next_state
            step_num += 1
            stop_cnt += 1
        self.env.bulk_log()
        self.env.end_sumo()

        if self.dic_traffic_env_conf["DONE_ENABLE"]:
            run_cnt_log = open(os.path.join(ROOT_DIR, self.dic_path["PATH_TO_WORK_DIRECTORY"],
                                            "generator_stop_cnt_log.txt"), "a")
            run_cnt_log.write("%s, %10s, %d\n"%("generator", "round_"+str(self.cnt_round), stop_cnt))
            run_cnt_log.close()

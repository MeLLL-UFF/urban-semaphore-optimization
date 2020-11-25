import os
import pickle

import numpy as np

from algorithm.frap_pub.definitions import ROOT_DIR


class ConstructSample:

    def __init__(self, path_to_samples, cnt_round, dic_traffic_env_conf):
        self.parent_dir = path_to_samples
        self.path_to_samples = path_to_samples + "/round_" + str(cnt_round)
        self.cnt_round = cnt_round
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_phase_expansion = self.dic_traffic_env_conf["phase_expansion"]

        self.measure_time = None
        self.min_action_time = None
        self.logging_data = None
        self.samples = None

    def load_data(self, folder):

        try:
            # load settings
            self.measure_time = self.dic_traffic_env_conf["MEASURE_TIME"]
            self.min_action_time = self.dic_traffic_env_conf["MIN_ACTION_TIME"]
            f_logging_data = open(os.path.join(ROOT_DIR, self.path_to_samples, folder, "inter_0.pkl"), "rb")
            self.logging_data = pickle.load(f_logging_data)
            f_logging_data.close()
            return 1

        except FileNotFoundError:
            print(os.path.join(ROOT_DIR, self.path_to_samples, folder), "files not found")
            return 0

    def construct_state(self, features, index, time):
        state = self.logging_data[index]
        assert time == state["time"]
        if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
            state_after_selection = {}
            for key, value in state["state"].items():
                if key in features:
                    if key == "cur_phase":
                        state_after_selection[key] = self.dic_phase_expansion[value[0]]
                    else:
                        state_after_selection[key] = value
        else:
            state_after_selection = {key: value for key, value in state["state"].items() if key in features}
        return state_after_selection

    def construct_reward(self, index, time):

        rs = self.logging_data[index + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        r_instant = rs['reward']

        # average
        list_r = []
        for i, t in zip(range(index, index + self.measure_time), range(time, time + self.measure_time)):
            rs = self.logging_data[i]
            assert t == rs["time"]
            r = rs['reward']
            list_r.append(r)
        r_average = np.average(list_r)

        return r_instant, r_average

    def judge_action(self, index):
        if self.logging_data[index]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data[index]['action']

    def make_reward(self, start_index=0):
        self.samples = []
        for folder in os.listdir(ROOT_DIR + '/' + self.path_to_samples):
            if "generator" not in folder:
                continue

            if not self.load_data(folder):
                continue
            list_samples = []
            total_time = int(self.logging_data[-1]['time'] + 1)
            # construct samples
            for index in range(start_index, len(self.logging_data) - self.measure_time + 1, self.min_action_time):
                time = int(self.logging_data[index]['time'])
                state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"], index, time)
                reward_instant, reward_average = self.construct_reward(index, time)
                action = self.judge_action(index)

                if time + self.min_action_time == total_time:
                    next_state = self.construct_state(
                        self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                        index + self.min_action_time - 1,
                        time + self.min_action_time - 1
                    )
                else:
                    next_state = self.construct_state(
                        self.dic_traffic_env_conf["LIST_STATE_FEATURE"],
                        index + self.min_action_time,
                        time + self.min_action_time
                    )

                sample = [state, action, next_state, reward_average, reward_instant, time]
                list_samples.append(sample)

            list_samples = self.evaluate_sample(list_samples)
            self.samples.extend(list_samples)

        self.dump_sample(self.samples, "")

    def evaluate_sample(self, list_samples):
        return list_samples

    def dump_sample(self, samples, folder):
        if folder == "":
            with open(os.path.join(ROOT_DIR, self.parent_dir, "total_samples.pkl"),"ab+") as f:
                pickle.dump(samples, f, -1)
        else:
            with open(os.path.join(ROOT_DIR, self.path_to_samples, folder, "samples_{0}.pkl".format(folder)), 'wb') as f:
                pickle.dump(samples, f, -1)

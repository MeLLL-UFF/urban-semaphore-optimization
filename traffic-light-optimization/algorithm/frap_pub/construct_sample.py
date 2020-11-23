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
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))
        self.dic_phase_expansion = self.dic_traffic_env_conf["phase_expansion"]

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

    def get_reward_from_features(self, rs):
        reward = {}
        reward["sum_lane_queue_length"] = np.sum(rs["lane_queue_length"])
        reward["sum_lane_wait_time"] = np.sum(rs["lane_sum_waiting_time"])
        reward["sum_lane_num_vehicle_left"] = np.sum(rs["lane_num_vehicle_left"])
        reward["sum_duration_vehicle_left"] = np.sum(rs["lane_sum_duration_vehicle_left"])
        reward["sum_num_vehicle_been_stopped_thres01"] = np.sum(rs["lane_num_vehicle_been_stopped_thres01"])
        reward["sum_num_vehicle_been_stopped_thres1"] = np.sum(rs["lane_num_vehicle_been_stopped_thres1"])
        return reward

    def cal_reward(self, rs, rewards_components):
        r = 0
        for component, weight in rewards_components.items():
            if weight == 0:
                continue
            if component not in rs.keys():
                continue
            if rs[component] is None:
                continue
            r += rs[component] * weight
        return r

    def construct_reward(self,rewards_components, index, time):

        rs = self.logging_data[index + self.measure_time - 1]
        assert time + self.measure_time - 1 == rs["time"]
        rs = self.get_reward_from_features(rs['state'])
        r_instant = self.cal_reward(rs, rewards_components)

        # average
        list_r = []
        for i, t in zip(range(index, index + self.measure_time), range(time, time + self.measure_time)):
            #print("t is ", t)
            rs = self.logging_data[i]
            assert t == rs["time"]
            rs = self.get_reward_from_features(rs['state'])
            r = self.cal_reward(rs, rewards_components)
            list_r.append(r)
        r_average = np.average(list_r)

        return r_instant, r_average

    def judge_action(self, index):
        if self.logging_data[index]['action'] == -1:
            raise ValueError
        else:
            return self.logging_data[index]['action']

    def make_reward(self):
        self.samples = []
        for folder in os.listdir(ROOT_DIR + '/' + self.path_to_samples):
            if "generator" not in folder:
                continue

            if not self.load_data(folder):
                continue
            list_samples = []
            initial_time = int(self.logging_data[0]['time'])
            total_time = int(self.logging_data[-1]['time'] + 1)
            # construct samples
            for index in range(0, len(self.logging_data) - self.measure_time + 1, self.min_action_time):
                time = int(self.logging_data[index]['time'])
                state = self.construct_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"], index, time)
                reward_instant, reward_average = self.construct_reward(
                    self.dic_traffic_env_conf["DIC_REWARD_INFO"], index, time)
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

    def evaluate_sample(self,list_samples):
        return list_samples

    def dump_sample(self, samples, folder):
        if folder == "":
            with open(os.path.join(ROOT_DIR, self.parent_dir, "total_samples.pkl"),"ab+") as f:
                pickle.dump(samples, f, -1)
        else:
            with open(os.path.join(ROOT_DIR, self.path_to_samples, folder, "samples_{0}.pkl".format(folder)), 'wb') as f:
                pickle.dump(samples, f, -1)



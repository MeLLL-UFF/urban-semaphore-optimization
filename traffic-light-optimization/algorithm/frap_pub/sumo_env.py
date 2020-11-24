import os
import sys
import pickle
import json
import uuid
from sys import platform

import numpy as np
import pandas as pd
import traci
import traci.constants as tc
from sumolib import checkBinary

from utils import sumo_util, sumo_traci_util

from algorithm.frap_pub.definitions import ROOT_DIR
from algorithm.frap_pub.intersection import Intersection
from algorithm.frap_pub import synchronization_util

class SumoEnv:

    LIST_LANE_VARIABLES_TO_SUB = [
        "LAST_STEP_VEHICLE_NUMBER",
        "LAST_STEP_VEHICLE_ID_LIST",
        "LAST_STEP_VEHICLE_HALTING_NUMBER",
        "VAR_WAITING_TIME",

        "LANE_EDGE_ID",
        ### "LAST_STEP_VEHICLE_ID_LIST",
        "VAR_LENGTH",
        "LAST_STEP_MEAN_SPEED",
        "VAR_MAXSPEED"
    ]

    LIST_VEHICLE_VARIABLES_TO_SUB = [
        "VAR_POSITION",
        "VAR_SPEED",
        # "VAR_ACCELERATION",
        # "POSITION_LON_LAT",
        "VAR_WAITING_TIME",
        "VAR_ACCUMULATED_WAITING_TIME",
        # "VAR_LANEPOSITION_LAT",
        "VAR_LANEPOSITION",
        
        ### "VAR_SPEED",
        "VAR_ALLOWED_SPEED",
        "VAR_MINGAP",
        "VAR_TAU",
        ### "VAR_LANEPOSITION",
        # "VAR_LEADER",  # Problems with subscription
        # "VAR_SECURE_GAP",  # Problems with subscription
        "VAR_LENGTH",
        "VAR_LANE_ID",
        "VAR_DECEL",

        'VAR_WIDTH',
        ### 'VAR_LENGTH',
        ### 'VAR_POSITION',
        'VAR_ANGLE',
        ### 'VAR_SPEED',
        'VAR_STOPSTATE',
        ### 'VAR_LANE_ID',
        ### 'VAR_WAITING_TIME',
        'VAR_EDGES',
        'VAR_ROUTE_INDEX'
    ]

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf, dic_path,
                 external_configurations=None, mode='train', write_mode=True, sumo_output_enabled=True):

        if external_configurations is None:
            external_configurations = {}

        # mode: train, test, replay

        if mode != 'train' and mode != 'test' and mode != 'replay':
            raise ValueError("Mode must be either 'train', 'test', or replay, current value is " + mode)
        self.mode = mode
        self.write_mode = write_mode
        self.sumo_output_enabled = sumo_output_enabled

        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.external_configurations = external_configurations

        base_dir = self.path_to_work_directory.rsplit('/', 1)[0]
        self.environment_state_path = os.path.join(base_dir, 'environment', 'temp')

        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            # raise ValueError

        if self.write_mode:
            # touch new inter_{}.pkl (if exists, remove)
            for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
                path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
                f = open(ROOT_DIR + '/' + path_to_log_file, "wb")
                f.close()

        self.execution_name = None

    def reset(self, execution_name):

        self.execution_name = execution_name + '__' + str(uuid.uuid4())

        # initialize intersections
        # self.list_intersection = [Intersection(i, self.LIST_VEHICLE_VARIABLES_TO_SUB)
        #                           for i in range(self.dic_sumo_env_conf["NUM_INTERSECTIONS"])]

        self.list_intersection = []
        for i in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            for j in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
                self.list_intersection.append(Intersection("{0}_{1}".format(i, j), self.LIST_VEHICLE_VARIABLES_TO_SUB,
                                                           self.dic_traffic_env_conf, self.dic_path,
                                                           execution_name=self.execution_name,
                                                           external_configurations=self.external_configurations))

        self.list_inter_log = [[] for _ in range(len(self.list_intersection))]
        # get lanes list
        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        if self.sumo_output_enabled:

            output_file = self.external_configurations['SUMOCFG_PARAMETERS']['--log']

            split_output_filename = output_file.rsplit('.', 2)
            execution_base = split_output_filename[0].rsplit('/', 1)[1]
            split_output_filename[0] += '_' + self.execution_name
            output_file = '.'.join(split_output_filename)

            split_output_filename = output_file.rsplit('/', 1)
            split_output_filename.insert(1, execution_base)
            output_file = '/'.join(split_output_filename)

            output_file_path = output_file.rsplit('/', 1)[0]
            if not os.path.isdir(output_file_path):
                os.makedirs(output_file_path)

            self.external_configurations['SUMOCFG_PARAMETERS']['--log'] = output_file
        else:
            self.external_configurations['SUMOCFG_PARAMETERS'].pop('--log', None)

        sumo_cmd_str = self._get_sumo_cmd()

        stops_to_issue = []
        print("start sumo")
        synchronization_util.traci_start_lock.acquire()
        try:
            traci.start(sumo_cmd_str, label=self.execution_name)
        except Exception as e:
            traci.close()

            if '--load-state' in self.external_configurations['SUMOCFG_PARAMETERS']:

                save_state = self.external_configurations['SUMOCFG_PARAMETERS']['--load-state']
                time = self.external_configurations['SUMOCFG_PARAMETERS']['--begin']

                net_file = self.external_configurations['SUMOCFG_PARAMETERS']['-n']
                net_xml = sumo_util.get_xml(net_file)
                stops_to_issue = sumo_util.fix_save_state_stops(net_xml, save_state, time)

            try:
                traci.start(sumo_cmd_str, label=self.execution_name)
            except Exception as e:
                print('TRACI TERMINATED')
                traci.close()
                print(str(e))
                raise e

        traci_connection = traci.getConnection(self.execution_name)
        print("succeed in start sumo")
        synchronization_util.traci_start_lock.release()

        for stop_info in stops_to_issue:
            traci_connection.vehicle.setStop(**stop_info)

        # start subscription
        for lane in self.list_lanes:
            traci_connection.lane.subscribe(lane, [getattr(tc, var) for var in self.LIST_LANE_VARIABLES_TO_SUB])

        # get new measurements
        for inter in self.list_intersection:
            inter.update_current_measurements()

        state, done = self.get_state()

        next_action = [None]*len(self.list_intersection)

        return state, next_action

    @staticmethod
    def convert_dic_to_df(dic):
        list_df = []
        for key in dic:
            df = pd.Series(dic[key], name=key)
            list_df.append(df)
        return pd.DataFrame(list_df)

    def bulk_log(self):

        if self.write_mode:

            valid_flag = {}
            for inter_ind, inter in enumerate(self.list_intersection):
                path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
                dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
                df = self.convert_dic_to_df(dic_vehicle)
                df.to_csv(ROOT_DIR + '/' + path_to_log_file, na_rep="nan")

                feature = inter.get_feature()
                if max(feature['lane_num_vehicle']) > self.dic_traffic_env_conf["VALID_THRESHOLD"]:
                    valid_flag[inter_ind] = 0
                else:
                    valid_flag[inter_ind] = 1
            json.dump(valid_flag, open(os.path.join(ROOT_DIR, self.path_to_log, "valid_flag.json"), "w"))

            for inter_ind in range(len(self.list_inter_log)):
                path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
                f = open(ROOT_DIR + '/' + path_to_log_file, "wb+")
                pickle.dump(self.list_inter_log[inter_ind], f)
                f.close()

                if self.mode == 'test':
                    path_to_detailed_log_file = os.path.join(
                        self.path_to_log, "inter_{0}_detailed.pkl".format(inter_ind))
                    f = open(ROOT_DIR + '/' + path_to_detailed_log_file, "wb+")
                    pickle.dump(self.list_inter_log[inter_ind], f)
                    f.close()

            self.list_inter_log = [[] for _ in range(len(self.list_intersection))]

    def end_sumo(self):
        traci_connection = traci.getConnection(self.execution_name)
        traci_connection.close()

    def get_current_time(self):
        traci_connection = traci.getConnection(self.execution_name)
        return traci_connection.simulation.getTime()

    def get_feature(self):

        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self):

        list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
                      for inter in self.list_intersection]
        done = self._check_episode_done(list_state)

        return list_state, done

    def get_reward(self):

        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"])
                       for inter in self.list_intersection]

        return list_reward

    def log(self, cur_time, before_action_feature, action, reward):

        for inter_ind, inter in enumerate(self.list_intersection):

            if self.mode == 'test':

                traffic_light = sumo_traci_util.get_traffic_light_state(inter.node_light, self.execution_name)
                time_loss = sumo_traci_util.get_time_loss(inter.dic_vehicle_sub_current_step, self.execution_name)
                relative_occupancy = sumo_traci_util.get_lane_relative_occupancy(
                    inter.dic_lane_sub_current_step,
                    inter.dic_lane_vehicle_sub_current_step,
                    inter.dic_vehicle_sub_current_step,
                    inter.edges, 
                    self.execution_name)
                relative_mean_speed = sumo_traci_util.get_relative_mean_speed(
                    inter.dic_lane_sub_current_step, inter.edges)
                absolute_number_of_cars = sumo_traci_util.get_absolute_number_of_cars(
                    inter.dic_lane_sub_current_step, inter.edges)

                extra = {
                    "traffic_light": traffic_light,
                    "time_loss": time_loss,
                    "relative_occupancy": relative_occupancy,
                    "relative_mean_speed": relative_mean_speed,
                    "absolute_number_of_cars": absolute_number_of_cars
                }

                self.list_inter_log[inter_ind].append({
                    "time": cur_time,
                    "state": before_action_feature[inter_ind],
                    "action": action[inter_ind],
                    "reward": reward[inter_ind],
                    "extra": extra})
            else:
                self.list_inter_log[inter_ind].append({
                    "time": cur_time,
                    "state": before_action_feature[inter_ind],
                    "action": action[inter_ind],
                    "reward": reward[inter_ind]})

    def save_state(self, name=None):

        if not os.path.isdir(ROOT_DIR + '/' + self.environment_state_path):
            os.makedirs(ROOT_DIR + '/' + self.environment_state_path)

        if name is None:
            state_name = self.execution_name + '_' + 'save_state' + '_' + str(self.get_current_time()) + '.sbx'

        filepath = os.path.join(ROOT_DIR, self.environment_state_path, state_name)

        traci_connection = traci.getConnection(self.execution_name)
        traci_connection.simulation.saveState(filepath)

        return filepath

    def check_for_active_action_time_actions(self, action):
        
        for inter_ind, inter in enumerate(self.list_intersection):
            
            action_time_action = inter.select_active_action_time_action()
            
            if action_time_action != -1:
                action[inter_ind] = action_time_action

        return action
    
    def check_for_time_restricted_actions(self, action, waiting_time_restriction=120):

        for inter_ind, inter in enumerate(self.list_intersection):

            time_restricted_action = inter.select_action_based_on_time_restriction(waiting_time_restriction)
            
            if time_restricted_action != -1:
                action[inter_ind] = time_restricted_action

        return action

    def step(self, action):

        waiting_time_restriction = self.dic_traffic_env_conf["WAITING_TIME_RESTRICTION"]

        action = self.check_for_time_restricted_actions(action, waiting_time_restriction)
        action = self.check_for_active_action_time_actions(action)

        if None in action:
            raise ValueError('Action cannot be None')

        step = 0
        average_reward_action = 0
        while None not in action:

            instant_time = self.get_current_time()

            before_action_feature = self.get_feature()
            state = self.get_state()

            # _step
            self._inner_step(action)

            # get reward
            reward = self.get_reward()
            average_reward_action = (average_reward_action*step + reward[0])/(step+1)

            if step == 0 or self.dic_traffic_env_conf['DEBUG']:
                print("time: {0}, phase: {1}, time this phase: {2}, action: {3}, reward: {4}".
                      format(instant_time,
                             before_action_feature[0]["cur_phase"],
                             before_action_feature[0]["time_this_phase"],
                             action[0],
                             reward[0]))

            # log
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action,
                     reward=reward)

            next_state, done = self.get_state()

            step += 1

            action = [None]*len(self.list_intersection)
            action = self.check_for_active_action_time_actions(action)

        next_action = action

        return next_state, reward, done, step, next_action, [average_reward_action]

    def _inner_step(self, action):

        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()

        # set signals
        for inter_ind, inter in enumerate(self.list_intersection):

            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                all_red_time=self.dic_traffic_env_conf["ALL_RED_TIME"]
            )

        # run one step

        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            traci_connection = traci.getConnection(self.execution_name)
            traci_connection.simulationStep()

        deadlock_waiting_too_long_threshold = self.dic_traffic_env_conf["DEADLOCK_WAITING_TOO_LONG_THRESHOLD"]

        # get new measurements
        for inter in self.list_intersection:
            inter.update_current_measurements()

            blocked_vehicles = sumo_traci_util.detect_deadlock(inter.net_file_xml, inter.dic_vehicle_sub_current_step, 
                waiting_too_long_threshold=deadlock_waiting_too_long_threshold, traci_label=self.execution_name)
            sumo_traci_util.resolve_deadlock(blocked_vehicles, inter.net_file_xml, inter.dic_vehicle_sub_current_step, 
                traci_label=self.execution_name)

    def _check_episode_done(self, list_state):

        # ======== to implement ========

        return False

    def _get_sumo_cmd(self):

        if platform == "linux" or platform == "linux2":
            if os.environ['SUMO_HOME'] == '/usr/share/sumo':
                sumo_binary = r"/usr/bin/sumo-gui"
                sumo_binary_nogui = r"/usr/bin/sumo"
            elif os.environ['SUMO_HOME'] == '/headless/sumo':
                sumo_binary = r"/headless/sumo/bin/sumo-gui"
                sumo_binary_nogui = r"/headless/sumo/bin/sumo"
            else:
                sys.exit("linux sumo binary path error")
            # for FIB-Server
            # sumo_binary = r"/usr/bin/sumo/bin/sumo-gui"
            # sumo_binary_nogui = r"/usr/bin/sumo"
        elif platform == "darwin":
            sumo_binary = r"/opt/local/bin/sumo-gui"
            sumo_binary_nogui = r"/opt/local/bin/sumo"
        elif platform == "win32":
            sumo_binary = checkBinary('sumo-gui')
            sumo_binary_nogui = checkBinary('sumo')
            # sumo_binary = r'D:\\software\\sumo-0.32.0\\bin\\sumo-gui.exe'
            # sumo_binary_nogui = r'D:\\software\\sumo-0.32.0\\bin\\sumo.exe'
        else:
            sys.exit("platform error")

        sumocfg_file = self.external_configurations['SUMOCFG_FILE']

        real_path_to_sumo_files = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                                               self.path_to_work_directory, sumocfg_file)

        sumocfg_parameters = self.external_configurations['SUMOCFG_PARAMETERS']

        if not sumocfg_parameters:
            sumocfg_parameters = {
                '-c': r'{0}'.format(real_path_to_sumo_files)
            }

        sumocfg_parameters['--step-length'] = str(self.dic_traffic_env_conf["INTERVAL"])

        if not self.write_mode:
            sumocfg_parameters.pop('--log', None)

        sumocfg_parameters_list = [str(item)
                                   for key_value_pair in sumocfg_parameters.items()
                                   for item in key_value_pair]

        sumo_cmd = [sumo_binary, *sumocfg_parameters_list]

        sumo_cmd_nogui = [sumo_binary_nogui, *sumocfg_parameters_list]

        if self.dic_traffic_env_conf["IF_GUI"]:
            return sumo_cmd
        else:
            return sumo_cmd_nogui

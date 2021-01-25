import os
import sys
import pickle
import uuid

import numpy as np
import traci
import traci.constants as tc

from utils import sumo_util, sumo_traci_util, xml_util

from algorithm.frap_pub.definitions import ROOT_DIR
from algorithm.frap_pub.intersection import Intersection
from algorithm.frap_pub import synchronization_util


class SumoEnv:

    LANE_VARIABLES_TO_SUBSCRIBE = [
        tc.LAST_STEP_VEHICLE_NUMBER,
        tc.LAST_STEP_VEHICLE_ID_LIST,
        tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
        tc.VAR_WAITING_TIME,

        tc.LANE_EDGE_ID,
        ### tc.LAST_STEP_VEHICLE_ID_LIST,
        tc.VAR_LENGTH,
        tc.LAST_STEP_MEAN_SPEED,
        tc.VAR_MAXSPEED
    ]

    VEHICLE_VARIABLES_TO_SUBSCRIBE = [
        tc.VAR_POSITION,
        tc.VAR_SPEED,
        # tc.VAR_ACCELERATION,
        # tc.POSITION_LON_LAT,
        tc.VAR_WAITING_TIME,
        tc.VAR_ACCUMULATED_WAITING_TIME,
        # tc."VAR_LANEPOSITION_LAT,
        tc.VAR_LANEPOSITION,
        
        ### tc."VAR_SPEED",
        tc.VAR_ALLOWED_SPEED,
        tc.VAR_MINGAP,
        tc.VAR_TAU,
        ### tc.VAR_LANEPOSITION,
        # tc.VAR_LEADER,  # Problems with subscription
        # tc.VAR_SECURE_GAP,  # Problems with subscription
        tc.VAR_LENGTH,
        tc.VAR_LANE_ID,
        tc.VAR_DECEL,

        tc.VAR_WIDTH,
        ### tc.VAR_LENGTH,
        ### tc.VAR_POSITION,
        tc.VAR_ANGLE,
        ### tc.VAR_SPEED,
        tc.VAR_STOPSTATE,
        ### tc.VAR_LANE_ID,
        ### tc.VAR_WAITING_TIME,
        tc.VAR_EDGES,
        tc.VAR_ROUTE_INDEX
    ]

    SIMULATION_VARIABLES_TO_SUBSCRIBE = [
        tc.VAR_LOADED_VEHICLES_NUMBER,
        tc.VAR_DEPARTED_VEHICLES_NUMBER
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

        net_file = os.path.join(ROOT_DIR, dic_path["PATH_TO_WORK_DIRECTORY"], dic_traffic_env_conf['NET_FILE'])
        self.net_file_xml = xml_util.get_xml(net_file)

        base_dir = self.path_to_work_directory.rsplit('/', 1)[0]
        self.environment_state_path = os.path.join(base_dir, 'environment', 'temp')

        self.intersections = None
        self.intersection_logs = None
        self.action_logs = None
        self.network_logs = None
        self.lanes_list = None
        self.edges_list = None

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            pass
            # raise ValueError

        if self.write_mode:
            # touch new inter_{}.pkl (if exists, remove)
            for intersection_id in self.dic_traffic_env_conf['INTERSECTION_ID']:
                path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(intersection_id))
                f = open(ROOT_DIR + '/' + path_to_log_file, "wb")
                f.close()

        self.execution_name = None

        self.current_step_lane_subscription = None
        self.previous_step_lane_subscription = None
        self.current_step_vehicle_subscription = None
        self.previous_step_vehicle_subscription = None
        self.current_step_lane_vehicle_subscription = None
        self.previous_step_lane_vehicle_subscription = None
        self.current_step_vehicles = []
        self.previous_step_vehicles = []

        self.vehicle_arrive_leave_time_dict = {}  # cumulative

    def reset(self, execution_name):

        self.execution_name = execution_name + '__' + str(uuid.uuid4())

        self.intersections = []

        intersection_ids = self.dic_traffic_env_conf['INTERSECTION_ID']

        for intersection_index in range(0, len(intersection_ids)):
            self.intersections.append(Intersection(intersection_index,
                                                   self.VEHICLE_VARIABLES_TO_SUBSCRIBE,
                                                   self.dic_traffic_env_conf,
                                                   self.dic_path,
                                                   execution_name=self.execution_name,
                                                   external_configurations=self.external_configurations))

        self.intersection_logs = [[] for _ in range(len(self.intersections))]
        self.action_logs = [[] for _ in range(len(self.intersections))]
        self.network_logs = []

        self.edges_list = []
        self.lanes_list = []
        for intersection in self.intersections:
            self.edges_list += intersection.all_edges
            self.lanes_list += intersection.all_lanes

        self.edges_list = np.unique(self.edges_list).tolist()
        self.lanes_list = np.unique(self.lanes_list).tolist()

        self.current_step_lane_subscription = None
        self.previous_step_lane_subscription = None
        self.current_step_vehicle_subscription = None
        self.previous_step_vehicle_subscription = None
        self.current_step_lane_vehicle_subscription = None
        self.previous_step_lane_vehicle_subscription = None
        self.current_step_vehicles = []
        self.previous_step_vehicles = []

        self.total_loaded_vehicles = 0
        self.total_departed_vehicles = 0
        self.total_running_vehicles = 0

        self.vehicle_arrive_leave_time_dict = {}  # cumulative

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

            # Sumo 1.7.0 only
            '''
            if '--load-state' in self.external_configurations['SUMOCFG_PARAMETERS']:

                save_state = self.external_configurations['SUMOCFG_PARAMETERS']['--load-state']
                time = self.external_configurations['SUMOCFG_PARAMETERS']['--begin']

                net_file = self.external_configurations['SUMOCFG_PARAMETERS']['-n']
                net_xml = xml_util.get_xml(net_file)
                stops_to_issue = sumo_util.fix_save_state_stops(net_xml, save_state, time)
            '''

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

        print('SUMO VERSION', traci_connection.getVersion()[1])

        # start subscription
        for lane in self.lanes_list:
            traci_connection.lane.subscribe(lane, [var for var in self.LANE_VARIABLES_TO_SUBSCRIBE])

        traci_connection.simulation.subscribe([var for var in self.SIMULATION_VARIABLES_TO_SUBSCRIBE])

        # get new measurements
        self.update_current_measurements()
        for intersection in self.intersections:
            intersection.update_current_measurements()

        state, done = self.get_state()

        next_action = [None]*len(self.intersections)

        return state, next_action

    def end_sumo(self):
        traci_connection = traci.getConnection(self.execution_name)
        traci_connection.close()

    def update_previous_measurements(self):

        self.previous_step_lane_subscription = self.current_step_lane_subscription
        self.previous_step_vehicle_subscription = self.current_step_vehicle_subscription
        self.previous_step_lane_vehicle_subscription = self.current_step_lane_vehicle_subscription
        self.previous_step_vehicles = self.current_step_vehicles

        self.previous_simulation_subscription = self.current_simulation_subscription

    def update_current_measurements(self):

        traci_connection = traci.getConnection(self.execution_name)

        # ====== lane level observations =======

        self.current_step_lane_subscription = {lane_id: traci_connection.lane.getSubscriptionResults(lane_id)
                                               for lane_id in self.lanes_list}

        self.current_simulation_subscription = traci_connection.simulation.getSubscriptionResults()

        self.total_loaded_vehicles += self.current_simulation_subscription[tc.VAR_LOADED_VEHICLES_NUMBER]
        self.total_departed_vehicles += self.current_simulation_subscription[tc.VAR_DEPARTED_VEHICLES_NUMBER]
        self.total_running_vehicles = traci.getConnection(self.execution_name).vehicle.getIDCount()

        # ====== vehicle level observations =======

        # get vehicle list
        current_step_vehicles = []
        for lane_id, values in self.current_step_lane_subscription.items():
            lane_vehicles = self.current_step_lane_subscription[lane_id][tc.LAST_STEP_VEHICLE_ID_LIST]
            current_step_vehicles += lane_vehicles

        self.current_step_vehicles = current_step_vehicles
        recently_arrived_vehicles = list(set(self.current_step_vehicles) - set(self.previous_step_vehicles))
        recently_left_vehicles = list(set(self.previous_step_vehicles) - set(self.current_step_vehicles))

        # update subscriptions
        for vehicle_id in recently_arrived_vehicles:
            traci_connection.vehicle.subscribe(vehicle_id, [var for var in self.VEHICLE_VARIABLES_TO_SUBSCRIBE])

        # vehicle level observations
        self.current_step_vehicle_subscription = {
            vehicle_id: traci_connection.vehicle.getSubscriptionResults(vehicle_id)
            for vehicle_id in self.current_step_vehicles
        }

        self.current_step_lane_vehicle_subscription = {}
        for vehicle_id, values in self.current_step_vehicle_subscription.items():
            lane_id = values[tc.VAR_LANE_ID]
            if lane_id in self.current_step_lane_vehicle_subscription:
                self.current_step_lane_vehicle_subscription[lane_id][vehicle_id] = \
                    self.current_step_vehicle_subscription[vehicle_id]
            else:
                self.current_step_lane_vehicle_subscription[lane_id] = \
                    {vehicle_id: self.current_step_vehicle_subscription[vehicle_id]}

        # update vehicle arrive and left time
        self._update_arrive_time(recently_arrived_vehicles)
        self._update_left_time(recently_left_vehicles)

    def _update_arrive_time(self, list_vehicles_arrive):

        time = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle_id in list_vehicles_arrive:
            if vehicle_id not in self.vehicle_arrive_leave_time_dict:
                self.vehicle_arrive_leave_time_dict[vehicle_id] = \
                    {"enter_time": time, "leave_time": np.nan}
            else:
                print("vehicle already exists!")
                sys.exit(-1)

    def _update_left_time(self, list_vehicles_left):

        time = self.get_current_time()
        # update the time for vehicle to leave
        for vehicle_id in list_vehicles_left:
            try:
                self.vehicle_arrive_leave_time_dict[vehicle_id]["leave_time"] = time
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def get_current_time(self):
        traci_connection = traci.getConnection(self.execution_name)
        return traci_connection.simulation.getTime()

    def get_feature(self, feature_name_list):

        feature_list = [intersection.get_feature(feature_name_list) for intersection in self.intersections]
        return feature_list

    def get_state(self):

        state_list = [intersection.get_state(self.dic_traffic_env_conf["STATE_FEATURE_LIST"])
                      for intersection in self.intersections]
        done = self._check_episode_done(state_list)

        return state_list, done

    def get_reward(self):

        reward_list = [intersection.get_reward(self.dic_traffic_env_conf["REWARD_INFO_DICT"])
                       for intersection in self.intersections]

        return reward_list

    def log(self, current_time, state_feature, action, reward):

        relative_occupancy_by_lane = {}
        relative_mean_speed_by_lane = {}
        absolute_number_of_cars_by_lane = {}

        if self.mode == 'test' or self.mode == 'replay':

            time_loss = sumo_traci_util.get_time_loss(
                self.current_step_vehicle_subscription,
                self.execution_name)
            relative_occupancy_by_lane = sumo_traci_util.get_lane_relative_occupancy(
                self.current_step_lane_subscription,
                self.current_step_lane_vehicle_subscription,
                self.execution_name)
            relative_mean_speed_by_lane = sumo_traci_util.get_lane_relative_mean_speed(
                self.current_step_lane_subscription)
            absolute_number_of_cars_by_lane = sumo_traci_util.get_lane_absolute_number_of_cars(
                self.current_step_lane_subscription)
            extra = {
                "time_loss": time_loss + (self.total_loaded_vehicles - self.total_departed_vehicles) * 1,
                "total_loaded_vehicles": self.total_loaded_vehicles,
                "total_departed_vehicles": self.total_departed_vehicles,
                "total_running_vehicles": self.total_running_vehicles,
                "relative_occupancy": relative_occupancy_by_lane,
                "relative_mean_speed": relative_mean_speed_by_lane,
                "absolute_number_of_cars_by_lane": absolute_number_of_cars_by_lane
            }

            self.network_logs.append({
                "time": current_time,
                "extra": extra})

        for intersection_index, intersection in enumerate(self.intersections):

            if self.mode == 'replay':

                traffic_light = sumo_traci_util.get_traffic_light_state(
                    intersection.id,
                    self.execution_name)
                time_loss = sumo_traci_util.get_time_loss(
                    intersection.current_step_vehicle_subscription,
                    self.execution_name)

                lanes = intersection.all_lanes

                intersection_relative_occupancy_by_lane = {
                    lane: relative_occupancy_by_lane[lane] for lane in lanes
                }
                intersection_relative_mean_speed_by_lane = {
                    lane: relative_mean_speed_by_lane[lane] for lane in lanes
                }
                intersection_absolute_number_of_cars_by_lane = {
                    lane: absolute_number_of_cars_by_lane[lane] for lane in lanes
                }

                extra = {
                    "traffic_light": traffic_light,
                    "time_loss": time_loss,
                    "relative_occupancy": intersection_relative_occupancy_by_lane,
                    "relative_mean_speed": intersection_relative_mean_speed_by_lane,
                    "absolute_number_of_cars_by_lane": intersection_absolute_number_of_cars_by_lane
                }

            else:
                extra = {}

            self.intersection_logs[intersection_index].append({
                "time": current_time,
                "state": state_feature[intersection_index],
                "action": action[intersection_index],
                "reward": reward[intersection_index],
                "extra": extra})

            if self.mode == 'test':

                self.action_logs[intersection_index].append({
                    "time": current_time,
                    "action": action[intersection_index],
                })

    def save_log(self):

        if self.write_mode:

            if self.mode == 'test' or self.mode == 'replay':
                path_to_detailed_log_file = os.path.join(self.path_to_log, "network_detailed.pkl")
                f = open(ROOT_DIR + '/' + path_to_detailed_log_file, "wb+")
                pickle.dump(self.network_logs, f)
                f.close()

            for intersection_index, intersection in enumerate(self.intersections):

                path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(intersection.id))
                f = open(ROOT_DIR + '/' + path_to_log_file, "wb+")
                pickle.dump(self.intersection_logs[intersection_index], f)
                f.close()

                if self.mode == 'replay':
                    path_to_detailed_log_file = os.path.join(
                        self.path_to_log, "inter_{0}_detailed.pkl".format(intersection.id))
                    f = open(ROOT_DIR + '/' + path_to_detailed_log_file, "wb+")
                    pickle.dump(self.intersection_logs[intersection_index], f)
                    f.close()

                if self.mode == 'test':
                    path_to_actions_log_file = os.path.join(
                        self.path_to_log, "inter_{0}_actions.pkl".format(intersection.id))
                    f = open(ROOT_DIR + '/' + path_to_actions_log_file, "wb+")
                    pickle.dump(self.action_logs[intersection_index], f)
                    f.close()

            self.intersection_logs = [[] for _ in range(len(self.intersections))]
            self.action_logs = [[] for _ in range(len(self.intersections))]
            self.network_logs = []

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
        
        for intersection_index, intersection in enumerate(self.intersections):
            
            action_time_action = intersection.select_active_action_time_action()
            
            if action_time_action != -1:
                action[intersection_index] = action_time_action

        return action
    
    def check_for_time_restricted_actions(self, action, waiting_time_restriction=120):

        for intersection_index, intersection in enumerate(self.intersections):

            time_restricted_action = intersection.select_action_based_on_time_restriction(waiting_time_restriction)
            
            if time_restricted_action != -1:
                action[intersection_index] = time_restricted_action

        return action

    def step(self, action):

        waiting_time_restriction = self.dic_traffic_env_conf["WAITING_TIME_RESTRICTION"]

        action = self.check_for_time_restricted_actions(action, waiting_time_restriction)
        action = self.check_for_active_action_time_actions(action)

        if None in action:
            raise ValueError('Action cannot be None')

        step = 0
        while None not in action:

            instant_time = self.get_current_time()

            state_feature = self.get_feature(
                list(set(self.dic_traffic_env_conf['STATE_FEATURE_LIST'] + ['current_phase', 'time_this_phase'])))

            # _step
            self._inner_step(action)

            # get reward
            reward = self.get_reward()

            if step == 0 or self.dic_traffic_env_conf['DEBUG']:
                print("time: {0}, phase: {1}, time this phase: {2}, action: {3}, reward: {4}".
                      format(instant_time,
                             state_feature[0]['current_phase'],
                             state_feature[0]['time_this_phase'],
                             action[0],
                             reward[0]))

            self.log(current_time=instant_time, state_feature=state_feature, action=action, reward=reward)

            next_state, done = self.get_state()

            step += 1

            action = [None]*len(self.intersections)
            action = self.check_for_active_action_time_actions(action)

        next_action = action

        return next_state, reward, done, step, next_action

    def _inner_step(self, action):

        # copy current measurements to previous measurements
        self.update_previous_measurements()
        for intersection in self.intersections:
            intersection.update_previous_measurements()

        # set signals
        for intersection_index, intersection in enumerate(self.intersections):

            intersection.set_signal(
                action=action[intersection_index],
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
        self.update_current_measurements()
        for intersection in self.intersections:
            intersection.update_current_measurements()

            # Sumo 1.7.0 only
            '''
            blocked_vehicles = sumo_traci_util.detect_deadlock(
                intersection.id,
                intersection.net_file_xml,
                intersection.current_step_vehicle_subscription,
                waiting_too_long_threshold=deadlock_waiting_too_long_threshold,
                traci_label=self.execution_name
            )

            sumo_traci_util.resolve_deadlock(
                blocked_vehicles,
                intersection.net_file_xml,
                intersection.current_step_vehicle_subscription,
                traci_label=self.execution_name
            )
            '''

    def _check_episode_done(self, state_list):

        # ======== to implement ========

        return False

    def _get_sumo_cmd(self):

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

        sumo_binary = sumo_util.get_sumo_binary(self.dic_traffic_env_conf["IF_GUI"])
        sumo_cmd = [sumo_binary, *sumocfg_parameters_list]

        return sumo_cmd

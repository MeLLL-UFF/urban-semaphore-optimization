import os
import sys
import copy
import pickle
import uuid

import numpy as np
import traci
import traci.constants as tc

from utils import sumo_util, sumo_traci_util, xml_util

from algorithm.frap_pub.definitions import ROOT_DIR
from algorithm.frap_pub.intersection import Intersection


class SumoEnv:

    LANE_VARIABLES_TO_SUBSCRIBE = [
        tc.LAST_STEP_VEHICLE_NUMBER,
        tc.LAST_STEP_VEHICLE_ID_LIST,
        tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
        tc.VAR_WAITING_TIME,
        tc.VAR_LENGTH,
        tc.LAST_STEP_MEAN_SPEED,
        tc.VAR_MAXSPEED
    ]

    VEHICLE_VARIABLES_TO_SUBSCRIBE = [
        tc.VAR_POSITION,
        tc.VAR_SPEED,
        tc.VAR_WAITING_TIME,
        tc.VAR_LANEPOSITION,
        tc.VAR_ALLOWED_SPEED,
        tc.VAR_MINGAP,
        tc.VAR_TAU,
        # tc.VAR_LEADER,  # Problems with subscription
        # tc.VAR_SECURE_GAP,  # Problems with subscription
        tc.VAR_LENGTH,
        tc.VAR_LANE_ID,
        tc.VAR_DECEL,
        tc.VAR_WIDTH,
        tc.VAR_ANGLE,
        tc.VAR_STOPSTATE,
    ]

    SIMULATION_VARIABLES_TO_SUBSCRIBE = [
        tc.VAR_DEPARTED_VEHICLES_NUMBER,
        tc.VAR_PENDING_VEHICLES,
        tc.VAR_DEPARTED_VEHICLES_IDS,
        tc.VAR_ARRIVED_VEHICLES_IDS
    ]

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf, dic_path,
                 external_configurations=None, mode='train', write_mode=True):

        if external_configurations is None:
            external_configurations = {}

        # mode: train, test, replay

        if mode != 'train' and mode != 'test' and mode != 'replay':
            raise ValueError("Mode must be either 'train', 'test', or replay, current value is " + mode)
        self.mode = mode
        self.write_mode = write_mode

        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.external_configurations = copy.deepcopy(external_configurations)

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
        movement_to_yellow_time_list = self.dic_traffic_env_conf["MOVEMENT_TO_YELLOW_TIME"]
        for movement_to_yellow_time in movement_to_yellow_time_list:
            for _, yellow_time in movement_to_yellow_time.items():
                assert yellow_time < self.dic_traffic_env_conf["MIN_ACTION_TIME"]

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

        self.vehicle_departure_arrival_time_dict = {}  # cumulative
        self.travel_time_dict = {}
        self.time_loss_dict = {}
        self.vehicle_pending_departure_time_dict = {}
        self.pending_travel_time_dict = {}
        self.pending_time_loss_dict = {}

        self.total_time_loss = 0

        self.is_planning = False

    def reset(self, execution_name):

        self.execution_name = execution_name + '__' + str(uuid.uuid4())
        self.is_planning = False

        self.intersections = []

        intersection_ids = self.dic_traffic_env_conf['INTERSECTION_ID']

        for intersection_index in range(0, len(intersection_ids)):
            self.intersections.append(Intersection(intersection_index,
                                                   self.dic_traffic_env_conf,
                                                   self.dic_path,
                                                   execution_name=self.execution_name,
                                                   external_configurations=self.external_configurations))

        self.intersection_logs = [[] for _ in range(len(self.intersections))]
        self.action_logs = [[] for _ in range(len(self.intersections))]
        self.network_logs = []

        # for intersection_index, intersection in enumerate(self.intersections):
        #     path_to_actions_log_file = os.path.join(
        #         self.path_to_log, "inter_{0}_actions.pkl".format(intersection.id))
        #     f = open(ROOT_DIR + '/' + path_to_actions_log_file, "rb")
        #     self.action_logs[intersection_index] = pickle.load(f)
        #     f.close()

        self.edges_list = \
            sumo_util.get_all_edges(self.net_file_xml) + sumo_util.get_all_internal_edges(self.net_file_xml)
        self.lanes_list = \
            sumo_util.get_all_lanes(self.net_file_xml) + sumo_util.get_all_internal_lanes(self.net_file_xml)

        self.current_step_lane_subscription = None
        self.previous_step_lane_subscription = None
        self.current_step_vehicle_subscription = None
        self.previous_step_vehicle_subscription = None
        self.current_step_lane_vehicle_subscription = None
        self.previous_step_lane_vehicle_subscription = None
        self.current_step_vehicles = []
        self.previous_step_vehicles = []

        self.total_departed_vehicles = 0
        self.total_pending_vehicles = 0
        self.total_running_vehicles = 0
        self.total_arrived_vehicles = 0

        self.vehicle_departure_arrival_time_dict = {}  # cumulative
        self.travel_time_dict = {}
        self.time_loss_dict = {}
        self.pending_travel_time_dict = {}
        self.vehicle_pending_departure_time_dict = {}
        self.pending_time_loss_dict = {}

        self.total_time_loss = 0

        sumo_cmd_str = self._get_sumo_cmd()

        print("start sumo")
        trace_file_path = ROOT_DIR + '/' + self.path_to_log + '/' + 'trace_file_log.txt'
        try:
            if traci.isLibsumo():
                version = traci.start(sumo_cmd_str, traceFile=trace_file_path, traceGetters=False)
            else:
                version = traci.start(sumo_cmd_str, label=self.execution_name, doSwitch=False,
                                      traceFile=trace_file_path, traceGetters=False)
        except Exception as e:

            try:
                self.end_sumo()
            except Exception as e:
                print(str(e))

            try:
                if traci.isLibsumo():
                    version = traci.start(sumo_cmd_str, traceFile=trace_file_path, traceGetters=False)
                else:
                    version = traci.start(sumo_cmd_str, label=self.execution_name, doSwitch=False,
                                          traceFile=trace_file_path, traceGetters=False)
            except Exception as e:
                print('TRACI TERMINATED')
                self.end_sumo()
                print(str(e))
                raise e

        traci_connection = sumo_traci_util.get_traci_connection(self.execution_name)
        print("succeed in start sumo")
        print('SUMO VERSION', version[1])

        # start subscription
        for lane in self.lanes_list:
            traci_connection.lane.subscribe(lane, [var for var in self.LANE_VARIABLES_TO_SUBSCRIBE])

        traci_connection.simulation.subscribe([var for var in self.SIMULATION_VARIABLES_TO_SUBSCRIBE])

        # get new measurements
        self.update_current_measurements()
        for intersection in self.intersections:
            intersection.reset()
            intersection.update_current_measurements()

        state, done = self.get_state()

        next_action = [None]*len(self.intersections)

        return state, next_action

    def reset_for_planning(self, execution_name):

        self.execution_name = execution_name + '__' + str(uuid.uuid4())
        self.is_planning = True

        for intersection in self.intersections:
            intersection.execution_name = self.execution_name

        self.intersection_logs = [[] for _ in range(len(self.intersections))]
        self.action_logs = [[] for _ in range(len(self.intersections))]
        self.network_logs = []

        sumo_cmd_str = self._get_sumo_cmd()

        print("start sumo")
        trace_file_path = ROOT_DIR + '/' + self.path_to_log + '/' + 'trace_file_log.txt'
        try:
            if traci.isLibsumo():
                version = traci.start(sumo_cmd_str)
            else:
                version = sumo_traci_util.start(sumo_cmd_str, label=self.execution_name,
                                                numRetries=100, waitBetweenRetries=0.01,
                                                doSwitch=False,
                                                traceFile=trace_file_path, traceGetters=False)

        except Exception as e:

            try:
                self.end_sumo()
            except Exception as e:
                print(str(e))

            try:
                if traci.isLibsumo():
                    version = traci.start(sumo_cmd_str)
                else:
                    version = sumo_traci_util.start(sumo_cmd_str, label=self.execution_name,
                                                    numRetries=100, waitBetweenRetries=0.01,
                                                    doSwitch=False,
                                                    traceFile=trace_file_path, traceGetters=False)
            except Exception as e:
                print('TRACI TERMINATED')
                self.end_sumo()
                print(str(e))
                raise e

        traci_connection = sumo_traci_util.get_traci_connection(self.execution_name)
        print("succeed in start sumo")
        print('SUMO VERSION', version[1])

        # start subscription
        for lane in self.lanes_list:
            traci_connection.lane.subscribe(lane, [var for var in self.LANE_VARIABLES_TO_SUBSCRIBE])

        traci_connection.simulation.subscribe([var for var in self.SIMULATION_VARIABLES_TO_SUBSCRIBE])

        vehicle_ids = traci_connection.vehicle.getIDList()
        for vehicle_id in vehicle_ids:
            traci_connection.vehicle.subscribe(vehicle_id, [var for var in self.VEHICLE_VARIABLES_TO_SUBSCRIBE])

        state, done = self.get_state()

        next_action = [None] * len(self.intersections)

        return state, next_action

    def end_sumo(self):
        sumo_traci_util.close_connection(self.execution_name)

    def update_previous_measurements(self):

        self.previous_step_lane_subscription = self.current_step_lane_subscription
        self.previous_step_vehicle_subscription = self.current_step_vehicle_subscription
        self.previous_step_lane_vehicle_subscription = self.current_step_lane_vehicle_subscription
        self.previous_step_vehicles = self.current_step_vehicles

        self.previous_simulation_subscription = self.current_simulation_subscription

    def update_current_measurements(self):

        traci_connection = sumo_traci_util.get_traci_connection(self.execution_name)

        # ====== lane level observations =======

        self.current_step_lane_subscription = {lane_id: traci_connection.lane.getSubscriptionResults(lane_id)
                                               for lane_id in self.lanes_list}

        self.current_simulation_subscription = traci_connection.simulation.getSubscriptionResults()

        recently_departed_vehicles = self.current_simulation_subscription[tc.VAR_DEPARTED_VEHICLES_IDS]
        recently_arrived_vehicles = self.current_simulation_subscription[tc.VAR_ARRIVED_VEHICLES_IDS]

        self.total_departed_vehicles += len(recently_departed_vehicles)
        self.total_pending_vehicles = len(self.current_simulation_subscription[tc.VAR_PENDING_VEHICLES])
        self.total_running_vehicles = traci_connection.vehicle.getIDCount()
        self.total_arrived_vehicles += len(recently_arrived_vehicles)

        # ====== vehicle level observations =======

        # get vehicle list
        current_step_vehicles = []
        for lane_id, values in self.current_step_lane_subscription.items():
            lane_vehicles = self.current_step_lane_subscription[lane_id][tc.LAST_STEP_VEHICLE_ID_LIST]
            current_step_vehicles += lane_vehicles
        self.current_step_vehicles = current_step_vehicles

        # update subscriptions
        for vehicle_id in recently_departed_vehicles:
            traci_connection.vehicle.subscribe(vehicle_id, [var for var in self.VEHICLE_VARIABLES_TO_SUBSCRIBE])

        # vehicle level observations
        self.current_step_vehicle_subscription = {
            vehicle_id: traci_connection.vehicle.getSubscriptionResults(vehicle_id)
            for vehicle_id in current_step_vehicles
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

        if not self.is_planning:

            # update vehicle arrive and left time
            self._update_pending_departure_time(self.current_simulation_subscription[tc.VAR_PENDING_VEHICLES])
            self._update_departure_time(recently_departed_vehicles)
            self._update_arrival_time(recently_arrived_vehicles)

            self._update_pending_time_loss(self.current_simulation_subscription[tc.VAR_PENDING_VEHICLES])
            self._update_time_loss(self.current_step_vehicles)

    def _update_pending_departure_time(self, pending_vehicles):

        time = self.get_current_time()
        for vehicle_id in pending_vehicles:
            if vehicle_id not in self.vehicle_pending_departure_time_dict:
                self.vehicle_pending_departure_time_dict[vehicle_id] = \
                    {"departure_time": time}
                self.pending_time_loss_dict[vehicle_id] = 0

    def _update_departure_time(self, recently_departed_vehicles):

        time = self.get_current_time()
        for vehicle_id in recently_departed_vehicles:
            if vehicle_id not in self.vehicle_departure_arrival_time_dict:
                self.vehicle_departure_arrival_time_dict[vehicle_id] = \
                    {"departure_time": time, "arrival_time": np.nan}
                self.time_loss_dict[vehicle_id] = 0

    def _update_arrival_time(self, recently_arrived_vehicles):

        time = self.get_current_time()
        for vehicle_id in recently_arrived_vehicles:
            try:
                self.vehicle_departure_arrival_time_dict[vehicle_id]["arrival_time"] = time
                self.travel_time_dict[vehicle_id] = \
                    self.vehicle_departure_arrival_time_dict[vehicle_id]["arrival_time"] - \
                    self.vehicle_departure_arrival_time_dict[vehicle_id]["departure_time"]
                if vehicle_id in self.vehicle_pending_departure_time_dict:
                    self.pending_travel_time_dict[vehicle_id] = \
                        self.vehicle_departure_arrival_time_dict[vehicle_id]["arrival_time"] - \
                        self.vehicle_pending_departure_time_dict[vehicle_id]["departure_time"]
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_pending_time_loss(self, pending_vehicles):

        for vehicle_id in pending_vehicles:
            time_loss = 1
            self.pending_time_loss_dict[vehicle_id] += time_loss

    def _update_time_loss(self, current_step_vehicles):

        for vehicle_id in current_step_vehicles:
            subscription_data = self.current_step_vehicle_subscription[vehicle_id]
            time_loss = sumo_traci_util.get_time_loss(subscription_data, self.execution_name)
            self.time_loss_dict[vehicle_id] += time_loss

    def get_current_time(self):
        traci_connection = sumo_traci_util.get_traci_connection(self.execution_name)
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

        if self.mode == 'test' or self.mode == 'replay':

            trip_time_loss_dict = {}
            for key in self.travel_time_dict.keys():
                trip_time_loss_dict[key] = self.time_loss_dict[key]

            time_losses = trip_time_loss_dict.values()
            average_time_loss = sum(time_losses)/len(time_losses) if len(time_losses) != 0 else 0

            # trip_time_loss_with_pending_dict = copy.deepcopy(trip_time_loss_dict)
            # for key, value in self.pending_time_loss_dict.items():
            #     if key in trip_time_loss_dict:
            #         trip_time_loss_with_pending_dict[key] += value
            #
            # time_losses = trip_time_loss_with_pending_dict.values()
            # average_time_loss_with_pending = sum(time_losses) / len(time_losses) if len(time_losses) != 0 else 0

            travel_times = self.travel_time_dict.values()
            average_travel_time = sum(travel_times)/len(travel_times) if len(travel_times) != 0 else 0

            # travel_time_with_pending_dict = copy.deepcopy(self.travel_time_dict)
            # for key, value in self.pending_travel_time_dict.items():
            #     travel_time_with_pending_dict[key] = self.pending_travel_time_dict[key]
            #
            # travel_times = travel_time_with_pending_dict.values()
            # average_travel_time_with_pending = sum(travel_times) / len(travel_times) if len(travel_times) != 0 else 0

            throughput = self.total_arrived_vehicles

            time_loss = sumo_traci_util.get_network_time_loss(
                self.current_step_vehicle_subscription,
                self.execution_name)
            time_loss += self.total_pending_vehicles * 1

            self.total_time_loss += time_loss
            total_vehicles = self.total_departed_vehicles + self.total_pending_vehicles

            consolidated_time_loss_per_driver = self.total_time_loss / total_vehicles if total_vehicles != 0 else 0
            instant_time_loss_per_driver = time_loss / (self.total_running_vehicles + self.total_pending_vehicles) \
                if (self.total_running_vehicles + self.total_pending_vehicles) != 0 else 0
            extra = {
                "average_time_loss": average_time_loss,
                "average_travel_time": average_travel_time,
                "throughput": throughput,
                "consolidated_time_loss_per_driver": consolidated_time_loss_per_driver,
                "instant_time_loss_per_driver": instant_time_loss_per_driver
            }

            self.network_logs.append({
                "time": current_time,
                "extra": extra})

        for intersection_index, intersection in enumerate(self.intersections):

            if self.mode == 'replay':

                traffic_light = sumo_traci_util.get_traffic_light_state(
                    intersection.traffic_light_id,
                    self.execution_name)

                time_loss = sumo_traci_util.get_network_time_loss(
                    intersection.current_step_vehicle_subscription,
                    self.execution_name)

                relative_occupancy_by_movement = {
                    movement: intersection.get_density(lanes)
                    for movement, lanes in intersection.movement_to_entering_lane.items()
                }

                relative_mean_speed_by_movement = {
                    movement: intersection.get_mean_relative_speed(lanes)
                    for movement, lanes in intersection.movement_to_entering_lane.items()
                }

                absolute_number_of_vehicles_by_movement = {
                    movement: intersection.get_number_of_vehicles(lanes)
                    for movement, lanes in intersection.movement_to_entering_lane.items()
                }

                extra = {
                    "traffic_light": traffic_light,
                    "time_loss": time_loss,
                    "relative_occupancy": relative_occupancy_by_movement,
                    "relative_mean_speed": relative_mean_speed_by_movement,
                    "absolute_number_of_vehicles": absolute_number_of_vehicles_by_movement
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
                    "action": action[intersection_index]
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
                f = open(ROOT_DIR + '/' + path_to_log_file, "wb")
                pickle.dump(self.intersection_logs[intersection_index], f)
                f.close()

                if self.mode == 'replay':
                    path_to_detailed_log_file = os.path.join(
                        self.path_to_log, "inter_{0}_detailed.pkl".format(intersection.id))
                    f = open(ROOT_DIR + '/' + path_to_detailed_log_file, "wb")
                    pickle.dump(self.intersection_logs[intersection_index], f)
                    f.close()

                if self.mode == 'test':
                    path_to_actions_log_file = os.path.join(
                        self.path_to_log, "inter_{0}_actions.pkl".format(intersection.id))
                    f = open(ROOT_DIR + '/' + path_to_actions_log_file, "wb")
                    pickle.dump(self.action_logs[intersection_index], f)
                    f.close()

            self.intersection_logs = [[] for _ in range(len(self.intersections))]
            self.action_logs = [[] for _ in range(len(self.intersections))]
            self.network_logs = []

    def save_state(self, name=None):

        if not os.path.isdir('/dev/shm' + '/' + self.environment_state_path):
            os.makedirs('/dev/shm' + '/' + self.environment_state_path)

        if name is None:
            state_name = self.execution_name + '_' + 'save_state' + '_' + str(self.get_current_time()) + '.sbx'
        else:
            state_name = name

        filepath = os.path.join('/dev/shm', self.environment_state_path, state_name)

        traci_connection = sumo_traci_util.get_traci_connection(self.execution_name)
        traci_connection.simulation.saveState(filepath)

        return filepath

    def check_for_active_action_time_actions(self, action):
        
        for intersection_index, intersection in enumerate(self.intersections):

            if action[intersection_index] == 'no_op':
                continue
            
            action_time_action = intersection.select_active_action_time_action()
            
            if action_time_action != -1:
                action[intersection_index] = action_time_action

        return action
    
    def check_for_time_restricted_actions(self, action, waiting_time_restriction=120):

        for intersection_index, intersection in enumerate(self.intersections):

            if action[intersection_index] == 'no_op':
                continue

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

            # action = [self.action_logs[intersection_index][int(instant_time)]['action']
            #           for intersection_index, _ in enumerate(self.intersections)]

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
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"]
            )

        # run one step

        traci_connection = sumo_traci_util.get_traci_connection(self.execution_name)
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            traci_connection.simulationStep()

        # get new measurements
        self.update_current_measurements()
        for intersection in self.intersections:
            intersection.update_current_measurements()

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

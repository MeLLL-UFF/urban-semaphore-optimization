import os
import sys
import copy
import shutil
import pickle

import numpy as np
import pandas as pd
from lxml import etree
import traci
import traci.constants as tc

from utils import sumo_util, sumo_traci_util, summary_util

from algorithm.sumo_based.definitions import ROOT_DIR


class SumoBase:

    # add more variables here if you need more measurements
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

    # add more variables here if you need more measurements
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
        "VAR_DECEL"
    ]

    def __init__(self, net_file, scenario, _type, traffic_level_configuration):

        self.dic_traffic_env_conf = {
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "lane_num_vehicle",
            ],
            'DIC_REWARD_INFO': {
                "sum_num_vehicle_been_stopped_thres1": -0.25
            }
        }

        self.path_to_log = os.path.join(
            'records', _type, scenario + '__' + _type + '__' + '_'.join(traffic_level_configuration), 'test')

        self.list_inter_log = []

        self._type = _type

        parser = etree.XMLParser(remove_blank_text=True)

        self.net_file_xml = etree.parse(net_file, parser)

        intersection_id = sumo_util.get_intersections_ids(self.net_file_xml)[0]
        self.node_light = intersection_id

        self.list_vehicle_variables_to_sub = self.LIST_VEHICLE_VARIABLES_TO_SUB

        # ===== sumo intersection settings =====

        movements, movement_to_connection = \
            sumo_util.detect_movements(self.net_file_xml, False)

        self.movements = movements
        self.movement_to_connection = movement_to_connection

        self.incoming_edges, self.outgoing_edges = sumo_util.get_intersection_edge_ids(self.net_file_xml)
        self.edges = [] + self.incoming_edges + self.outgoing_edges

        self.list_entering_lanes = [connection.get('from') + '_' + connection.get('fromLane')
                                    for _, connection in movement_to_connection.items()
                                    if connection.get('dir') != 'r']
        self.list_exiting_lanes = [connection.get('to') + '_' + connection.get('toLane')
                                   for _, connection in movement_to_connection.items()
                                   if connection.get('dir') != 'r']
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        # -1: all yellow, -2: all red, -3: none
        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.current_phase_duration = -1

        self.dic_lane_sub_current_step = None
        self.dic_lane_sub_previous_step = None
        self.dic_vehicle_sub_current_step = None
        self.dic_vehicle_sub_previous_step = None
        self.dic_lane_vehicle_sub_current_step = None
        self.dic_lane_vehicle_sub_previous_step = None
        self.list_vehicles_current_step = []
        self.list_vehicles_previous_step = []

        self.dic_vehicle_min_speed = {}  # this second
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second

    def run(self, net_file, route_file, output_file):

        self._start_traci(net_file, route_file, output_file)
        self._run_simulation(max_time=3600)
        traci.close()
        self._bulk_log()
        self._single_experiment_summary()
        self._consolidate_output_file(output_file)

    def visualize_policy_behavior(self, scenario, _type, traffic_level_configuration, experiment='last',
                                  _round='best_time_loss'):
        pass

    def summary(self, experiment, plots='all', _round=None):
        pass


    def _start_traci(self, net_file, route_file, output_file):

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([
            sumo_util.get_sumo_binary(),
            '-n', net_file,
            '-r', route_file,
            '--log', output_file,
            '--duration-log.statistics', str(True),
            '--time-to-teleport', str(-1),
            '--collision.stoptime', str(10),
            '--collision.mingap-factor', str(0),
            '--collision.action', 'warn',
            '--collision.check-junctions', str(True)
        ])

        # start subscription
        for lane in self.list_lanes:
            traci.lane.subscribe(lane, [getattr(tc, var) for var in self.LIST_LANE_VARIABLES_TO_SUB])

    def _run_simulation(self, max_time=float('inf')):
        while traci.simulation.getMinExpectedNumber() > 0 and \
                traci.simulation.getTime() < max_time:

            self.update_previous_measurements()

            traci.simulationStep()

            self.update_current_measurements()

            self._collect_simulation_data()

    def _collect_simulation_data(self):
        
        current_time = sumo_traci_util.get_current_time()
        state = self.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
        action = -1
        reward = self.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"])

        traffic_light = sumo_traci_util.get_traffic_light_state(self.node_light)
        time_loss = sumo_traci_util.get_time_loss(self.dic_vehicle_sub_current_step)
        relative_occupancy = sumo_traci_util.get_lane_relative_occupancy(
            self.dic_lane_sub_current_step,
            self.dic_lane_vehicle_sub_current_step,
            self.dic_vehicle_sub_current_step,
            self.edges
        )
        relative_mean_speed = sumo_traci_util.get_relative_mean_speed(
            self.dic_lane_sub_current_step,
            self.edges
        )
        absolute_number_of_cars = sumo_traci_util.get_absolute_number_of_cars(
            self.dic_lane_sub_current_step,
            self.edges
        )

        extra = {
            "traffic_light": traffic_light,
            "time_loss": time_loss,
            "relative_occupancy": relative_occupancy,
            "relative_mean_speed": relative_mean_speed,
            "absolute_number_of_cars": absolute_number_of_cars
        }

        self.list_inter_log.append({"time": current_time,
                                    "state": state,
                                    "action": action,
                                    "reward": reward,
                                    "extra": extra})

    def _bulk_log(self):

        path_to_log_file = os.path.join(self.path_to_log, "inter_0.pkl")

        if not os.path.exists(ROOT_DIR + '/' + self.path_to_log):
            os.makedirs(ROOT_DIR + '/' + self.path_to_log)

        f = open(ROOT_DIR + '/' + path_to_log_file, "wb")
        pickle.dump(self.list_inter_log, f)
        f.close()

        detailed_copy = os.path.join(self.path_to_log, "inter_0_detailed.pkl")
        shutil.copy(ROOT_DIR + '/' + path_to_log_file, ROOT_DIR + '/' + detailed_copy)

    def _single_experiment_summary(self):

        name_base = self._type + '-' + 'test'

        connections = sumo_util.get_connections(self.net_file_xml)
        entering_lanes = [connection.get('from') + '_' + connection.get('fromLane') for connection in connections]
        lane_to_traffic_light_index_mapping = sumo_util.get_lane_traffic_light_controller(self.net_file_xml,
                                                                                          entering_lanes)

        reward_each_step = []
        time_loss_each_step = []
        traffic_light_each_step = []
        relative_occupancy_each_step = []
        relative_mean_speed_each_step = []
        absolute_number_of_cars_each_step = []

        # summary items (queue_length) from pickle
        f = open(os.path.join(ROOT_DIR, self.path_to_log, "inter_0_detailed.pkl"), "rb")
        samples = pickle.load(f)
        for sample in samples:
            reward_each_step.append(sample['reward'])
            time_loss_each_step.append(sample['extra']['time_loss'])
            traffic_light_each_step.append(sample['extra']['traffic_light'])
            relative_occupancy_each_step.append(sample['extra']['relative_occupancy'])
            relative_mean_speed_each_step.append(sample['extra']['relative_mean_speed'])
            absolute_number_of_cars_each_step.append(sample['extra']['absolute_number_of_cars'])
        f.close()

        save_path = ROOT_DIR + '/' + self.path_to_log

        serializable_movement_to_connection = dict(copy.deepcopy(self.movement_to_connection))
        for movement in serializable_movement_to_connection.keys():
            serializable_movement_to_connection[movement] = dict(serializable_movement_to_connection[movement].attrib)

        summary_util.consolidate_time_loss(time_loss_each_step, save_path, name_base)
        summary_util.consolidate_reward(reward_each_step, save_path, name_base)

        summary_util.consolidate_occupancy_and_speed_inflow_outflow(
            relative_occupancy_each_step,
            relative_mean_speed_each_step,
            self.movements,
            serializable_movement_to_connection,
            save_path,
            name_base)

        summary_util.consolidate_phase_and_demand(
            absolute_number_of_cars_each_step,
            traffic_light_each_step,
            self.movements,
            serializable_movement_to_connection,
            lane_to_traffic_light_index_mapping,
            save_path,
            name_base)


    def _consolidate_output_file(self, output_file):

        duration = sumo_util.get_average_duration_statistic(output_file)

        filename = output_file.rsplit('.', 2)[0]
        duration_df = pd.DataFrame()
        duration_df.loc[0, 'test'] = duration
        duration_df.to_csv(filename + '_' + 'result' + '.csv')

    def _plot_consolidate_output(self, output_folder, experiment_name, duration_list, 
            scenario, traffic_level_configuration):
        pass



    def update_previous_measurements(self):

        self.previous_phase_index = self.current_phase_index
        self.dic_lane_sub_previous_step = self.dic_lane_sub_current_step
        self.dic_vehicle_sub_previous_step = self.dic_vehicle_sub_current_step
        self.dic_lane_vehicle_sub_previous_step = self.dic_lane_vehicle_sub_current_step
        self.list_vehicles_previous_step = self.list_vehicles_current_step

    def update_current_measurements(self):
        # need change, debug in seeing format

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        # ====== lane level observations =======

        self.dic_lane_sub_current_step = {lane: traci.lane.getSubscriptionResults(lane) for lane in self.list_lanes}

        # ====== vehicle level observations =======

        # get vehicle list
        self.list_vehicles_current_step = traci.vehicle.getIDList()
        list_vehicles_new_arrive = list(set(self.list_vehicles_current_step) - set(self.list_vehicles_previous_step))
        list_vehicles_new_left = list(set(self.list_vehicles_previous_step) - set(self.list_vehicles_current_step))
        list_vehicles_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicles_new_left_entering_lane = []
        for l in list_vehicles_new_left_entering_lane_by_lane:
            list_vehicles_new_left_entering_lane += l

        # update subscriptions
        for vehicle in list_vehicles_new_arrive:
            traci.vehicle.subscribe(vehicle, [getattr(tc, var) for var in self.list_vehicle_variables_to_sub])

        # vehicle level observations
        self.dic_vehicle_sub_current_step = {vehicle: traci.vehicle.getSubscriptionResults(vehicle)
                                             for vehicle in self.list_vehicles_current_step}
        self.dic_lane_vehicle_sub_current_step = {}
        for vehicle, values in self.dic_vehicle_sub_current_step.items():
            lane = values[tc.VAR_LANE_ID]
            if lane in self.dic_lane_vehicle_sub_current_step:
                self.dic_lane_vehicle_sub_current_step[lane][vehicle] = self.dic_vehicle_sub_current_step[vehicle]
            else:
                self.dic_lane_vehicle_sub_current_step[lane] = {vehicle: self.dic_vehicle_sub_current_step[vehicle]}

        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicles_new_arrive)
        self._update_left_time(list_vehicles_new_left_entering_lane)

        # update vehicle minimum speed in history
        self._update_vehicle_min_speed()

        # update feature
        self._update_feature()

    def _update_leave_entering_approach_vehicle(self):

        list_entering_lane_vehicle_left = []

        # update vehicles leaving entering lane
        if self.dic_lane_sub_previous_step is None:
            for _ in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            for lane in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append(
                    list(
                        set(self.dic_lane_sub_previous_step[lane]
                            [tc.LAST_STEP_VEHICLE_ID_LIST]) -
                        set(self.dic_lane_sub_current_step[lane]
                            [tc.LAST_STEP_VEHICLE_ID_LIST])
                    )
                )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicles_arrive):

        ts = sumo_traci_util.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicles_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                print("vehicle already exists!")
                sys.exit(-1)

    def _update_left_time(self, list_vehicles_left):

        ts = sumo_traci_util.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicles_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_vehicle_min_speed(self):
        '''
        record the minimum speed of one vehicle so far
        :return:
        '''
        dic_result = {}
        for vec_id, vec_var in self.dic_vehicle_sub_current_step.items():
            speed = vec_var[tc.VAR_SPEED]
            if vec_id in self.dic_vehicle_min_speed:  # this vehicle appeared in previous time stamps:
                dic_result[vec_id] = min(speed, self.dic_vehicle_min_speed[vec_id])
            else:
                dic_result[vec_id] = speed
        self.dic_vehicle_min_speed = dic_result

    def _update_feature(self):

        dic_feature = dict()

        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["vehicle_position_img"] = None  # self._get_lane_vehicle_position(self.list_entering_lanes)
        dic_feature["vehicle_speed_img"] = None  # self._get_lane_vehicle_speed(self.list_entering_lanes)
        dic_feature["vehicle_acceleration_img"] = None
        dic_feature["vehicle_waiting_time_img"] = None
        # self._get_lane_vehicle_accumulated_waiting_time(self.list_entering_lanes)

        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres01"] = \
            self._get_lane_num_vehicle_been_stopped(0.1, self.list_entering_lanes)
        dic_feature["lane_num_vehicle_been_stopped_thres1"] = \
            self._get_lane_num_vehicle_been_stopped(1, self.list_entering_lanes)
        dic_feature["lane_queue_length"] = self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_vehicle_left"] = None
        dic_feature["lane_sum_duration_vehicle_left"] = None
        dic_feature["lane_sum_waiting_time"] = self._get_lane_sum_waiting_time(self.list_entering_lanes)

        dic_feature["terminal"] = None

        self.dic_feature = dic_feature

    # ================= calculate features from current observations ======================

    def _get_lane_queue_length(self, list_lanes):
        '''
        queue length for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
                for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][tc.LAST_STEP_VEHICLE_NUMBER]
                for lane in list_lanes]

    def _get_lane_sum_waiting_time(self, list_lanes):
        '''
        waiting time for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][tc.VAR_WAITING_TIME]
                for lane in list_lanes]

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):

        list_num_of_vec_ever_stopped = []
        for lane in list_lanes:
            cnt_vec = 0
            list_vec_id = self.dic_lane_sub_current_step[lane][tc.LAST_STEP_VEHICLE_ID_LIST]
            for vec in list_vec_id:
                if self.dic_vehicle_min_speed[vec] < thres:
                    cnt_vec += 1
            list_num_of_vec_ever_stopped.append(cnt_vec)

        return list_num_of_vec_ever_stopped

    # ================= get functions from outside ======================

    def get_state(self, list_state_features):
        dic_state = {state_feature_name: self.dic_feature[state_feature_name]
                     for state_feature_name in list_state_features}
        return dic_state

    def get_reward(self, dic_reward_info):

        dic_reward = dict()
        dic_reward["flickering"] = None
        dic_reward["sum_lane_queue_length"] = None
        dic_reward["sum_lane_wait_time"] = None
        dic_reward["sum_lane_num_vehicle_left"] = None
        dic_reward["sum_duration_vehicle_left"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres01"] = None
        dic_reward["sum_num_vehicle_been_stopped_thres1"] = \
            np.sum(self.dic_feature["lane_num_vehicle_been_stopped_thres1"])

        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]

        return reward

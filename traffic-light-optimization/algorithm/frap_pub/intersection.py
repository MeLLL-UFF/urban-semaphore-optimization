import os
import sys
import json

import numpy as np
import traci
import traci.constants as tc

from utils import sumo_traci_util, sumo_util, xml_util

from algorithm.frap_pub.definitions import ROOT_DIR


class Intersection:

    def __init__(self, intersection_index, dic_traffic_env_conf, dic_path, execution_name,
                 external_configurations=None):

        if external_configurations is None:
            external_configurations = {}

        self.intersection_index = intersection_index
        self.id = dic_traffic_env_conf['INTERSECTION_ID'][intersection_index]
        self.traffic_light_id = dic_traffic_env_conf['TRAFFIC_LIGHT_ID'][intersection_index]
        self.execution_name = execution_name

        net_file = os.path.join(ROOT_DIR, dic_path["PATH_TO_WORK_DIRECTORY"], dic_traffic_env_conf['NET_FILE'])
        self.net_file_xml = xml_util.get_xml(net_file)
        traffic_light_file = os.path.join(ROOT_DIR, dic_path["PATH_TO_WORK_DIRECTORY"],
                                          dic_traffic_env_conf['TRAFFIC_LIGHT_FILE'])
        self.traffic_light_file_xml = xml_util.get_xml(traffic_light_file)

        multi_intersection_traffic_light_file = os.path.join(
            ROOT_DIR, dic_path["PATH_TO_DATA"],
            dic_traffic_env_conf['MULTI_INTERSECTION_TRAFFIC_LIGHT_FILE'])

        if os.path.isfile(multi_intersection_traffic_light_file):
            with open(multi_intersection_traffic_light_file, 'r') as handle:
                multi_intersection_traffic_light_configuration = json.load(handle)
        else:
            multi_intersection_traffic_light_configuration = {}

        self.multi_intersection_traffic_light_configuration = multi_intersection_traffic_light_configuration

        self.movements = dic_traffic_env_conf['MOVEMENT'][intersection_index]
        self.phases = dic_traffic_env_conf['PHASE'][intersection_index]
        self.link_states = dic_traffic_env_conf['LINK_STATES'][intersection_index]
        self.movement_to_connection = dic_traffic_env_conf['MOVEMENT_TO_CONNECTION'][intersection_index]
        self.movement_to_traffic_light_link_index = \
            dic_traffic_env_conf['MOVEMENT_TO_TRAFFIC_LIGHT_LINK_INDEX'][intersection_index]

        traffic_light_link_indices = set([])
        for _, indices in self.movement_to_traffic_light_link_index.items():
            traffic_light_link_indices.update(indices)
        self.traffic_light_link_indices = list(traffic_light_link_indices)

        self.unique_movements = dic_traffic_env_conf['UNIQUE_MOVEMENT']
        self.unique_phases = dic_traffic_env_conf['UNIQUE_PHASE']

        self.is_right_on_red = dic_traffic_env_conf['IS_RIGHT_ON_RED']

        self.min_action_time = dic_traffic_env_conf['MIN_ACTION_TIME']
        self.has_per_second_decision = dic_traffic_env_conf.get('PER_SECOND_DECISION', False)

        self.dic_path = dic_path

        self.entering_edges, self.exiting_edges = sumo_util.get_intersection_edge_ids(
            self.net_file_xml,
            intersection_ids=self.id,
            multi_intersection_traffic_light_configuration=self.multi_intersection_traffic_light_configuration
        )
        self.edges = self.entering_edges + self.exiting_edges
        self.edge_ids = [edge.get('id') for edge in self.edges]

        self.internal_edges = sumo_util.get_internal_edges(
            self.net_file_xml, self.id, self.multi_intersection_traffic_light_configuration)
        self.internal_edge_ids = [edge.get('id') for edge in self.internal_edges]

        self.intermediary_edges = sumo_util.get_intermediary_edges(
            self.net_file_xml, self.id, self.multi_intersection_traffic_light_configuration)
        self.intermediary_edge_ids = [edge.get('id') for edge in self.intermediary_edges]

        self.all_edge_ids = self.edge_ids + self.internal_edge_ids + self.intermediary_edge_ids

        self.entering_lanes = [lane for edge in self.entering_edges for lane in list(edge)]
        self.entering_lane_ids = [lane.get('id') for lane in self.entering_lanes]
        self.exiting_lanes = [lane for edge in self.exiting_edges for lane in list(edge)]
        self.exiting_lane_ids = [lane.get('id') for lane in self.exiting_lanes]

        self.lanes = self.entering_lanes + self.exiting_lanes
        self.lane_ids = [lane.get('id') for lane in self.lanes]

        self.internal_lanes = sumo_util.get_internal_lanes(
            self.net_file_xml, self.id, self.multi_intersection_traffic_light_configuration)
        self.internal_lane_ids = [lane.get('id') for lane in self.internal_lanes]

        self.intermediary_lanes = sumo_util.get_intermediary_lanes(
            self.net_file_xml, self.id, self.multi_intersection_traffic_light_configuration)
        self.intermediary_lane_ids = [lane.get('id') for lane in self.intermediary_lanes]

        self.all_lane_ids = \
            self.entering_lane_ids + self.exiting_lane_ids + self.internal_lane_ids + self.intermediary_lane_ids

        self.controlled_entering_lane_ids = []
        self.controlled_exiting_lane_ids = []

        self.uncontrolled_movements = []

        self.movement_to_entering_lane = {}
        self.movement_to_exiting_lane = {}

        self.controlled_entering_lane_ids = set(self.controlled_entering_lane_ids)
        self.controlled_exiting_lane_ids = set(self.controlled_exiting_lane_ids)
        for movement, connections in self.movement_to_connection.items():

            if movement not in self.movements:
                self.uncontrolled_movements.append(movement)
                continue

            self.movement_to_entering_lane[movement] = set([])
            self.movement_to_exiting_lane[movement] = set([])

            for connection in connections:

                from_lane = connection.get('from') + '_' + connection.get('fromLane')
                if from_lane not in self.intermediary_lane_ids:
                    self.controlled_entering_lane_ids.add(from_lane)
                    self.movement_to_entering_lane[movement].add(from_lane)

                to_lane = connection.get('to') + '_' + connection.get('toLane')
                if to_lane not in self.intermediary_lane_ids:
                    self.controlled_exiting_lane_ids.add(to_lane)
                    self.movement_to_exiting_lane[movement].add(to_lane)

            self.movement_to_entering_lane[movement] = list(self.movement_to_entering_lane[movement])
            self.movement_to_exiting_lane[movement] = list(self.movement_to_exiting_lane[movement])

        self.controlled_entering_lane_ids = list(self.controlled_entering_lane_ids)
        self.controlled_exiting_lane_ids = list(self.controlled_exiting_lane_ids)

        self.maximum_detector_length = dic_traffic_env_conf['DETECTOR_EXTENSION']
        self.subscription_extension = self.get_subscription_extension(self.maximum_detector_length)

        entering_lane_extension_ids = np.unique([
            lane_tuple[0]
            for lane_tuples in self.subscription_extension['entering_lanes'].values()
            for lane_tuple in lane_tuples
            if lane_tuple[0] not in self.entering_lane_ids
        ]).tolist()

        exiting_lane_extension_ids = np.unique([
            lane_tuple[0]
            for lane_tuples in self.subscription_extension['exiting_lanes'].values()
            for lane_tuple in lane_tuples
            if lane_tuple[0] not in self.exiting_lane_ids
        ]).tolist()

        self.all_lane_ids_subscription = \
            self.all_lane_ids + entering_lane_extension_ids + exiting_lane_extension_ids

        self.phase_traffic_lights = self.get_phase_traffic_lights()

        self.default_yellow_time = dic_traffic_env_conf['DEFAULT_YELLOW_TIME']
        self.movement_to_yellow_time = dic_traffic_env_conf['MOVEMENT_TO_YELLOW_TIME'][intersection_index]

        self.current_yellow_time = 0
        self.yellow_phase_index = -1
        self.all_red_phase_index = -2

        # grid settings
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane//self.length_grid)

        # initialization

        # -1: all yellow, -2: all red, -3: none
        self.current_phase_index = 0
        self.previous_phase_index = 0
        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.current_min_action_duration = -1
        self.all_red_flag = False
        self.yellow_flag = False
        self.flicker = 0

        self.current_step_lane_subscription = None
        self.previous_step_lane_subscription = None
        self.current_step_vehicle_subscription = None
        self.previous_step_vehicle_subscription = None
        self.current_step_lane_area_detector_vehicle_ids = None
        self.previous_step_lane_area_detector_vehicle_ids = None
        self.current_step_vehicles = []
        self.previous_step_vehicles = []

        self.detector_additional_info = None

        self.vehicle_min_speed_dict = {}  # this second

        self.detector_subscription_function = self.build_detector_subscription_function()

        self.feature_dict = {}  # this second

        self.state_feature_list = dic_traffic_env_conf["STATE_FEATURE_LIST"]

        self.feature_dict_function = {
            'current_phase': lambda: [self.current_phase_index],
            'time_this_phase': lambda: [self.current_phase_duration],
            'vehicle_position_img': lambda: None,
            'vehicle_speed_img': lambda: None,
            'vehicle_acceleration_img': lambda: None,
            'vehicle_waiting_time_img': lambda: None,
            'movement_number_of_vehicles': lambda:
                self.pad_movements(self.get_number_of_vehicles, self.movement_to_entering_lane),
            'movement_number_of_vehicles_been_stopped_threshold_01': lambda:
                self.pad_movements(
                    self.get_number_of_vehicles_been_stopped, self.movement_to_entering_lane, threshold=0.1),
            'movement_number_of_vehicles_been_stopped_threshold_1': lambda:
                self.pad_movements(
                    self.get_number_of_vehicles_been_stopped, self.movement_to_entering_lane, threshold=1),
            'movement_queue_length': lambda:
                self.pad_movements(self.get_queue_length, self.movement_to_entering_lane),
            'movement_number_of_vehicles_left': lambda: None,
            'movement_sum_duration_vehicles_left': lambda: None,
            'movement_sum_waiting_time': lambda:
                self.pad_movements(self.get_waiting_time, self.movement_to_entering_lane),
            'terminal': lambda: None,
            'movement_pressure_presslight': lambda:
                np.array(self.pad_movements(self.get_density, self.movement_to_entering_lane)) -
                np.array(self.pad_movements(self.get_density, self.movement_to_exiting_lane)),
            'movement_pressure_mplight': lambda:
                np.array(self.pad_movements(self.get_number_of_vehicles, self.movement_to_entering_lane)) -
                np.array(self.pad_movements(self.get_number_of_vehicles, self.movement_to_exiting_lane)),
            'movement_pressure_time_loss': lambda:
                np.array(self.pad_movements(self.get_time_loss, self.movement_to_entering_lane)) -
                np.array(self.pad_movements(self.get_time_loss, self.movement_to_exiting_lane)),
            'movement_sum_time_loss': lambda:
                self.pad_movements(self.get_time_loss, self.movement_to_entering_lane)
        }

        self.reward_dict_function = {
            'flickering': lambda: None,
            'sum_queue_length': lambda: -np.sum(self.get_queue_length(self.controlled_entering_lane_ids)),
            'avg_movement_queue_length': lambda: -np.average(self.get_feature('movement_queue_length')),
            'sum_waiting_time': lambda: -np.sum(self.get_waiting_time(self.controlled_entering_lane_ids)),
            'sum_num_vehicle_left': lambda: None,
            'sum_duration_vehicles_left': lambda: None,
            'sum_number_of_vehicles_been_stopped_threshold_01':
                lambda: -np.sum(self.get_number_of_vehicles_been_stopped(self.controlled_entering_lane_ids, 0.1)),
            'sum_number_of_vehicles_been_stopped_threshold_1':
                lambda: -np.sum(self.get_number_of_vehicles_been_stopped(self.controlled_entering_lane_ids, 1)),
            'pressure_presslight': lambda:
                -np.abs(np.sum(self.get_feature('movement_pressure_presslight'))),
            'pressure_mplight': lambda:
                -(np.sum(self.get_queue_length(self.controlled_entering_lane_ids)) -
                  np.sum(self.get_queue_length(self.controlled_exiting_lane_ids))),
            'pressure_time_loss': lambda:
                -(np.sum(self.get_time_loss(self.controlled_entering_lane_ids)) -
                  np.sum(self.get_time_loss(self.controlled_exiting_lane_ids))),
            'time_loss': lambda:
                -np.sum(self.get_time_loss(self.controlled_entering_lane_ids))
        }

    def reset(self):

        self.detector_additional_info = {}

        # get vehicle list
        entering_lane_tuples_list = list(self.subscription_extension['entering_lanes'].items())
        zipped_entering_lane_tuples_list = \
            list(zip(entering_lane_tuples_list, ['entering'] * len(entering_lane_tuples_list)))

        exiting_lane_tuples_list = list(self.subscription_extension['exiting_lanes'].items())
        zipped_exiting_lane_tuples_list = \
            list(zip(exiting_lane_tuples_list, ['exiting'] * len(exiting_lane_tuples_list)))

        for (detector_id, lane_tuples), lane_type in zipped_entering_lane_tuples_list + zipped_exiting_lane_tuples_list:

            self.detector_additional_info[detector_id] = {}

            for lane_id, lane_length, accumulated_detector_length in lane_tuples:

                self.detector_additional_info[detector_id][lane_id] = {}

                start_position = 0
                end_position = lane_length

                is_partial_detector = False
                if accumulated_detector_length > self.maximum_detector_length:
                    detector_partial_length = lane_length - (accumulated_detector_length - self.maximum_detector_length)
                    if lane_type == 'entering':
                        start_position = lane_length - detector_partial_length
                    elif lane_type == 'exiting':
                        end_position = detector_partial_length

                    is_partial_detector = True

                self.detector_additional_info[detector_id][lane_id][
                    sumo_traci_util.VAR_LANE_START_POSITION] = start_position
                self.detector_additional_info[detector_id][lane_id][
                    sumo_traci_util.VAR_LANE_END_POSITION] = end_position
                self.detector_additional_info[detector_id][lane_id][
                    sumo_traci_util.VAR_IS_PARTIAL_DETECTOR] = is_partial_detector

    def update_previous_measurements(self):

        self.previous_phase_index = self.current_phase_index
        self.previous_step_lane_subscription = self.current_step_lane_subscription
        self.previous_step_vehicle_subscription = self.current_step_vehicle_subscription
        self.previous_step_lane_area_detector_vehicle_ids = \
            self.current_step_lane_area_detector_vehicle_ids
        self.previous_step_vehicles = self.current_step_vehicles

    def update_current_measurements(self):
        # need change, debug in seeing format
        
        traci_connection = traci.getConnection(self.execution_name)

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.current_min_action_duration += 1

        self.current_step_lane_subscription = {lane_id: traci_connection.lane.getSubscriptionResults(lane_id)
                                               for lane_id in self.all_lane_ids_subscription}

        # ====== vehicle level observations =======

        self.current_step_vehicle_subscription = {}
        current_step_vehicles = self._update_vehicle_subscription()

        self.current_step_lane_area_detector_vehicle_ids = {}
        self._update_detector_vehicles()

        self.current_step_vehicles = current_step_vehicles
        recently_arrived_vehicles = list(set(self.current_step_vehicles) - set(self.previous_step_vehicles))
        recently_left_vehicles = list(set(self.previous_step_vehicles) - set(self.current_step_vehicles))

        # update vehicle minimum speed in history
        self._update_vehicle_min_speed()

        # update feature
        self._update_feature()

    def set_signal(self, action, action_pattern):

        if self.yellow_flag:
            # in yellow phase
            self.flicker = 0

            if self.current_phase_duration >= self.current_yellow_time:  # yellow time reached

                current_traffic_light = sumo_traci_util.get_traffic_light_state(
                    self.traffic_light_id, self.execution_name)

                self.current_phase_index = self.next_phase_to_set_index
                phase = self.phases[self.current_phase_index]
                next_traffic_light = self.phase_traffic_lights[phase]

                for index in range(0, len(current_traffic_light)):

                    if index not in self.traffic_light_link_indices:
                        continue

                    current_traffic_light = \
                        current_traffic_light[:index] + \
                        next_traffic_light[index] + \
                        current_traffic_light[index + 1:]

                sumo_traci_util.set_traffic_light_state(
                    self.traffic_light_id, current_traffic_light, self.execution_name)
                self.yellow_flag = False
            else:
                pass
        else:

            if self.next_phase_to_set_index is None or self.current_min_action_duration >= self.min_action_time:

                if action == 'no_op':
                    return

                # determine phase
                if action_pattern == "switch":  # switch by order
                    if action == 0:  # keep the phase
                        self.next_phase_to_set_index = self.current_phase_index
                    elif action == 1:  # change to the next phase
                        self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.phases)
                    else:
                        sys.exit("action not recognized\n action must be 0 or 1")

                elif action_pattern == "set":  # set to certain phase
                    phase = self.unique_phases[action]
                    phase_index = self.phases.index(phase)
                    self.next_phase_to_set_index = phase_index

                # set phase
                if self.current_phase_index == self.next_phase_to_set_index:  # the light phase keeps unchanged
                    if not self.has_per_second_decision:
                        self.current_min_action_duration = 0
                else:  # the light phase needs to change
                    # change to yellow first, and activate the counter and flag
                    current_traffic_light = sumo_traci_util.get_traffic_light_state(
                        self.traffic_light_id, self.execution_name)

                    current_phase = self.phases[self.current_phase_index]
                    current_phase_movements = current_phase.split('_')
                    next_phase = self.phases[self.next_phase_to_set_index]
                    next_traffic_light = self.phase_traffic_lights[next_phase]

                    yellow_signal_indices = []
                    for index in range(0, len(current_traffic_light)):

                        if index not in self.traffic_light_link_indices:
                            continue

                        if (current_traffic_light[index] == 'g' or current_traffic_light[index] == 'G') and \
                                (next_traffic_light[index] != 'g' and next_traffic_light[index] != 'G'):

                            yellow_signal_indices.append(index)

                            current_traffic_light = \
                                current_traffic_light[:index] + \
                                'y' + \
                                current_traffic_light[index + 1:]

                    sumo_traci_util.set_traffic_light_state(
                        self.traffic_light_id, current_traffic_light, self.execution_name)

                    yellow_signal_movements = []
                    for movement in current_phase_movements:
                        traffic_light_indices = set(self.movement_to_traffic_light_link_index[movement])

                        if traffic_light_indices.intersection(yellow_signal_indices):
                            yellow_signal_movements.append(movement)

                    max_yellow_time = 0
                    for movement in yellow_signal_movements:
                        yellow_time = self.movement_to_yellow_time.get(movement, self.default_yellow_time)
                        max_yellow_time = max(max_yellow_time, yellow_time)
                    self.current_yellow_time = max_yellow_time

                    self.current_phase_index = self.yellow_phase_index
                    self.yellow_flag = True
                    self.flicker = 1

                    self.current_min_action_duration = 0

    # ================= update current step measurements ======================

    def _update_vehicle_min_speed(self):
        result_dict = {}
        for vehicle_id, vehicle in self.current_step_vehicle_subscription.items():
            speed = vehicle[tc.VAR_SPEED]
            if vehicle_id in self.vehicle_min_speed_dict:  # this vehicle appeared in previous time stamps:
                result_dict[vehicle_id] = min(speed, self.vehicle_min_speed_dict[vehicle_id])
            else:
                result_dict[vehicle_id] = speed
        self.vehicle_min_speed_dict = result_dict

    def _update_vehicle_subscription(self):

        traci_connection = traci.getConnection(self.execution_name)

        current_step_vehicles = []

        visited_lane_length = {}

        # get vehicle list
        entering_lane_tuples_list = list(self.subscription_extension['entering_lanes'].items())
        exiting_lane_tuples_list = list(self.subscription_extension['exiting_lanes'].items())

        zipped_entering_lane_tuples_list = \
            list(zip(entering_lane_tuples_list, ['entering'] * len(entering_lane_tuples_list)))
        zipped_exiting_lane_tuples_list = \
            list(zip(exiting_lane_tuples_list, ['exiting'] * len(exiting_lane_tuples_list)))

        for (detector_id, lane_tuples), lane_type in zipped_entering_lane_tuples_list + zipped_exiting_lane_tuples_list:
            for lane_id, lane_length, _ in lane_tuples:

                start_position = \
                    self.detector_additional_info[detector_id][lane_id][sumo_traci_util.VAR_LANE_START_POSITION]
                end_position = \
                    self.detector_additional_info[detector_id][lane_id][sumo_traci_util.VAR_LANE_END_POSITION]
                is_partial_detector = \
                    self.detector_additional_info[detector_id][lane_id][sumo_traci_util.VAR_IS_PARTIAL_DETECTOR]

                detector_partial_length = end_position - start_position
                previously_visited_length = visited_lane_length.get(lane_id, None)
                if previously_visited_length is not None:
                    if detector_partial_length > previously_visited_length:
                        if lane_type == 'entering':
                            end_position = lane_length - previously_visited_length
                        elif lane_type == 'exiting':
                            start_position = previously_visited_length
                    else:
                        continue

                lane_vehicle_ids = self.current_step_lane_subscription[lane_id][tc.LAST_STEP_VEHICLE_ID_LIST]

                if is_partial_detector:

                    for vehicle_id in lane_vehicle_ids:
                        vehicle_subscription = traci_connection.vehicle.getSubscriptionResults(vehicle_id)
                        lane_position = vehicle_subscription[tc.VAR_LANEPOSITION]

                        if start_position < lane_position <= end_position:
                            self.current_step_vehicle_subscription[vehicle_id] = vehicle_subscription
                            current_step_vehicles.append(vehicle_id)

                else:
                    for vehicle_id in lane_vehicle_ids:
                        vehicle_subscription = traci_connection.vehicle.getSubscriptionResults(vehicle_id)
                        self.current_step_vehicle_subscription[vehicle_id] = vehicle_subscription
                        current_step_vehicles.append(vehicle_id)

                visited_lane_length[lane_id] = detector_partial_length

        intersection_lane_tuples_list = self.subscription_extension['intersection_lanes'].values()

        for lane_tuples in intersection_lane_tuples_list:
            for lane_id, lane_length, _ in lane_tuples:

                previously_visited_length = visited_lane_length.get(lane_id, None)
                if previously_visited_length is not None:
                    continue

                lane_vehicle_ids = self.current_step_lane_subscription[lane_id][tc.LAST_STEP_VEHICLE_ID_LIST]

                for vehicle_id in lane_vehicle_ids:
                    vehicle_subscription = traci_connection.vehicle.getSubscriptionResults(vehicle_id)
                    self.current_step_vehicle_subscription[vehicle_id] = vehicle_subscription

                current_step_vehicles += lane_vehicle_ids

                visited_lane_length[lane_id] = lane_length

        return current_step_vehicles

    def _update_detector_vehicles(self):

        traci_connection = traci.getConnection(self.execution_name)

        entering_lane_tuples_list = list(self.subscription_extension['entering_lanes'].items())
        exiting_lane_tuples_list = list(self.subscription_extension['exiting_lanes'].items())

        for detector_id, lane_tuples in entering_lane_tuples_list + exiting_lane_tuples_list:

            self.current_step_lane_area_detector_vehicle_ids[detector_id] = []
            for lane_id, lane_length, _ in lane_tuples:

                lane_vehicle_ids = self.current_step_lane_subscription[lane_id][tc.LAST_STEP_VEHICLE_ID_LIST]

                is_partial_detector = \
                    self.detector_additional_info[detector_id][lane_id][sumo_traci_util.VAR_IS_PARTIAL_DETECTOR]

                if is_partial_detector:
                    for vehicle_id in lane_vehicle_ids:
                        start_position = \
                            self.detector_additional_info[detector_id][lane_id][sumo_traci_util.VAR_LANE_START_POSITION]
                        end_position = \
                            self.detector_additional_info[detector_id][lane_id][sumo_traci_util.VAR_LANE_END_POSITION]

                        vehicle_subscription = traci_connection.vehicle.getSubscriptionResults(vehicle_id)
                        lane_position = vehicle_subscription[tc.VAR_LANEPOSITION]

                        if start_position < lane_position <= end_position:
                            self.current_step_lane_area_detector_vehicle_ids[detector_id].append(vehicle_id)
                else:
                    self.current_step_lane_area_detector_vehicle_ids[detector_id].extend(lane_vehicle_ids)

    def _update_feature(self):

        feature_dict = {}
        for f in self.state_feature_list:
            feature_dict[f] = self.feature_dict_function[f]()

        self.feature_dict = feature_dict

    # ================= calculate detectors from current observations ======================

    def build_detector_subscription_function(self):

        detector_subscription_function = {
            tc.LAST_STEP_VEHICLE_NUMBER: lambda vehicle_ids:
                len(vehicle_ids),
            tc.LAST_STEP_VEHICLE_ID_LIST: lambda vehicle_ids:
                vehicle_ids,
            tc.LAST_STEP_VEHICLE_HALTING_NUMBER: lambda vehicle_ids:
                sum(1 if self.current_step_vehicle_subscription[vehicle_id][tc.VAR_SPEED] < 0.1 else 0
                    for vehicle_id in vehicle_ids),
            tc.VAR_WAITING_TIME: lambda vehicle_ids:
                sum(self.current_step_vehicle_subscription[vehicle_id][tc.VAR_WAITING_TIME]
                    for vehicle_id in vehicle_ids),
            tc.LAST_STEP_MEAN_SPEED: self.get_detector_last_step_mean_speed,
            sumo_traci_util.VAR_CUMULATIVE_LENGTH: self.get_detector_cumulative_length,
            tc.VAR_MAXSPEED: self.get_detector_max_speed
        }

        return detector_subscription_function

    def get_detector_last_step_mean_speed(self, vehicle_ids):
        speed_list = [self.current_step_vehicle_subscription[vehicle_id][tc.VAR_SPEED]
                      for vehicle_id in vehicle_ids]

        return np.average(speed_list) if len(speed_list) > 0 else 0

    def get_detector_cumulative_length(self, lane_tuples):

        length = sum([lane_length
                      if accumulated_detector_length < self.maximum_detector_length
                      else lane_length - (accumulated_detector_length - self.maximum_detector_length)
                      for _, lane_length, accumulated_detector_length in lane_tuples])

        return length

    def get_detector_max_speed(self, lane_tuples):

        max_speed = np.average(
            a=[
                self.current_step_lane_subscription[lane_id][tc.VAR_MAXSPEED]
                for lane_id, _, _ in lane_tuples
            ],
            weights=[
                lane_length
                if accumulated_detector_length < self.maximum_detector_length
                else lane_length - (accumulated_detector_length - self.maximum_detector_length)
                for _, lane_length, accumulated_detector_length in lane_tuples
            ]
        )

        return max_speed

    def get_detector_subscription_lane_aggregated_data(self, variable, detector_ids):

        selected_lane_tuples = set([])
        for _, detectors in self.subscription_extension.items():
            for detector_id, lane_tuples in detectors.items():
                if detector_id in detector_ids:
                    selected_lane_tuples.update(lane_tuples)

        result = self.detector_subscription_function[variable](selected_lane_tuples)

        return result

    def get_detector_subscription_vehicle_aggregated_data(self, variable, detector_ids):

        vehicle_ids = list(set([
            vehicle_id for detector_id in detector_ids
            for vehicle_id in self.current_step_lane_area_detector_vehicle_ids[detector_id]]))

        result = self.detector_subscription_function[variable](vehicle_ids)

        return result

    # ================= calculate features from current observations ======================

    def pad_movements(self, function, data_source, *args, **kwargs):

        state = [function(lanes, *args, **kwargs)
                 if len(lanes) > 0 else 0
                 for movement in self.unique_movements
                 for lanes in [data_source.get(movement, [])]]

        return state

    def get_density(self, detector_ids):

        vehicle_ids = self.get_detector_subscription_vehicle_aggregated_data(tc.LAST_STEP_VEHICLE_ID_LIST, detector_ids)

        vehicle_subscription_data = {
            vehicle_id: self.current_step_vehicle_subscription[vehicle_id] for vehicle_id in vehicle_ids
        }

        detector_additional_info = self.merge_detector_additional_info(detector_ids)

        detector_cumulative_length = self.get_detector_subscription_lane_aggregated_data(
            sumo_traci_util.VAR_CUMULATIVE_LENGTH, detector_ids)

        density = sumo_traci_util.get_relative_occupancy(
            vehicle_subscription_data,
            detector_cumulative_length,
            detector_additional_info,
            self.execution_name
        )

        return density

    def get_mean_relative_speed(self, detector_ids):

        vehicle_ids = self.get_detector_subscription_vehicle_aggregated_data(tc.LAST_STEP_VEHICLE_ID_LIST, detector_ids)

        vehicle_subscription_data = {
            vehicle_id: self.current_step_vehicle_subscription[vehicle_id] for vehicle_id in vehicle_ids
        }

        if vehicle_subscription_data:

            relative_speeds = [data[tc.VAR_SPEED] / data[tc.VAR_ALLOWED_SPEED]
                               for data in vehicle_subscription_data.values()]

            mean_relative_speed = np.average(relative_speeds)
        else:
            mean_relative_speed = 0

        return mean_relative_speed

    def get_time_loss(self, detector_ids):

        vehicle_ids = self.get_detector_subscription_vehicle_aggregated_data(tc.LAST_STEP_VEHICLE_ID_LIST, detector_ids)

        vehicle_subscription_data = {
            vehicle_id: self.current_step_vehicle_subscription[vehicle_id] for vehicle_id in vehicle_ids
        }

        time_loss = sumo_traci_util.get_time_loss(vehicle_subscription_data, self.execution_name)

        return time_loss

    def get_queue_length(self, detector_ids):

        queue_length = self.get_detector_subscription_vehicle_aggregated_data(
            tc.LAST_STEP_VEHICLE_HALTING_NUMBER, detector_ids)

        return queue_length

    def get_number_of_vehicles(self, detector_ids):

        number_of_vehicles = self.get_detector_subscription_vehicle_aggregated_data(
            tc.LAST_STEP_VEHICLE_NUMBER, detector_ids)

        return number_of_vehicles

    def get_waiting_time(self, detector_ids):

        waiting_time = self.get_detector_subscription_vehicle_aggregated_data(tc.VAR_WAITING_TIME, detector_ids)

        return waiting_time

    def get_number_of_vehicles_been_stopped(self, detector_ids, threshold):

        number_of_vehicles_ever_stopped = 0

        vehicle_ids = self.get_detector_subscription_vehicle_aggregated_data(tc.LAST_STEP_VEHICLE_ID_LIST, detector_ids)
        for vehicle_id in vehicle_ids:
            if self.vehicle_min_speed_dict[vehicle_id] < threshold:
                number_of_vehicles_ever_stopped += 1

        return number_of_vehicles_ever_stopped

    def get_current_time(self):
        traci_connection = traci.getConnection(self.execution_name)
        return traci_connection.simulation.getTime()

    def get_feature(self, feature_names):

        single_output = False
        if isinstance(feature_names, str):
            feature_names = [feature_names]
            single_output = True

        features = {}
        feature = None
        for feature_name in feature_names:
            if feature_name in self.feature_dict:
                feature = self.feature_dict[feature_name]
            elif feature_name in self.feature_dict_function:
                feature = self.feature_dict_function[feature_name]()
                self.feature_dict[feature_name] = feature
            else:
                raise ValueError("There is no " + str(feature_name))
            features[feature_name] = feature

        if single_output:
            return feature
        else:
            return features

    def get_state(self, state_feature_list):
        state_dict = {state_feature_name: self.feature_dict[state_feature_name]
                      for state_feature_name in state_feature_list}
        return state_dict

    def get_reward(self, reward_info_dict):

        reward = 0
        for r in reward_info_dict:
            if reward_info_dict[r] != 0:
                reward += reward_info_dict[r] * self.reward_dict_function[r]()

        return reward

    def get_phase_traffic_lights(self):
        phase_traffic_lights = {}

        for phase in self.phases:

            links_length = sumo_util.get_traffic_light_links_length(self.traffic_light_file_xml, self.traffic_light_id)
            phase_signal_string = ['r'] * links_length

            phase_movements = phase.split('_')

            for uncontrolled_movement in self.uncontrolled_movements:
                uncontrolled_movement_traffic_light_indices = \
                    self.movement_to_traffic_light_link_index[uncontrolled_movement]

                for uncontrolled_movement_traffic_light_index in uncontrolled_movement_traffic_light_indices:
                    if self.is_right_on_red and 'R' in uncontrolled_movement:

                        movement_link_state_list = []
                        for phase_movement in phase_movements:
                            if phase_movement[0] == uncontrolled_movement[0]:
                                movement_link_state = self.link_states[phase_movement]
                                movement_link_state_list.append(movement_link_state)

                        if 'M' in movement_link_state_list:
                            phase_signal_string[uncontrolled_movement_traffic_light_index] = 'G'
                        else:
                            phase_signal_string[uncontrolled_movement_traffic_light_index] = 's'

            for phase_movement_index, movement in enumerate(phase_movements):

                traffic_light_indices = self.movement_to_traffic_light_link_index[movement]

                for traffic_light_index in traffic_light_indices:
                    movement_link_state = self.link_states[movement]
                    if movement_link_state == 'm':
                        phase_signal_string[traffic_light_index] = 'g'
                    else:
                        phase_signal_string[traffic_light_index] = 'G'

            phase_traffic_lights[phase] = "".join(phase_signal_string)

        return phase_traffic_lights

    def get_subscription_extension(self, maximum_detector_length):

        subscription_extension = {
            'entering_lanes': {},
            'exiting_lanes': {},
            'intersection_lanes': {}
        }

        for entering_edge in self.entering_edges:

            lanes = list(entering_edge)

            for lane in lanes:

                lane_length_list = []

                lane_length = float(lane.get('length'))
                lane_length_list.append(lane_length)

                lane_id = lane.get('id')
                subscription_extension['entering_lanes'][lane_id] = \
                    [(lane_id, lane_length, sum(lane_length_list))]

                previous_internal_lane_chains = sumo_util.get_previous_lanes(self.net_file_xml, lane, internal=True)
                previous_external_lanes = sumo_util.get_previous_lanes(self.net_file_xml, lane)

                previous_lanes = list(zip(previous_internal_lane_chains, previous_external_lanes))

                previous_lane_stack = [previous_lanes]
                while len(previous_lane_stack) != 0:

                    while len(previous_lane_stack[-1]) != 0:

                        previous_internal_lanes, previous_external_lane = previous_lane_stack[-1].pop(0)

                        for previous_internal_lane in previous_internal_lanes[::-1]:
                            if sum(lane_length_list) > maximum_detector_length:
                                break

                            previous_internal_lane_id = previous_internal_lane.get('id')
                            previous_internal_lane_length = float(previous_internal_lane.get('length'))

                            keep_clear = previous_internal_lane.get('keepClear')
                            if keep_clear is not None and not keep_clear:
                                lane_length_list.append(previous_internal_lane_length)

                                subscription_extension['entering_lanes'][lane_id].append(
                                    (previous_internal_lane_id, previous_internal_lane_length, sum(lane_length_list))
                                )
                            else:
                                subscription_extension['entering_lanes'][lane_id].append(
                                    (previous_internal_lane_id, previous_internal_lane_length, 0)
                                )

                        if sum(lane_length_list) > maximum_detector_length:
                            continue

                        previous_external_lane_id = previous_external_lane.get('id')
                        previous_external_lane_length = float(previous_external_lane.get('length'))
                        lane_length_list.append(previous_external_lane_length)

                        subscription_extension['entering_lanes'][lane_id].append(
                            (previous_external_lane_id, previous_external_lane_length, sum(lane_length_list))
                        )

                        previous_internal_lane_chains = \
                            sumo_util.get_previous_lanes(self.net_file_xml, lane, internal=True)
                        previous_external_lanes = sumo_util.get_previous_lanes(
                            self.net_file_xml, previous_external_lane)

                        previous_lanes = list(zip(previous_internal_lane_chains, previous_external_lanes))

                        previous_lane_stack.append(previous_lanes)

                    previous_lane_stack.pop()
                    lane_length_list.pop()

        for exiting_edge in self.exiting_edges:

            lanes = list(exiting_edge)

            for lane in lanes:

                lane_length_list = []

                lane_length = float(lane.get('length'))
                lane_length_list.append(lane_length)

                lane_id = lane.get('id')
                subscription_extension['exiting_lanes'][lane_id] = \
                    [(lane_id, lane_length, sum(lane_length_list))]

                next_internal_lane_chains = sumo_util.get_next_lanes(self.net_file_xml, lane, internal=True)
                next_external_lanes = sumo_util.get_next_lanes(self.net_file_xml, lane)

                next_lanes = list(zip(next_internal_lane_chains, next_external_lanes))

                next_lane_stack = [next_lanes]
                while len(next_lane_stack) != 0:

                    while len(next_lane_stack[-1]) != 0:

                        next_internal_lanes, next_external_lane = next_lane_stack[-1].pop(0)

                        for next_internal_lane in next_internal_lanes:
                            if sum(lane_length_list) > maximum_detector_length:
                                break

                            next_internal_lane_id = next_internal_lane.get('id')
                            next_internal_lane_length = float(next_internal_lane.get('length'))

                            keep_clear = next_internal_lane.get('keepClear')
                            if keep_clear is not None and not keep_clear:
                                lane_length_list.append(next_internal_lane_length)

                                subscription_extension['exiting_lanes'][lane_id].append(
                                    (next_internal_lane_id, next_internal_lane_length, sum(lane_length_list))
                                )
                            else:
                                subscription_extension['exiting_lanes'][lane_id].append(
                                    (next_internal_lane_id, next_internal_lane_length, 0)
                                )

                        if sum(lane_length_list) > maximum_detector_length:
                            continue

                        next_external_lane_id = next_external_lane.get('id')
                        next_external_lane_length = float(next_external_lane.get('length'))
                        lane_length_list.append(next_external_lane_length)

                        subscription_extension['exiting_lanes'][lane_id].append(
                            (next_external_lane_id, next_external_lane_length, sum(lane_length_list))
                        )

                        next_internal_lane_chains = sumo_util.get_next_lanes(self.net_file_xml, lane, internal=True)
                        next_external_lanes = sumo_util.get_next_lanes(self.net_file_xml, lane)

                        next_lanes = list(zip(next_internal_lane_chains, next_external_lanes))

                        next_lane_stack.append(next_lanes)

                    next_lane_stack.pop()
                    lane_length_list.pop()

        for intersection_lane in self.internal_lanes + self.intermediary_lanes:

            intersection_lane_id = intersection_lane.get('id')
            lane_length = intersection_lane.get('length')

            subscription_extension['intersection_lanes'][intersection_lane_id] = [
                (intersection_lane_id, lane_length, 0)
            ]

        connections = sumo_util.get_intersection_connections(
            self.net_file_xml, self.id, self.multi_intersection_traffic_light_configuration)

        from_lane_to_intermediary_lane = {}
        for connection in connections:

            if isinstance(connection, list):
                inner_connections = connection
            else:
                inner_connections = [connection]

            from_lane = inner_connections[0].get('from') + '_' + inner_connections[0].get('fromLane')

            if from_lane not in from_lane_to_intermediary_lane:
                from_lane_to_intermediary_lane[from_lane] = set([])

            for inner_connection in inner_connections[1:]:

                connection_from_lane = inner_connection.get('from') + '_' + inner_connection.get('fromLane')
                inner_from_lane = self.net_file_xml.find('.//lane[@id="' + connection_from_lane + '"]')

                from_lane_to_intermediary_lane[from_lane].add(inner_from_lane)

        intermediary_lanes_by_entering_lane_dict = {}
        for entering_edge in self.entering_edges:

            lanes = list(entering_edge)

            for lane in lanes:

                lane_id = lane.get('id')

                intermediary_lanes_by_entering_lane_dict[lane_id] = []

                intermediary_lanes = from_lane_to_intermediary_lane.get(lane_id, [])

                for intermediary_lane in intermediary_lanes:
                    intermediary_lane_id = intermediary_lane.get('id')

                    lane_length = float(intermediary_lane.get('length'))

                    intermediary_lanes_by_entering_lane_dict[lane_id].append(
                        (intermediary_lane_id, lane_length, 0)
                    )

        subscription_extension['entering_lanes'] = {
            key: value + subscription_extension['entering_lanes'][key]
            for key, value in intermediary_lanes_by_entering_lane_dict.items()
        }

        return subscription_extension

    def merge_detector_additional_info(self, detector_ids):

        additional_info = {}
        for detector_id in detector_ids:
            for key, info in self.detector_additional_info[detector_id].items():
                if key in additional_info:
                    additional_info[key][sumo_traci_util.VAR_LANE_START_POSITION] = \
                        min(
                            additional_info[key][sumo_traci_util.VAR_LANE_START_POSITION],
                            info[sumo_traci_util.VAR_LANE_START_POSITION]
                        )
                    additional_info[key][sumo_traci_util.VAR_LANE_END_POSITION] = \
                        max(
                            additional_info[key][sumo_traci_util.VAR_LANE_END_POSITION],
                            info[sumo_traci_util.VAR_LANE_END_POSITION]
                        )

                    if additional_info[key][sumo_traci_util.VAR_IS_PARTIAL_DETECTOR] and \
                            not info[sumo_traci_util.VAR_IS_PARTIAL_DETECTOR]:
                        additional_info[key][sumo_traci_util.VAR_IS_PARTIAL_DETECTOR] = \
                            info[sumo_traci_util.VAR_IS_PARTIAL_DETECTOR]
                else:
                    additional_info[key] = info

        return additional_info

    def select_action_based_on_time_restriction(self, threshold=120):
        # order movements by the waiting time of the first vehicle
        # select all phases, covering all movements in order
        # check time necessary to avoid transgressing waiting time threshold

        if threshold == -1:
            return -1

        movement_waiting_time_dict = sumo_traci_util.get_movements_first_stopped_vehicle_greatest_waiting_time(
            self.movement_to_entering_lane, self.current_step_lane_area_detector_vehicle_ids,
            self.current_step_vehicle_subscription)

        movement_waiting_time_dict = {k: v for k, v in sorted(
            movement_waiting_time_dict.items(), key=lambda x: x[1], reverse=True)}

        phase_movements_list = [phase.split('_') for phase in self.phases]
        waiting_time_sum_per_phase = [sum(movement_waiting_time_dict[movement] for movement in phase_movements)
                                      for phase_movements in phase_movements_list]

        selected_phases = [self.phases[item[0]] for item in
                           sorted(enumerate(waiting_time_sum_per_phase), key=lambda x: x[1])]

        for index, phase in enumerate(selected_phases):
            
            movements = phase.split('_')
            
            waiting_times = []
            for movement in movements:
                waiting_times.append(movement_waiting_time_dict[movement])

            if max(waiting_times) + (index + 1) * self.min_action_time >= threshold:
                return self.unique_phases.index(selected_phases[0])

        return -1

    def select_active_action_time_action(self):
        
        if self.current_min_action_duration < self.min_action_time:
            # next phase only changes with a new action choice
            if self.next_phase_to_set_index is not None:
                return self.next_phase_to_set_index

        return -1

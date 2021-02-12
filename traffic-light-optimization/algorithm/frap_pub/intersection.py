import os
import sys

import numpy as np
import traci
import traci.constants as tc

from utils import sumo_traci_util, sumo_util, xml_util

from algorithm.frap_pub.definitions import ROOT_DIR


class Intersection:

    def __init__(self, intersection_index, vehicle_subscription_variables, dic_traffic_env_conf, dic_path,
                 execution_name, external_configurations=None):

        if external_configurations is None:
            external_configurations = {}

        self.intersection_index = intersection_index
        self.id = dic_traffic_env_conf['INTERSECTION_ID'][intersection_index]
        self.execution_name = execution_name

        net_file = os.path.join(ROOT_DIR, dic_path["PATH_TO_WORK_DIRECTORY"], dic_traffic_env_conf['NET_FILE'])
        self.net_file_xml = xml_util.get_xml(net_file)

        self.vehicle_subscription_variables = vehicle_subscription_variables

        self.movements = dic_traffic_env_conf['MOVEMENT'][intersection_index]
        self.phases = dic_traffic_env_conf['PHASE'][intersection_index]
        self.link_states = dic_traffic_env_conf['LINK_STATES'][intersection_index]
        self.movement_to_connection = dic_traffic_env_conf['MOVEMENT_TO_CONNECTION'][intersection_index]
        self.traffic_light_link_index_to_movement = \
            dic_traffic_env_conf['TRAFFIC_LIGHT_LINK_INDEX_TO_MOVEMENT'][intersection_index]

        self.is_right_on_red = dic_traffic_env_conf['IS_RIGHT_ON_RED']

        self.min_action_time = dic_traffic_env_conf['MIN_ACTION_TIME']
        self.has_per_second_decision = dic_traffic_env_conf.get('PER_SECOND_DECISION', False)

        self.dic_path = dic_path
        self.entering_edges, self.exiting_edges = sumo_util.get_intersection_edge_ids(
            self.net_file_xml,
            intersection_ids=self.id
        )
        self.edges = [edge.get('id') for edge in self.entering_edges + self.exiting_edges]
        self.internal_edges = [edge.get('id') for edge in sumo_util.get_internal_edges(self.net_file_xml, self.id)]
        self.all_edges = self.edges + self.internal_edges

        self.controlled_entering_lanes = []
        self.controlled_exiting_lanes = []
        self.entering_lanes = []
        self.exiting_lanes = []

        self.movement_to_entering_lane = {}
        self.movement_to_exiting_lane = {}

        self.uncontrolled_movements = []

        self.entering_lanes = set(self.entering_lanes)
        self.exiting_lanes = set(self.exiting_lanes)
        self.controlled_entering_lanes = set(self.controlled_entering_lanes)
        self.controlled_exiting_lanes = set(self.controlled_exiting_lanes)
        for movement, connections in self.movement_to_connection.items():

            if movement in self.movements:
                self.movement_to_entering_lane[movement] = set([])
                self.movement_to_exiting_lane[movement] = set([])
            else:
                self.uncontrolled_movements.append(movement)

            for connection in connections:

                from_lane = connection.get('from') + '_' + connection.get('fromLane')
                to_lane = connection.get('to') + '_' + connection.get('toLane')

                self.entering_lanes.add(from_lane)
                self.exiting_lanes.add(to_lane)

                if movement in self.movements:
                    self.controlled_entering_lanes.add(from_lane)
                    self.controlled_exiting_lanes.add(to_lane)

                    self.movement_to_entering_lane[movement].add(from_lane)
                    self.movement_to_exiting_lane[movement].add(to_lane)

            if movement in self.movements:
                self.movement_to_entering_lane[movement] = list(self.movement_to_entering_lane[movement])
                self.movement_to_exiting_lane[movement] = list(self.movement_to_exiting_lane[movement])

        self.entering_lanes = list(self.entering_lanes)
        self.exiting_lanes = list(self.exiting_lanes)
        self.controlled_entering_lanes = list(self.controlled_entering_lanes)
        self.controlled_exiting_lanes = list(self.controlled_exiting_lanes)

        self.lanes = self.entering_lanes + self.exiting_lanes
        self.internal_lanes = [lane.get('id') for lane in sumo_util.get_internal_lanes(self.net_file_xml, self.id)]
        self.all_lanes = self.lanes + self.internal_lanes

        self.controlled_lanes = self.controlled_entering_lanes + self.controlled_exiting_lanes

        self.movement_to_traffic_light_index = sumo_util.get_movement_traffic_light_controller(
            self.movement_to_connection)
        self.phase_traffic_lights = self.get_phase_traffic_lights()

        self.default_yellow_time = dic_traffic_env_conf['DEFAULT_YELLOW_TIME']
        self.movement_to_yellow_time = dic_traffic_env_conf['MOVEMENT_TO_YELLOW_TIME'][intersection_index]

        self.current_yellow_signal_movements = []
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
        self.current_step_lane_vehicle_subscription = None
        self.previous_step_lane_vehicle_subscription = None
        self.current_step_vehicles = []
        self.previous_step_vehicles = []

        self.vehicle_min_speed_dict = {}  # this second

        self.feature_dict = {}  # this second

        self.state_feature_list = dic_traffic_env_conf["STATE_FEATURE_LIST"]

        self.feature_dict_function = {
            'current_phase': lambda: [self.current_phase_index],
            'time_this_phase': lambda: [self.current_phase_duration],
            'vehicle_position_img': lambda: self._get_lane_vehicle_position(
                [[lane] for lane in self.controlled_entering_lanes]),
            'vehicle_speed_img': lambda: self._get_lane_vehicle_speed(
                [[lane] for lane in self.controlled_entering_lanes]),
            'vehicle_acceleration_img': lambda: None,
            'vehicle_waiting_time_img': lambda:
                self._get_lane_vehicle_accumulated_waiting_time(self.movement_to_entering_lane.values()),
            'movement_number_of_vehicles': lambda:
                self._get_number_of_vehicles(self.movement_to_entering_lane.values()),
            'movement_number_of_vehicles_been_stopped_threshold_01': lambda:
                self._get_number_of_vehicles_been_stopped(self.movement_to_entering_lane.values(), 0.1),
            'movement_number_of_vehicles_been_stopped_threshold_1': lambda:
                self._get_number_of_vehicles_been_stopped(self.movement_to_entering_lane.values(), 1),
            'movement_queue_length': lambda: self._get_queue_length(self.movement_to_entering_lane.values()),
            'movement_number_of_vehicles_left': lambda: None,
            'movement_sum_duration_vehicles_left': lambda: None,
            'movement_sum_waiting_time': lambda: self._get_waiting_time(self.movement_to_entering_lane.values()),
            'terminal': lambda: None,
            'movement_pressure_presslight': lambda:
                np.array(self._get_density(self.movement_to_entering_lane.values())) -
                np.array(self._get_density(self.movement_to_exiting_lane.values())),
            'movement_pressure_mplight': lambda:
                np.array(self._get_number_of_vehicles(self.movement_to_entering_lane.values())) -
                np.array(self._get_number_of_vehicles(self.movement_to_exiting_lane.values())),
            'movement_pressure_time_loss': lambda:
                np.array(self._get_time_loss(self.movement_to_entering_lane.values())) -
                np.array(self._get_time_loss(self.movement_to_exiting_lane.values())),
            'movement_sum_time_loss': lambda: self._get_time_loss(self.movement_to_entering_lane.values())
        }

        self.reward_dict_function = {
            'flickering': lambda: None,
            'sum_queue_length': lambda: -np.sum(self._get_queue_length(
                [[lane] for lane in self.controlled_entering_lanes])),
            'avg_movement_queue_length': lambda: -np.average(self.get_feature('movement_queue_length')),
            'sum_waiting_time': lambda: -np.sum(self._get_waiting_time(
                [[lane] for lane in self.controlled_entering_lanes])),
            'sum_num_vehicle_left': lambda: None,
            'sum_duration_vehicles_left': lambda: None,
            'sum_number_of_vehicles_been_stopped_threshold_01':
                lambda: -np.sum(self._get_number_of_vehicles_been_stopped(
                    [[lane] for lane in self.controlled_entering_lanes], 0.1)),
            'sum_number_of_vehicles_been_stopped_threshold_1':
                lambda: -np.sum(self._get_number_of_vehicles_been_stopped(
                    [[lane] for lane in self.controlled_entering_lanes], 1)),
            'pressure_presslight': lambda:
                -np.abs(np.sum(self.get_feature('movement_pressure_presslight'))),
            'pressure_mplight': lambda:
                -(np.sum(self._get_queue_length([[lane] for lane in self.controlled_entering_lanes])) -
                  np.sum(self._get_queue_length([[lane] for lane in self.controlled_exiting_lanes]))),
            'pressure_time_loss': lambda:
                -(np.sum(self._get_time_loss([[lane] for lane in self.controlled_entering_lanes])) -
                  np.sum(self._get_time_loss([[lane] for lane in self.controlled_exiting_lanes]))),
            'time_loss': lambda:
                -np.sum(self._get_time_loss([[lane] for lane in self.controlled_entering_lanes]))
        }

    def update_previous_measurements(self):

        self.previous_phase_index = self.current_phase_index
        self.previous_step_lane_subscription = self.current_step_lane_subscription
        self.previous_step_vehicle_subscription = self.current_step_vehicle_subscription
        self.previous_step_lane_vehicle_subscription = self.current_step_lane_vehicle_subscription
        self.previous_step_vehicles = self.current_step_vehicles

    def update_current_measurements(self):
        # need change, debug in seeing format
        
        traci_connection = traci.getConnection(self.execution_name)

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.current_min_action_duration += 1

        # ====== lane level observations =======

        self.current_step_lane_subscription = {lane_id: traci_connection.lane.getSubscriptionResults(lane_id)
                                               for lane_id in self.all_lanes}

        # ====== vehicle level observations =======

        # get vehicle list
        current_step_vehicles = []
        for lane_id, values in self.current_step_lane_subscription.items():

            lane_vehicles = self.current_step_lane_subscription[lane_id][tc.LAST_STEP_VEHICLE_ID_LIST]
            current_step_vehicles += lane_vehicles

        self.current_step_vehicles = current_step_vehicles
        recently_arrived_vehicles = list(set(self.current_step_vehicles) - set(self.previous_step_vehicles))
        recently_left_vehicles = list(set(self.previous_step_vehicles) - set(self.current_step_vehicles))

        # vehicle level observations
        self.current_step_vehicle_subscription = {vehicle: traci_connection.vehicle.getSubscriptionResults(vehicle)
                                                  for vehicle in self.current_step_vehicles}
        self.current_step_lane_vehicle_subscription = {}
        for vehicle_id, values in self.current_step_vehicle_subscription.items():
            lane_id = values[tc.VAR_LANE_ID]
            if lane_id in self.current_step_lane_vehicle_subscription:
                self.current_step_lane_vehicle_subscription[lane_id][vehicle_id] = \
                    self.current_step_vehicle_subscription[vehicle_id]
            else:
                self.current_step_lane_vehicle_subscription[lane_id] = \
                    {vehicle_id: self.current_step_vehicle_subscription[vehicle_id]}

        # update vehicle minimum speed in history
        self._update_vehicle_min_speed()

        # update feature
        self._update_feature()

    def set_signal(self, action, action_pattern):

        if self.yellow_flag:
            # in yellow phase
            self.flicker = 0

            yellow_signal_movements = self.current_yellow_signal_movements
            max_yellow_time = 0
            for movement in yellow_signal_movements:
                yellow_time = self.movement_to_yellow_time.get(movement, self.default_yellow_time)
                max_yellow_time = max(max_yellow_time, yellow_time)

            if self.current_phase_duration >= max_yellow_time:  # yellow time reached

                self.current_phase_index = self.next_phase_to_set_index
                phase = self.phases[self.current_phase_index]
                current_traffic_light = self.phase_traffic_lights[phase]

                sumo_traci_util.set_traffic_light_state(self.id, current_traffic_light, self.execution_name)
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
                    self.next_phase_to_set_index = action

                # set phase
                if self.current_phase_index == self.next_phase_to_set_index:  # the light phase keeps unchanged
                    if not self.has_per_second_decision:
                        self.current_min_action_duration = 0
                else:  # the light phase needs to change
                    # change to yellow first, and activate the counter and flag
                    current_traffic_light = sumo_traci_util.get_traffic_light_state(self.id, self.execution_name)

                    phase = self.phases[self.next_phase_to_set_index]
                    next_traffic_light = self.phase_traffic_lights[phase]

                    self.current_yellow_signal_movements = []
                    for index in range(0, len(current_traffic_light)):

                        if (current_traffic_light[index] == 'g' or current_traffic_light[index] == 'G') and \
                                (next_traffic_light[index] != 'g' and next_traffic_light[index] != 'G'):

                            movement = self.traffic_light_link_index_to_movement[index]
                            self.current_yellow_signal_movements.append(movement)

                            current_traffic_light = \
                                current_traffic_light[:index] + \
                                'y' + \
                                current_traffic_light[index + 1:]

                    sumo_traci_util.set_traffic_light_state(self.id, current_traffic_light, self.execution_name)
                    self.current_phase_index = self.yellow_phase_index
                    self.yellow_flag = True
                    self.flicker = 1

                    self.current_min_action_duration = 0

    # ================= update current step measurements ======================

    def _update_recently_left_vehicles_by_entering_lane(self):

        recently_left_vehicles_by_entering_lane = []

        # update vehicles leaving entering lane
        if self.previous_step_lane_subscription is None:
            for _ in self.entering_lanes:
                recently_left_vehicles_by_entering_lane.append([])
        else:
            for lane in self.entering_lanes:
                recently_left_vehicles_by_entering_lane.append(
                    list(
                        set(self.previous_step_lane_subscription[lane][tc.LAST_STEP_VEHICLE_ID_LIST]) -
                        set(self.current_step_lane_subscription[lane][tc.LAST_STEP_VEHICLE_ID_LIST])
                    )
                )
        return recently_left_vehicles_by_entering_lane

    def _update_vehicle_min_speed(self):
        result_dict = {}
        for vehicle_id, vehicle in self.current_step_vehicle_subscription.items():
            speed = vehicle[tc.VAR_SPEED]
            if vehicle_id in self.vehicle_min_speed_dict:  # this vehicle appeared in previous time stamps:
                result_dict[vehicle_id] = min(speed, self.vehicle_min_speed_dict[vehicle_id])
            else:
                result_dict[vehicle_id] = speed
        self.vehicle_min_speed_dict = result_dict

    def _update_feature(self):

        feature_dict = {}
        for f in self.state_feature_list:
            feature_dict[f] = self.feature_dict_function[f]()

        self.feature_dict = feature_dict

    # ================= calculate features from current observations ======================

    def _get_density(self, lanes_list):

        density_list = []
        for lanes in lanes_list:

            lane_subscription_data = {lane_id: self.current_step_lane_subscription[lane_id]
                                      for lane_id in lanes}

            lane_vehicle_subscription_data = {lane_id: self.current_step_lane_vehicle_subscription.get(lane_id, {})
                                              for lane_id in lanes}

            density = sumo_traci_util.get_movement_relative_occupancy(
                lane_subscription_data,
                lane_vehicle_subscription_data
            )

            density_list.append(density)

        return density_list

    def _get_time_loss(self, lanes_list):

        time_loss_list = []
        for lanes in lanes_list:

            time_loss = sum(sumo_traci_util.get_time_loss_by_lane(self.current_step_lane_vehicle_subscription, lanes,
                                                                  self.execution_name))

            time_loss_list.append(time_loss)

        return time_loss_list

    def _get_queue_length(self, lanes_list):

        queue_length_list = []
        for lanes in lanes_list:

            queue_length = sum(
                [self.current_step_lane_subscription[lane_id][tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
                 for lane_id in lanes])

            queue_length_list.append(queue_length)

        return queue_length_list

    def _get_number_of_vehicles(self, lanes_list):

        number_of_vehicles_list = []
        for lanes in lanes_list:

            number_of_vehicles = sum([self.current_step_lane_subscription[lane_id][tc.LAST_STEP_VEHICLE_NUMBER]
                                      for lane_id in lanes])

            number_of_vehicles_list.append(number_of_vehicles)

        return number_of_vehicles_list

    def _get_waiting_time(self, lanes_list):

        waiting_time_list = []
        for lanes in lanes_list:

            waiting_time = sum([self.current_step_lane_subscription[lane_id][tc.VAR_WAITING_TIME]
                                for lane_id in lanes])

            waiting_time_list.append(waiting_time)

        return waiting_time_list

    def _get_movements_list_vehicle_left(self, movement_to_entering_lane):

        return None

    def _get_lane_num_vehicle_left(self, lanes_list):
        list_lane_vehicle_left = self._get_movements_list_vehicle_left(lanes_list)
        list_lane_num_vehicle_left = [len(lane_vehicle_left) for lane_vehicle_left in list_lane_vehicle_left]
        return list_lane_num_vehicle_left

    def _get_lane_sum_duration_vehicle_left(self, lanes_list):

        # not implemented error
        raise NotImplementedError

    def _get_number_of_vehicles_been_stopped(self, lanes_list, threshold):

        number_of_vehicles_ever_stopped_list = []
        for lanes in lanes_list:

            vehicle_count = 0
            for lane_id in lanes:
                vehicle_ids = self.current_step_lane_subscription[lane_id][tc.LAST_STEP_VEHICLE_ID_LIST]
                for vehicle_id in vehicle_ids:
                    if self.vehicle_min_speed_dict[vehicle_id] < threshold:
                        vehicle_count += 1

            number_of_vehicles_ever_stopped_list.append(vehicle_count)

        return number_of_vehicles_ever_stopped_list

    def _get_position_grid_along_lane(self, vehicle):
        position = int(self.current_step_vehicle_subscription[vehicle][tc.VAR_LANEPOSITION])
        return min(position//self.length_grid, self.num_grid)

    def _get_lane_vehicle_position(self, lanes_list):

        lane_vector_list = []
        for lane in lanes_list:
            lane_vector = np.zeros(self.num_grid)
            vehicle_ids = self.current_step_lane_subscription[lane][tc.LAST_STEP_VEHICLE_ID_LIST]
            for vehicle_id in vehicle_ids:
                pos_grid = self._get_position_grid_along_lane(vehicle_id)
                lane_vector[pos_grid] = 1
            lane_vector_list.append(lane_vector)
        return np.array(lane_vector_list)

    def _get_lane_vehicle_speed(self, lanes_list):

        lane_vector_list = []
        for lane in lanes_list:
            lane_vector = np.full(self.num_grid, fill_value=np.nan)
            vehicle_ids = self.current_step_lane_subscription[lane][tc.LAST_STEP_VEHICLE_ID_LIST]
            for vehicle_id in vehicle_ids:
                pos_grid = self._get_position_grid_along_lane(vehicle_id)
                lane_vector[pos_grid] = self.current_step_vehicle_subscription[vehicle_id][tc.VAR_SPEED]
            lane_vector_list.append(lane_vector)
        return np.array(lane_vector_list)

    def _get_lane_vehicle_accumulated_waiting_time(self, lanes_list):

        lane_vector_list = []
        for lane in lanes_list:
            lane_vector = np.full(self.num_grid, fill_value=np.nan)
            vehicle_ids = self.current_step_lane_subscription[lane][tc.LAST_STEP_VEHICLE_ID_LIST]
            for vehicle_id in vehicle_ids:
                pos_grid = self._get_position_grid_along_lane(vehicle_id)
                lane_vector[pos_grid] = self.current_step_vehicle_subscription[
                    vehicle_id][tc.VAR_ACCUMULATED_WAITING_TIME]
            lane_vector_list.append(lane_vector)
        return np.array(lane_vector_list)

    def get_phase_traffic_lights(self):
        phase_traffic_lights = {}

        for phase in self.phases:

            phase_signal_string = ['r'] * len(self.traffic_light_link_index_to_movement.keys())

            phase_movements = phase.split('_')

            for uncontrolled_movement in self.uncontrolled_movements:
                uncontrolled_movement_traffic_light_indices = \
                    self.movement_to_traffic_light_index[uncontrolled_movement]

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

                traffic_light_indices = self.movement_to_traffic_light_index[movement]

                for traffic_light_index in traffic_light_indices:
                    movement_link_state = self.link_states[movement]
                    if movement_link_state == 'm':
                        phase_signal_string[traffic_light_index] = 'g'
                    else:
                        phase_signal_string[traffic_light_index] = 'G'

            phase_traffic_lights[phase] = "".join(phase_signal_string)

        return phase_traffic_lights

    # ================= get functions from outside ======================

    def get_current_time(self):
        traci_connection = traci.getConnection(self.execution_name)
        return traci_connection.simulation.getTime()

    def get_dic_vehicle_arrive_leave_time(self):
        return self.vehicle_arrive_leave_time_dict

    def get_feature(self, feature_names):

        single_output = False
        if isinstance(feature_names, str):
            feature_names = [feature_names]
            single_output = True

        features = {}
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

    def _update_feature(self):

        feature_dict = {}
        for f in self.state_feature_list:
            feature_dict[f] = self.feature_dict_function[f]()

        self.feature_dict = feature_dict

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

    def _get_vehicle_info(self, vehicle_id):
        try:
            lane_position = self.current_step_vehicle_subscription[vehicle_id][tc.VAR_LANEPOSITION]
            speed = self.current_step_vehicle_subscription[vehicle_id][tc.VAR_SPEED]
            return lane_position, speed
        except Exception as e:
            return None, None

    def select_action_based_on_time_restriction(self, threshold=120):
        # order movements by the waiting time of the first car
        # select all phases, covering all movements in order
        # check time necessary to avoid transgressing waiting time threshold

        if threshold == -1:
            return -1

        movement_waiting_time_dict = sumo_traci_util.get_movements_first_stopped_car_greatest_waiting_time(
            self.movement_to_entering_lane, self.current_step_lane_vehicle_subscription)

        movement_waiting_time_dict = {k: v for k, v in sorted(
            movement_waiting_time_dict.items(), key=lambda x: x[1], reverse=True)}

        ordered_movements = list(movement_waiting_time_dict.keys())

        selected_phases = sumo_util.match_ordered_movements_to_phases(ordered_movements, self.phases)

        for index, phase in enumerate(selected_phases):
            
            movements = phase.split('_')
            
            waiting_times = []
            for movement in movements:

                if movement in ordered_movements:
                    waiting_times.append(movement_waiting_time_dict[movement])
                    ordered_movements.remove(movement)

            if max(waiting_times) + (index + 1) * self.min_action_time >= threshold:
                return self.phases.index(selected_phases[0])

        return -1

    def select_active_action_time_action(self):
        
        if self.current_min_action_duration < self.min_action_time:
            # next phase only changes with a new action choice
            if self.next_phase_to_set_index is not None:
                return self.next_phase_to_set_index

        return -1

import os
import sys

import numpy as np
from lxml import etree
import traci
import traci.constants as tc

from utils import sumo_traci_util, sumo_util

from algorithm.frap_pub.definitions import ROOT_DIR


def get_traci_constant_mapping(constant_str):
    return getattr(tc, constant_str)


class Intersection:

    def __init__(self, light_id, list_vehicle_variables_to_sub, dic_traffic_env_conf, dic_path,
                 execution_name, external_configurations=None):
        '''
        still need to automate generation
        '''

        if external_configurations is None:
            external_configurations = {}

        self.execution_name = execution_name

        roadnet_file = external_configurations['ROADNET_FILE']

        net_file = os.path.join(ROOT_DIR, dic_path["PATH_TO_WORK_DIRECTORY"], roadnet_file)
        parser = etree.XMLParser(remove_blank_text=True)
        self.net_file_xml = etree.parse(net_file, parser)

        self.node_light = external_configurations['NODE_LIGHT']
        self.list_vehicle_variables_to_sub = list_vehicle_variables_to_sub

        self.phases = dic_traffic_env_conf["PHASE"]
        self.movements = dic_traffic_env_conf['list_lane_order']

        # ===== sumo intersection settings =====

        self.has_per_second_decision = dic_traffic_env_conf.get('PER_SECOND_DECISION', False)

        self.dic_path = dic_path
        self.incoming_edges, self.outgoing_edges = sumo_util.get_intersection_edge_ids(self.net_file_xml)
        self.edges = [] + self.incoming_edges + self.outgoing_edges

        self.list_approaches = [str(i) for i in range(dic_traffic_env_conf["N_LEG"])]
        self.dic_entering_approach_to_edge = {approach: self.incoming_edges[index]
                                              for index, approach in enumerate(self.list_approaches)}
        self.dic_exiting_approach_to_edge = {approach: self.outgoing_edges[index]
                                             for index, approach in enumerate(self.list_approaches)}

        self.min_action_time = dic_traffic_env_conf['MIN_ACTION_TIME']

        # grid settings
        self.length_lane = 300
        self.length_terminal = 50
        self.length_grid = 5
        self.num_grid = int(self.length_lane//self.length_grid)

        self.conflicts = dic_traffic_env_conf['CONFLICTS']

        self.movement_to_connection = dic_traffic_env_conf['movement_to_connection']

        self.list_entering_lanes = [connection.get('from') + '_' + connection.get('fromLane') 
                                    for _, connection in self.movement_to_connection.items() 
                                    if connection.get('dir') != 'r']
        self.list_exiting_lanes = [connection.get('to') + '_' + connection.get('toLane')
                                   for _, connection in self.movement_to_connection.items()
                                   if connection.get('dir') != 'r']
        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.lane_to_traffic_light_index_mapping = sumo_util.get_lane_traffic_light_controller(
            self.net_file_xml,
            self.list_entering_lanes)

        self.dic_phase_strs = {}

        for phase in self.phases:
            list_default_str = ["r"]*len(self.movements)

            for i, m in enumerate(self.movements):
                if 'r' in m.lower():
                    list_default_str[i] = 'g'

            phase_movements = phase.split('_')
            for index, movement in enumerate(phase_movements):

                movement_conflicts = self.conflicts[movement]

                if len(set(phase_movements[0:index]).intersection(set(movement_conflicts))) > 0:
                    list_default_str[self.movements.index(movement)] = 'g'
                else:
                    list_default_str[self.movements.index(movement)] = 'G'

            self.dic_phase_strs[phase] = "".join(list_default_str)

        self.all_yellow_phase_str = "y"*len(self.movements)
        self.all_red_phase_str = "r"*len(self.movements)

        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        # initialization

        # -1: all yellow, -2: all red, -3: none
        self.current_phase_index = 0
        self.previous_phase_index = 0
        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.current_min_action_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

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

    def set_signal(self, action, action_pattern, yellow_time, all_red_time):

        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time reached

                current_traffic_light = sumo_traci_util.get_traffic_light_state(self.node_light, self.execution_name)

                self.current_phase_index = self.next_phase_to_set_index
                phase = self.phases[self.current_phase_index]

                for lane_index, lane_id in enumerate(self.list_entering_lanes):
                    traffic_light_index = int(self.lane_to_traffic_light_index_mapping[lane_id])
                    new_lane_traffic_light = self.dic_phase_strs[phase][lane_index]
                    current_traffic_light = current_traffic_light[:traffic_light_index] + \
                                            new_lane_traffic_light + \
                                            current_traffic_light[traffic_light_index + 1:]

                sumo_traci_util.set_traffic_light_state(self.node_light, current_traffic_light, self.execution_name)
                self.all_yellow_flag = False
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
                    current_traffic_light = sumo_traci_util.get_traffic_light_state(self.node_light, self.execution_name)

                    phase = self.phases[self.next_phase_to_set_index]

                    for lane_index, lane_id in enumerate(self.list_entering_lanes):
                        traffic_light_index = int(self.lane_to_traffic_light_index_mapping[lane_id])
                        current_lane_traffic_light = current_traffic_light[traffic_light_index]
                        next_lane_traffic_light = self.dic_phase_strs[phase][lane_index]
                        
                        if (current_lane_traffic_light == 'g' or current_lane_traffic_light == 'G') and \
                        (next_lane_traffic_light != 'g' and next_lane_traffic_light != 'G'):
                            new_lane_traffic_light = self.all_yellow_phase_str[lane_index]
                            current_traffic_light = current_traffic_light[:traffic_light_index] + \
                                                    new_lane_traffic_light + \
                                                    current_traffic_light[traffic_light_index + 1:]

                    sumo_traci_util.set_traffic_light_state(self.node_light, current_traffic_light, self.execution_name)
                    self.current_phase_index = self.all_yellow_phase_index
                    self.all_yellow_flag = True
                    self.flicker = 1

                    self.current_min_action_duration = 0

    def update_previous_measurements(self):

        self.previous_phase_index = self.current_phase_index
        self.dic_lane_sub_previous_step = self.dic_lane_sub_current_step
        self.dic_vehicle_sub_previous_step = self.dic_vehicle_sub_current_step
        self.dic_lane_vehicle_sub_previous_step = self.dic_lane_vehicle_sub_current_step
        self.list_vehicles_previous_step = self.list_vehicles_current_step

    def update_current_measurements(self):
        # need change, debug in seeing format
        
        traci_connection = traci.getConnection(self.execution_name)

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.current_min_action_duration += 1

        # ====== lane level observations =======

        self.dic_lane_sub_current_step = {lane: traci_connection.lane.getSubscriptionResults(lane) for lane in self.list_lanes}

        # ====== vehicle level observations =======

        # get vehicle list
        self.list_vehicles_current_step = traci_connection.vehicle.getIDList()
        list_vehicles_new_arrive = list(set(self.list_vehicles_current_step) - set(self.list_vehicles_previous_step))
        list_vehicles_new_left = list(set(self.list_vehicles_previous_step) - set(self.list_vehicles_current_step))
        list_vehicles_new_left_entering_lane_by_lane = self._update_leave_entering_approach_vehicle()
        list_vehicles_new_left_entering_lane = []
        for l in list_vehicles_new_left_entering_lane_by_lane:
            list_vehicles_new_left_entering_lane += l

        # update subscriptions
        for vehicle in list_vehicles_new_arrive:
            traci_connection.vehicle.subscribe(vehicle, [getattr(tc, var) for var in self.list_vehicle_variables_to_sub])

        # vehicle level observations
        self.dic_vehicle_sub_current_step = {vehicle: traci_connection.vehicle.getSubscriptionResults(vehicle)
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

    # ================= update current step measurements ======================

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
                            [get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]) -
                        set(self.dic_lane_sub_current_step[lane]
                            [get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")])
                    )
                )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicles_arrive):

        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicles_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = \
                    {"enter_time": ts, "leave_time": np.nan}
            else:
                print("vehicle already exists!")
                sys.exit(-1)

    def _update_left_time(self, list_vehicles_left):

        ts = self.get_current_time()
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
            speed = vec_var[get_traci_constant_mapping("VAR_SPEED")]
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
        return [self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_HALTING_NUMBER")]
                for lane in list_lanes]

    def _get_lane_num_vehicle(self, list_lanes):
        '''
        vehicle number for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_NUMBER")]
                for lane in list_lanes]

    def _get_lane_sum_waiting_time(self, list_lanes):
        '''
        waiting time for each lane
        '''
        return [self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("VAR_WAITING_TIME")]
                for lane in list_lanes]

    def _get_lane_list_vehicle_left(self, list_lanes):
        '''
        get list of vehicles left at each lane
        ####### need to check
        '''

        return None

    def _get_lane_num_vehicle_left(self, list_lanes):

        list_lane_vehicle_left = self._get_lane_list_vehicle_left(list_lanes)
        list_lane_num_vehicle_left = [len(lane_vehicle_left) for lane_vehicle_left in list_lane_vehicle_left]
        return list_lane_num_vehicle_left

    def _get_lane_sum_duration_vehicle_left(self, list_lanes):

        # not implemented error
        raise NotImplementedError

    def _get_lane_num_vehicle_been_stopped(self, thres, list_lanes):

        list_num_of_vec_ever_stopped = []
        for lane in list_lanes:
            cnt_vec = 0
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                if self.dic_vehicle_min_speed[vec] < thres:
                    cnt_vec += 1
            list_num_of_vec_ever_stopped.append(cnt_vec)

        return list_num_of_vec_ever_stopped

    def _get_position_grid_along_lane(self, vec):
        pos = int(self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_LANEPOSITION")])
        return min(pos//self.length_grid, self.num_grid)

    def _get_lane_vehicle_position(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.zeros(self.num_grid)
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = 1
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    def _get_lane_vehicle_speed(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.full(self.num_grid, fill_value=np.nan)
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = self.dic_vehicle_sub_current_step[vec][get_traci_constant_mapping("VAR_SPEED")]
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    def _get_lane_vehicle_accumulated_waiting_time(self, list_lanes):

        list_lane_vector = []
        for lane in list_lanes:
            lane_vector = np.full(self.num_grid, fill_value=np.nan)
            list_vec_id = self.dic_lane_sub_current_step[lane][get_traci_constant_mapping("LAST_STEP_VEHICLE_ID_LIST")]
            for vec in list_vec_id:
                pos_grid = self._get_position_grid_along_lane(vec)
                lane_vector[pos_grid] = self.dic_vehicle_sub_current_step[vec][
                    get_traci_constant_mapping("VAR_ACCUMULATED_WAITING_TIME")
                ]
            list_lane_vector.append(lane_vector)
        return np.array(list_lane_vector)

    # ================= get functions from outside ======================

    def get_current_time(self):
        traci_connection = traci.getConnection(self.execution_name)
        return traci_connection.simulation.getTime()

    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

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

    def _get_vehicle_info(self, veh_id):
        try:
            pos = self.dic_vehicle_sub_current_step[veh_id][get_traci_constant_mapping("VAR_LANEPOSITION")]
            speed = self.dic_vehicle_sub_current_step[veh_id][get_traci_constant_mapping("VAR_SPEED")]
            return pos, speed
        except Exception as e:
            return None, None


    def select_action_based_on_time_restriction(self, threshold=120):
        # order movements by the waiting time of the first car
        # select all phases, covering all movements in order
        # check time necessary to avoid transgressing waiting time threshold

        if threshold == -1:
            return -1

        lane_waiting_time_dict = sumo_traci_util.get_lane_first_stopped_car_waiting_times(
            self.list_entering_lanes, self.dic_lane_vehicle_sub_current_step)

        movements_waiting_time_dict = {}
        for movement in self.movements:

            connection = self.movement_to_connection[movement]
            lane = connection['from'] + '_' + connection['fromLane']
            waiting_time = lane_waiting_time_dict[lane]

            movements_waiting_time_dict[movement] = waiting_time

        movements_waiting_time_dict = {k: v for k, v in sorted(
            movements_waiting_time_dict.items(), key=lambda x: x[1], reverse=True)}

        ordered_movements = list(movements_waiting_time_dict.keys())

        selected_phases = sumo_util.match_ordered_movements_to_phases(ordered_movements, self.phases)

        for index, phase in enumerate(selected_phases):
            
            movements = phase.split('_')
            
            waiting_times = []
            for movement in movements:

                if movement in ordered_movements:
                    waiting_times.append(movements_waiting_time_dict[movement])
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

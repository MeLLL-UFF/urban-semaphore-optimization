import os
import copy
import pickle

import numpy as np
import pandas as pd
from lxml import etree
import traci
import traci.constants as tc

from utils import sumo_util, sumo_traci_util, summary_util

from algorithm.sumo_based.definitions import ROOT_DIR


class SumoBased:

    # add more variables here if you need more measurements
    LIST_LANE_VARIABLES_TO_SUBSCRIBE = [
        'LAST_STEP_VEHICLE_NUMBER',
        'LAST_STEP_VEHICLE_ID_LIST',
        'LAST_STEP_VEHICLE_HALTING_NUMBER',
        'VAR_WAITING_TIME',

        'LANE_EDGE_ID',
        ### 'LAST_STEP_VEHICLE_ID_LIST',
        'VAR_LENGTH',
        'LAST_STEP_MEAN_SPEED',
        'VAR_MAXSPEED'
    ]

    # add more variables here if you need more measurements
    LIST_VEHICLE_VARIABLES_TO_SUBSCRIBE = [
        'VAR_POSITION',
        'VAR_SPEED',
        'VAR_WAITING_TIME',
        'VAR_ACCUMULATED_WAITING_TIME',
        'VAR_LANEPOSITION',

        ### 'VAR_SPEED',
        'VAR_ALLOWED_SPEED',
        'VAR_MINGAP',
        'VAR_TAU',
        ### 'VAR_LANEPOSITION',
        # 'VAR_LEADER',  # Problems with subscription
        # 'VAR_SECURE_GAP',  # Problems with subscription
        'VAR_LENGTH',
        'VAR_LANE_ID',
        'VAR_DECEL',

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

    def __init__(self, net_file, route_file, output_file, scenario, _type, traffic_level_configuration, step_length):

        self.net_file = net_file
        self.route_file = route_file
        self.output_file = output_file
        self.scenario = scenario
        self._type = _type
        self.traffic_level_configuration = traffic_level_configuration

        self.step_length = step_length

        self.name_base = scenario + '__' + _type + '__' + '_'.join(traffic_level_configuration)


        self.traffic_environment_configuration = {
            'LIST_STATE_FEATURE': [
                'current_phase',
                'lane_num_vehicle',
            ],
            'DIC_REWARD_INFO': {
                'number_of_stopped_vehicles_with_threshold_1': -0.25
            }
        }

        self.path_to_log = os.path.join(
            'records', _type, self.name_base, 'test')

        self.simulation_data = []

        parser = etree.XMLParser(remove_blank_text=True)
        self.net_file_xml = etree.parse(net_file, parser)

        self.intersection_id = sumo_util.get_intersections_ids(self.net_file_xml)[0]

        self.vehicle_variables_to_subscribe = self.LIST_VEHICLE_VARIABLES_TO_SUBSCRIBE

        # ===== sumo intersection settings =====

        movements, movement_to_connection = \
            sumo_util.detect_movements(self.net_file_xml, False)

        self.movements = movements
        self.movement_to_connection = movement_to_connection

        self.incoming_edges, self.outgoing_edges = sumo_util.get_intersection_edge_ids(self.net_file_xml)
        self.edges = [] + self.incoming_edges + self.outgoing_edges

        self.entering_lanes = [connection.get('from') + '_' + connection.get('fromLane')
                               for _, connection in movement_to_connection.items()
                               if connection.get('dir') != 'r']
        self.exiting_lanes = [connection.get('to') + '_' + connection.get('toLane')
                              for _, connection in movement_to_connection.items()
                              if connection.get('dir') != 'r']
        self.lanes = self.entering_lanes + self.exiting_lanes

        # -1: all yellow, -2: all red, -3: none
        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.current_phase_duration = -1

        self.current_step_lane_subscription_data = None
        self.previous_step_lane_subscription_data = None
        self.current_step_vehicle_subscription_data = None
        self.previous_step_vehicle_subscription_data = None
        self.current_step_lane_vehicle_subscription_data = None
        self.previous_step_lane_vehicle_subscription_data = None
        self.current_step_vehicles = []
        self.previous_step_vehicles = []

        self.vehicle_min_speed_dict = {}  # this second

        self.feature_dict = {}  # this second

    def run(self):

        self._start_traci()
        self._subscribe_to_traci_info()
        self._run_simulation(max_time=3600)
        self._end_traci()
        self._bulk_log()
        self.summary()
        self._consolidate_output_file(self.output_file)

    def visualize_policy_behavior(self):
        self._start_traci(gui=True)
        self._run_simulation(max_time=3600, visualize_only=True)
        self._end_traci()

    def summary(self):

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
        f = open(os.path.join(ROOT_DIR, self.path_to_log, 'inter_0_detailed.pkl'), 'rb')
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

        summary_util.consolidate_time_loss(time_loss_each_step, save_path, self.name_base)
        summary_util.consolidate_reward(reward_each_step, save_path, self.name_base)

        summary_util.consolidate_occupancy_and_speed_inflow_outflow(
            relative_occupancy_each_step,
            relative_mean_speed_each_step,
            self.movements,
            serializable_movement_to_connection,
            save_path,
            self.name_base)

        summary_util.consolidate_phase_and_demand(
            absolute_number_of_cars_each_step,
            traffic_light_each_step,
            self.movements,
            serializable_movement_to_connection,
            lane_to_traffic_light_index_mapping,
            save_path,
            self.name_base)


    def _start_traci(self, gui=False):

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([
            sumo_util.get_sumo_binary(gui),
            '-n', self.net_file,
            '-r', self.route_file,
            '--log', self.output_file,
            '--duration-log.statistics', str(True),
            '--time-to-teleport', str(-1),
            '--collision.stoptime', str(10),
            '--collision.mingap-factor', str(0),
            '--collision.action', 'warn',
            '--collision.check-junctions', str(True),
            '--ignore-junction-blocker', str(10), # Currently not working
            '--step-length', str(self.step_length)
        ])

    def _end_traci(self):
        traci.close()

    def _subscribe_to_traci_info(self):
        for lane in self.lanes:
            traci.lane.subscribe(lane, [getattr(tc, var) for var in self.LIST_LANE_VARIABLES_TO_SUBSCRIBE])

    def _run_simulation(self, max_time=float('inf'), visualize_only=False):
        while traci.simulation.getMinExpectedNumber() > 0 and \
                traci.simulation.getTime() < max_time:

            self._update_previous_measurements()

            for i in range(int(1/self.step_length)):
                traci.simulationStep()

            self._update_current_measurements()

            if not visualize_only:
                self._collect_simulation_data()

            blocked_vehicles = sumo_traci_util.detect_deadlock(self.net_file_xml, self.current_step_vehicle_subscription_data)
            sumo_traci_util.resolve_deadlock(blocked_vehicles, self.net_file_xml, self.current_step_vehicle_subscription_data)

    def _collect_simulation_data(self):
        
        current_time = sumo_traci_util.get_current_time()
        state = self._get_state(self.traffic_environment_configuration['LIST_STATE_FEATURE'])
        action = -1
        reward = self._get_reward(self.traffic_environment_configuration['DIC_REWARD_INFO'])

        traffic_light = sumo_traci_util.get_traffic_light_state(self.intersection_id)
        time_loss = sumo_traci_util.get_time_loss(self.current_step_vehicle_subscription_data)
        relative_occupancy = sumo_traci_util.get_lane_relative_occupancy(
            self.current_step_lane_subscription_data,
            self.current_step_lane_vehicle_subscription_data,
            self.current_step_vehicle_subscription_data,
            self.edges
        )
        relative_mean_speed = sumo_traci_util.get_relative_mean_speed(
            self.current_step_lane_subscription_data,
            self.edges
        )
        absolute_number_of_cars = sumo_traci_util.get_absolute_number_of_cars(
            self.current_step_lane_subscription_data,
            self.edges
        )

        extra = {
            'traffic_light': traffic_light,
            'time_loss': time_loss,
            'relative_occupancy': relative_occupancy,
            'relative_mean_speed': relative_mean_speed,
            'absolute_number_of_cars': absolute_number_of_cars
        }

        self.simulation_data.append({'time': current_time,
                                     'state': state,
                                     'action': action,
                                     'reward': reward,
                                     'extra': extra})

    def _bulk_log(self):

        path_to_log_file = os.path.join(self.path_to_log, 'inter_0_detailed.pkl')

        if not os.path.exists(ROOT_DIR + '/' + self.path_to_log):
            os.makedirs(ROOT_DIR + '/' + self.path_to_log)

        f = open(ROOT_DIR + '/' + path_to_log_file, 'wb')
        pickle.dump(self.simulation_data, f)
        f.close()

    def _consolidate_output_file(self, output_file):

        duration = sumo_util.get_average_duration_statistic(output_file)

        filename = output_file.rsplit('.', 2)[0]
        duration_df = pd.DataFrame()
        duration_df.loc[0, 'test'] = duration
        duration_df.to_csv(filename + '_' + 'result' + '.csv')

    def _update_previous_measurements(self):

        self.previous_phase_index = self.current_phase_index
        self.previous_step_lane_subscription_data = self.current_step_lane_subscription_data
        self.previous_step_vehicle_subscription_data = self.current_step_vehicle_subscription_data
        self.previous_step_lane_vehicle_subscription_data = self.current_step_lane_vehicle_subscription_data
        self.previous_step_vehicles = self.current_step_vehicles

    def _update_current_measurements(self):
        # need change, debug in seeing format

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        # ====== lane level observations =======

        self.current_step_lane_subscription_data = \
            {lane: traci.lane.getSubscriptionResults(lane) for lane in self.lanes}

        # ====== vehicle level observations =======

        # get vehicle list
        self.current_step_vehicles = traci.vehicle.getIDList()
        new_arrived_vehicles = list(set(self.current_step_vehicles) - set(self.previous_step_vehicles))
        just_left_vehicles = list(set(self.previous_step_vehicles) - set(self.current_step_vehicles))

        # update subscriptions
        for vehicle in new_arrived_vehicles:
            traci.vehicle.subscribe(vehicle, [getattr(tc, var) for var in self.vehicle_variables_to_subscribe])

        # vehicle level observations
        self.current_step_vehicle_subscription_data = {vehicle: traci.vehicle.getSubscriptionResults(vehicle)
                                                       for vehicle in self.current_step_vehicles}
        
        self.current_step_lane_vehicle_subscription_data = {}
        for vehicle, values in self.current_step_vehicle_subscription_data.items():
            lane = values[tc.VAR_LANE_ID]
            if lane in self.current_step_lane_vehicle_subscription_data:
                self.current_step_lane_vehicle_subscription_data[lane][vehicle] = \
                    self.current_step_vehicle_subscription_data[vehicle]
            else:
                self.current_step_lane_vehicle_subscription_data[lane] = \
                    {vehicle: self.current_step_vehicle_subscription_data[vehicle]}

        # update vehicle minimum speed in history
        self._update_vehicle_min_speed()

        # update feature
        self._update_feature_info()

    def _update_vehicle_min_speed(self):
        '''
        record the minimum speed of one vehicle so far
        :return:
        '''
        result = {}
        for vehicle_id, vehicle in self.current_step_vehicle_subscription_data.items():
            speed = vehicle[tc.VAR_SPEED]
            if vehicle_id in self.vehicle_min_speed_dict:  # this vehicle appeared in previous time stamps:
                result[vehicle_id] = min(speed, self.vehicle_min_speed_dict[vehicle_id])
            else:
                result[vehicle_id] = speed

        self.vehicle_min_speed_dict = result

    def _update_feature_info(self):

        feature_dict = dict()

        feature_dict['current_phase'] = [self.current_phase_index]
        feature_dict['lane_num_vehicle'] = sumo_traci_util.get_lane_number_of_vehicles(
            self.entering_lanes, self.current_step_lane_subscription_data)
        feature_dict['lane_number_of_vehicles_ever_stopped_with_threshold_1'] = \
            self._get_lane_number_of_vehicle_been_stopped(self.entering_lanes, 1)

        self.feature_dict = feature_dict

    def _get_lane_number_of_vehicle_been_stopped(self, lanes, threshold):

        number_of_vehicles_ever_stopped = []
        for lane in lanes:
            number_of_vehicles = 0
            vehicles = self.current_step_lane_subscription_data[lane][tc.LAST_STEP_VEHICLE_ID_LIST]
            for vehicle in vehicles:
                if self.vehicle_min_speed_dict[vehicle] < threshold:
                    number_of_vehicles += 1
            number_of_vehicles_ever_stopped.append(number_of_vehicles)

        return number_of_vehicles_ever_stopped

    def _get_state(self, state_features):
        state = {
            state_feature_name: self.feature_dict[state_feature_name]
            for state_feature_name in state_features
        }
        return state

    def _get_reward(self, reward_info):

        reward_dict = dict()
        reward_dict['number_of_stopped_vehicles_with_threshold_1'] = \
            np.sum(self.feature_dict['lane_number_of_vehicles_ever_stopped_with_threshold_1'])

        reward = 0
        for r in reward_info:
            if reward_info[r] != 0:
                reward += reward_info[r] * reward_dict[r]

        return reward


import math

import traci
import traci.constants as tc
import numpy as np
from shapely.geometry import Polygon, CAP_STYLE

from utils import sumo_util, math_util


def get_current_time(traci_label=None):

    if traci_label is None:
        traci_connection = traci
    else:
        traci_connection = traci.getConnection(traci_label)

    return traci_connection.simulation.getTime()


def set_traffic_light_state(intersection, state, traci_label=None):
    
    if traci_label is None:
        traci_connection = traci
    else:
        traci_connection = traci.getConnection(traci_label)

    traci_connection.trafficlight.setRedYellowGreenState(intersection, state)


def get_movements_first_stopped_vehicle_greatest_waiting_time(
        movement_to_entering_lane, lane_vehicle_subscription_data):

    result = {movement: 0 for movement in movement_to_entering_lane.keys()}

    for movement, entering_lanes in movement_to_entering_lane.items():
        entering_lanes = np.unique(entering_lanes).tolist()

        greatest_waiting_time = 0
        for lane_id in entering_lanes:

            if lane_id in lane_vehicle_subscription_data:
                subscription_data = lane_vehicle_subscription_data[lane_id]
                _, vehicle = next(iter(subscription_data.items()))
                vehicle_waiting_time = vehicle[tc.VAR_WAITING_TIME]
            else:
                vehicle_waiting_time = 0

            greatest_waiting_time = max(greatest_waiting_time, vehicle_waiting_time)

        result[movement] = greatest_waiting_time

    return result


def get_traffic_light_state(traffic_light_id, traci_label=None):

    if traci_label is None:
        traci_connection = traci
    else:
        traci_connection = traci.getConnection(traci_label)

    return traci_connection.trafficlight.getRedYellowGreenState(traffic_light_id)


def get_lane_number_of_vehicles(lanes, lane_subscription_data):
    return [lane_subscription_data[lane][tc.LAST_STEP_VEHICLE_NUMBER] for lane in lanes]


def get_time_loss(vehicle_subscription_data, traci_label=None):

    if traci_label is None:
        traci_connection = traci
    else:
        traci_connection = traci.getConnection(traci_label)

    if vehicle_subscription_data:
        
        relative_speeds = [data[tc.VAR_SPEED] / data[tc.VAR_ALLOWED_SPEED]
                           for data in vehicle_subscription_data.values()]
        
        running = len(relative_speeds)
        step_length = traci_connection.simulation.getDeltaT()        
        mean_relative_speed = sum(relative_speeds) / running
        
        time_loss = (1 - mean_relative_speed) * running * step_length
    else:
        time_loss = 0

    return time_loss


def get_time_loss_by_lane(lane_vehicle_subscription_data, lanes, traci_label=None):

    time_loss_by_lane = []
    for lane in lanes:

        if lane in lane_vehicle_subscription_data:
            time_loss = get_time_loss(lane_vehicle_subscription_data[lane], traci_label)
        else:
            time_loss = 0
        time_loss_by_lane.append(time_loss)

    return time_loss_by_lane


def get_relative_occupancy(lane_subscription_data, lane_vehicle_subscription_data, traci_label=None):
    if traci_label is None:
        traci_connection = traci
    else:
        traci_connection = traci.getConnection(traci_label)

    all_vehicles = {vehicle_id: vehicle
                    for lane_id in lane_subscription_data.keys()
                    for vehicle_id, vehicle in lane_vehicle_subscription_data.get(lane_id, {}).items()}

    total_occupied_length_list = []
    lane_length_list = []

    for lane_id, lane in lane_subscription_data.items():

        vehicles = lane_vehicle_subscription_data.get(lane_id, {})
        lane_length = lane[tc.VAR_LENGTH]

        total_occupied_length = 0

        # Accounts for lane next entering vehicle secure gap spacing
        if vehicles:
            vehicle_id, vehicle = next(iter(vehicles.items()))
            min_gap = vehicle[tc.VAR_MINGAP]
            last_vehicle_secure_gap_margin = vehicle[tc.VAR_TAU] * vehicle[tc.VAR_SPEED] + min_gap
            actual_distance = lane_length - vehicle[tc.VAR_LANEPOSITION]
            total_occupied_length += min(last_vehicle_secure_gap_margin, actual_distance)

        for vehicle_id, vehicle in vehicles.items():
            min_gap = vehicle[tc.VAR_MINGAP]
            leader_vehicle_result = traci_connection.vehicle.getLeader(vehicle_id)

            if leader_vehicle_result:
                leader_id, leader_vehicle_distance = leader_vehicle_result
            else:
                leader_id = None
                leader_vehicle_distance = None

            if leader_id in all_vehicles:
                leader_vehicle = all_vehicles[leader_id]

                actual_distance = max(0 + min_gap, leader_vehicle_distance + min_gap)
                secure_gap = traci_connection.vehicle.getSecureGap(
                    vehicle_id,
                    vehicle[tc.VAR_SPEED],
                    leader_vehicle[tc.VAR_SPEED],
                    leader_vehicle[tc.VAR_DECEL],
                    leader_id)
                secure_gap += min_gap

            else:
                actual_distance = lane_length - vehicle[tc.VAR_LANEPOSITION]
                secure_gap = vehicle[tc.VAR_TAU] * vehicle[tc.VAR_SPEED] + min_gap

            occupied_length = vehicle[tc.VAR_LENGTH] + min(secure_gap, actual_distance)

            total_occupied_length += occupied_length

        total_occupied_length_list.append(total_occupied_length)
        lane_length_list.append(lane_length)

    relative_occupancy = sum(total_occupied_length_list) / sum(lane_length_list)

    return relative_occupancy


def get_relative_mean_speed(lane_subscription_data):

    result = {}

    for lane_id, lane in lane_subscription_data.items():

        vehicles = lane[tc.LAST_STEP_VEHICLE_ID_LIST]

        if vehicles:
            mean_speed = lane[tc.LAST_STEP_MEAN_SPEED]
        else:
            mean_speed = 0
        
        result[lane_id] = mean_speed / lane[tc.VAR_MAXSPEED]

    return result


def get_bounding_box(vehicle_id, vehicle_subscription_data):

    front_center = vehicle_subscription_data[vehicle_id][tc.VAR_POSITION]
    length = vehicle_subscription_data[vehicle_id][tc.VAR_LENGTH]
    width = vehicle_subscription_data[vehicle_id][tc.VAR_WIDTH]
    sumo_angle = vehicle_subscription_data[vehicle_id][tc.VAR_ANGLE]

    angle = math.radians(sumo_util.convert_sumo_angle_to_canonical_angle(sumo_angle))

    _, back_center = math_util.line(front_center, angle - math.pi, length)
    
    _, front_1 = math_util.line(front_center, angle + math.pi/2, width/2)
    _, front_2 = math_util.line(front_center, angle - math.pi/2, width/2)

    _, back_1 = math_util.line(back_center, angle + math.pi/2, width/2)
    _, back_2 = math_util.line(back_center, angle - math.pi/2, width/2)

    vehicle_bounding_box = Polygon([front_1, front_2, back_2, back_1])

    return vehicle_bounding_box


def get_blocking_vehicles(vehicle_id, polyline_path, possible_blocking_vehicles, bounding_boxes,
                          vehicle_subscription_data):

    vehicle_position = vehicle_subscription_data[vehicle_id][tc.VAR_POSITION]
    remaining_path = math_util.retrieve_remaining_path(vehicle_position, polyline_path)

    vehicle_width = vehicle_subscription_data[vehicle_id][tc.VAR_WIDTH]
    path_area = remaining_path.buffer(vehicle_width/2, resolution=0, cap_style=CAP_STYLE.flat)

    blocking_vehicles = []
    for possible_blocking_vehicle in possible_blocking_vehicles:

        bounding_box = bounding_boxes[possible_blocking_vehicle]
        is_blocking = bounding_box.intersection(path_area)

        if is_blocking:
            blocking_vehicles.append(possible_blocking_vehicle)

    return blocking_vehicles


def get_next_lane(vehicle, net_xml, vehicle_subscription_data):

    route = vehicle_subscription_data[vehicle][tc.VAR_EDGES]
    route_index = vehicle_subscription_data[vehicle][tc.VAR_ROUTE_INDEX]
    current_edge = route[route_index]

    remaining_route = route[route_index + 1:]

    if remaining_route:
        next_edge = route[route_index + 1]
        connection = sumo_util.get_connections(net_xml, from_edge=current_edge, to_edge=next_edge)

        next_lane = next_edge + '_' + connection[0].get('toLane')
    else:
        next_lane = None

    return next_lane


# Sumo 1.7.0 only
def detect_deadlock(intersection_id, net_xml, vehicle_subscription_data,
                    waiting_too_long_threshold=10, traci_label=None):

    if waiting_too_long_threshold == -1:
        return []

    if traci_label is None:
        traci_connection = traci
    else:
        traci_connection = traci.getConnection(traci_label)

    internal_lanes = sumo_util.get_internal_lanes(net_xml, intersection_id)
    lane_path_dict = sumo_util.get_internal_lane_paths(net_xml, intersection_id, internal_lanes)

    vehicle_lane_dict = {}
    vehicle_waiting_times = {}

    for lane in internal_lanes:

        lane_id = lane.get('id')
        vehicles = traci_connection.lane.getLastStepVehicleIDs(lane_id)
        
        for vehicle_id in vehicles:
            vehicle_lane_dict[vehicle_id] = lane_id
            vehicle_waiting_times[vehicle_id] = vehicle_subscription_data[vehicle_id][tc.VAR_WAITING_TIME]

    vehicle_waiting_times = {k: v for k, v in
                             sorted(vehicle_waiting_times.items(), key=lambda item: item[1], reverse=True)}

    vehicles_stopped = []
    vehicles_waiting = []
    vehicles_waiting_too_long = []
    for vehicle_id, waiting_time in vehicle_waiting_times.items():
        
        stop_state = vehicle_subscription_data[vehicle_id][tc.VAR_STOPSTATE]
        is_stopped = int(f'{stop_state:08b}'[-1]) == 1

        if waiting_time > 0:
            vehicles_waiting.append(vehicle_id)
        if waiting_time >= waiting_too_long_threshold:
            vehicles_waiting_too_long.append(vehicle_id)
        if is_stopped:
            vehicles_stopped.append(vehicle_id)

    blocked_vehicles = {}

    bounding_boxes = {}
    if vehicles_waiting_too_long:

        bounding_boxes = {vehicle: get_bounding_box(vehicle, vehicle_subscription_data)
                          for vehicle in vehicles_waiting + vehicles_stopped}

    for vehicle_waiting_too_long in vehicles_waiting_too_long:

        possible_blocking_vehicles = \
            [vehicle_id for vehicle_id in vehicles_waiting
                if vehicle_lane_dict[vehicle_waiting_too_long] != vehicle_lane_dict[vehicle_id]] + \
            [vehicle_id for vehicle_id in vehicles_stopped
                if vehicles_stopped != vehicle_id]

        lane = vehicle_subscription_data[vehicle_waiting_too_long][tc.VAR_LANE_ID]
        polyline_path = lane_path_dict[lane]

        blocking_vehicles = get_blocking_vehicles(
            vehicle_waiting_too_long,
            polyline_path,
            possible_blocking_vehicles,
            bounding_boxes,
            vehicle_subscription_data)

        if blocking_vehicles:
            blocked_vehicles[vehicle_waiting_too_long] = lane

    return blocked_vehicles


# Sumo 1.7.0 only
def resolve_deadlock(blocked_vehicles, net_xml, vehicle_subscription_data, traci_label=None):

    if traci_label is None:
        traci_connection = traci
    else:
        traci_connection = traci.getConnection(traci_label)

    for index, (blocked_vehicle, lane) in enumerate(blocked_vehicles.items()):

        next_lane = get_next_lane(blocked_vehicle, net_xml, vehicle_subscription_data)

        lane_position = 0
        for other_blocked_vehicle, other_lane in list(blocked_vehicles.items())[index+1:]:
            if lane == other_lane:
                lane_position += vehicle_subscription_data[other_blocked_vehicle][tc.VAR_LENGTH] + 1

        traci_connection.vehicle.moveTo(blocked_vehicle, next_lane, lane_position)

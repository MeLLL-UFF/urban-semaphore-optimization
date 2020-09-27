
import traci
import traci.constants as tc


def get_current_time():
    return traci.simulation.getTime()


def set_traffic_light_state(intersection, state):
    traci.trafficlight.setRedYellowGreenState(intersection, state)


def get_lane_first_stopped_car_waiting_times(lanes, lane_vehicle_subscription_data):

    result = {lane: 0 for lane in lanes}

    for lane_id, subscription_data in lane_vehicle_subscription_data.items():
        vehicle_id, vehicle = next(iter(subscription_data.items()))
        result[lane_id] = vehicle[tc.VAR_WAITING_TIME]

    return result


def get_traffic_light_state(intersection):
    return traci.trafficlight.getRedYellowGreenState(intersection)

def get_lane_number_of_vehicles(lanes, lane_subscription_data):
    return [lane_subscription_data[lane][tc.LAST_STEP_VEHICLE_NUMBER] for lane in lanes]

def get_time_loss(vehicle_subscription_data):

    if vehicle_subscription_data:
        
        relative_speeds = [data[tc.VAR_SPEED] / data[tc.VAR_ALLOWED_SPEED]
                           for data in vehicle_subscription_data.values()]
        
        running = len(relative_speeds)
        step_length = traci.simulation.getDeltaT()        
        mean_relative_speed = sum(relative_speeds) / running
        
        time_loss = (1 - mean_relative_speed) * running * step_length
    else:
        time_loss = 0

    return time_loss


def get_lane_relative_occupancy(lane_subscription_data, lane_vehicle_subscription_data, vehicle_subscription_data,
                                edges):

    result = {}

    all_vehicles = vehicle_subscription_data

    for lane_id, lane in lane_subscription_data.items():

        edge_id = lane[tc.LANE_EDGE_ID]

        if edge_id not in edges:
            continue

        vehicles = lane_vehicle_subscription_data.get(lane_id, {})
        lane_length = lane[tc.VAR_LENGTH]

        total_occupied_length = 0

        # Accounts for lane next entering car secure gap spacing
        if vehicles:
            vehicle_id, vehicle = next(iter(vehicles.items()))
            min_gap = vehicle[tc.VAR_MINGAP]
            last_vehicle_secure_gap_margin = vehicle[tc.VAR_TAU] * vehicle[tc.VAR_SPEED] + min_gap
            actual_distance = lane_length - vehicle[tc.VAR_LANEPOSITION]
            total_occupied_length += min(last_vehicle_secure_gap_margin, actual_distance)

        for vehicle_id, vehicle in vehicles.items():
            min_gap = vehicle[tc.VAR_MINGAP]
            leader_vehicle_result = traci.vehicle.getLeader(vehicle_id)

            if leader_vehicle_result:

                leader_id, leader_vehicle_distance = leader_vehicle_result
                leader_vehicle = all_vehicles[leader_id]

                actual_distance = max(0 + min_gap, leader_vehicle_distance + min_gap)
                secure_gap = traci.vehicle.getSecureGap(
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

        lane_relative_occupancy = total_occupied_length / lane_length

        result[lane_id] = lane_relative_occupancy

    return result


def get_relative_mean_speed(lane_subscription_data, edges):

    result = {}

    for lane_id, lane in lane_subscription_data.items():

        edge_id = lane[tc.LANE_EDGE_ID]

        if edge_id not in edges:
            continue

        vehicles = lane[tc.LAST_STEP_VEHICLE_ID_LIST]

        if vehicles:
            mean_speed = lane[tc.LAST_STEP_MEAN_SPEED]
        else:
            mean_speed = 0
        
        result[lane_id] = mean_speed / lane[tc.VAR_MAXSPEED]

    return result


def get_absolute_number_of_cars(lane_subscription_data, edges):

    result = {}

    for lane_id, lane in lane_subscription_data.items():

        edge_id = lane[tc.LANE_EDGE_ID]

        if edge_id not in edges:
            continue

        vehicles = lane[tc.LAST_STEP_VEHICLE_ID_LIST]

        result[lane_id] = len(vehicles)

    return result

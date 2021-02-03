import os
import ast
import copy
import math
import itertools
from functools import cmp_to_key

import numpy as np
from sumolib import checkBinary
import lxml.etree as etree
from shapely.geometry import Point, LineString

from utils.bidict import bidict
from utils import math_util, xml_util
from city.flow.configurer.flow_configurer import FlowConfigurer
from city.traffic_light_system.traffic_light.configurer.traffic_light_configurer_factory import traffic_light_configurer_instances
from definitions import get_network_file_path, get_route_file_path


def get_sumo_binary(gui=False):

    if gui:
        sumo_binary = checkBinary('sumo-gui')
    else:
        sumo_binary = checkBinary('sumo')

    return sumo_binary


net_xml_copy = None
route_xml_copy = None


def configure_sumo_traffic_light_parameters(experiment, traffic_light_parameters=None):

    if traffic_light_parameters is None:
        traffic_light_parameters = {}

    global net_xml_copy

    configurer = traffic_light_configurer_instances.get(experiment.strategy, None)

    if not configurer:
        return

    configurer.set_parameters(traffic_light_parameters)

    scenario = experiment.scenario
    net_file = os.path.abspath(get_network_file_path(scenario))

    net_xml = xml_util.get_xml(net_file)
    net_xml_copy = copy.deepcopy(net_xml)
    configurer.set_network_definition(net_xml)

    tl_logics = net_xml.findall(".//tlLogic")

    for tlLogic in tl_logics:

        params = tlLogic.findall('.//param')
        optimizable_phases_param = params[0]
        phases = tlLogic.findall('.//phase')

        configurer.configure_traffic_light(tlLogic)
        optimizable_phases = ast.literal_eval(optimizable_phases_param.get('value'))

        for i, phase in enumerate(phases):

            if i in optimizable_phases:
                configurer.configure_optimizable_phase(tlLogic, phase)
            else:
                configurer.configure_non_optimizable_phase(tlLogic, phase)

    net_xml.write(net_file, pretty_print=True)


def configure_sumo_flow_parameters(scenario):
    global route_xml_copy

    route_file = os.path.abspath(get_route_file_path(scenario))
    route_xml = xml_util.get_xml(route_file)
    route_xml_copy = copy.deepcopy(route_xml)

    flows = route_xml.findall(".//flow")

    configurer = FlowConfigurer()
    configurer.set_scenario(scenario)

    for flow in flows:
        configurer.configure_flow(flow)

    route_xml.write(route_file, pretty_print=True)


def fix_flow_route_association(scenario):

    route_file = os.path.abspath(get_route_file_path(scenario))
    route_xml = xml_util.get_xml(route_file)

    flows = route_xml.findall(".//flow")

    for flow in flows:

        if not flow.get('route'):
            flow_id = flow.get('id')
            flow.set('route', flow_id.replace('flow_', ''))

    route_xml.write(route_file, pretty_print=True)


def reset_sumo_traffic_light_parameters(scenario):
    net_filename = os.path.abspath(get_network_file_path(scenario))

    if net_xml_copy is not None:
        net_xml_copy.write(net_filename, pretty_print=True)


def reset_sumo_flow_parameters(scenario):
    route_filename = os.path.abspath(get_route_file_path(scenario))

    if route_xml_copy is not None:
        route_xml_copy.write(route_filename, pretty_print=True)


def get_intersection_ids(net_xml, sorted_=True):

    intersections = net_xml.findall(".//junction[@type]")

    intersections = [intersection
                     for intersection in intersections
                     if intersection.get('type') != 'dead_end' and
                     intersection.get('type') != 'internal']

    if sorted_:
        intersection_ids = []
        intersection_points = []

        for intersection in intersections:
            intersection_id = intersection.get('id')
            intersection_ids.append(intersection_id)
            intersection_point = Point([float(intersection.get('x')), float(intersection.get('y'))])
            intersection_points.append(intersection_point)

        zipped_id_and_location = zip(intersection_ids, intersection_points)
        sorted_id_and_location = sorted(zipped_id_and_location, key=lambda x: cmp_to_key(location_comparator)(x[1]))

        intersection_ids = list(zip(*sorted_id_and_location))[0]
    else:
        intersection_ids = [intersection.get('id') for intersection in intersections]

    return intersection_ids


def get_intersections_with_traffic_light(net_xml, sorted_=True):
    intersections = net_xml.findall(".//junction[@type]")

    intersections = [intersection
                     for intersection in intersections
                     if intersection.get('type') == 'traffic_light' or
                     intersection.get('type') == 'traffic_light_right_on_red' or
                     intersection.get('type') == 'traffic_light_unregulated']

    if sorted_:
        intersection_ids = []
        intersection_points = []

        for intersection in intersections:
            intersection_id = intersection.get('id')
            intersection_ids.append(intersection_id)
            intersection_point = Point([float(intersection.get('x')), float(intersection.get('y'))])
            intersection_points.append(intersection_point)

        zipped_id_and_location = zip(intersection_ids, intersection_points)
        sorted_id_and_location = sorted(zipped_id_and_location, key=lambda x: cmp_to_key(location_comparator)(x[1]))

        intersection_ids = list(zip(*sorted_id_and_location))[0]
    else:
        intersection_ids = [intersection.get('id') for intersection in intersections]

    return intersection_ids


def get_connections(net_xml, from_edge='ALL', to_edge='ALL'):

    from_attribute = ''
    to_attribute = ''
    if from_edge != 'ALL':
        from_attribute = "[@from='" + from_edge + "']"
    if to_edge != 'ALL':
        to_attribute = "[@to='" + to_edge + "']"

    connections = net_xml.findall(".//connection" + from_attribute + to_attribute)

    edges = net_xml.findall(".//edge[@priority]")
    edge_ids = [edge.get('id') for edge in edges]

    actual_connections = []
    for connection in connections:
        connection_from = connection.get('from')
        connection_to = connection.get('to')

        if connection_from in edge_ids and connection_to in edge_ids:
            actual_connections.append(connection)

    return actual_connections


def location_comparator(p1, p2):

    p1_x, p1_y = p1.coords[0]
    p2_x, p2_y = p2.coords[0]

    if p1_y < p2_y:
        return 1
    if p1_y > p2_y:
        return -1

    if p1_x < p2_x:
        return -1
    if p1_x > p2_x:
        return 1

    return 0


def get_network_border_edges(net_xml):

    dead_end_intersections = net_xml.findall('.//junction[@type="dead_end"]')

    intersection_ids = [intersection.get('id') for intersection in dead_end_intersections]

    junction_to_network_entering_edges_mapping = {}
    junction_to_network_exiting_edges_mapping = {}

    for intersection_id in intersection_ids:

        entering_edges = net_xml.findall('.//edge[@from="' + intersection_id + '"]')
        for entering_edge in entering_edges:
            to_intersection = entering_edge.get('to')

            if to_intersection in junction_to_network_entering_edges_mapping:
                junction_to_network_entering_edges_mapping[to_intersection].append(entering_edge)
            else:
                junction_to_network_entering_edges_mapping[to_intersection] = [entering_edge]

        exiting_edges = net_xml.findall('.//edge[@to="' + intersection_id + '"]')
        for exiting_edge in exiting_edges:
            to_intersection = exiting_edge.get('to')

            if to_intersection in junction_to_network_exiting_edges_mapping:
                junction_to_network_exiting_edges_mapping[to_intersection].append(exiting_edge)
            else:
                junction_to_network_exiting_edges_mapping[to_intersection] = [exiting_edge]

    return junction_to_network_entering_edges_mapping, junction_to_network_exiting_edges_mapping


def sort_edges_by_angle(edges, incoming=True, clockwise=True):

    edges_and_angles = []
    for edge in edges:
        lane = edge[0]
        polyline = lane.get('shape')
        polyline_points = polyline.split()

        first_point = Point(map(float, polyline_points[0].split(',')))
        last_point = Point(map(float, polyline_points[-1].split(',')))

        if incoming:
            first_point, last_point = last_point, first_point

        normalized_point = Point([last_point.x - first_point.x, last_point.y - first_point.y])

        angle = math.atan2(normalized_point.x, normalized_point.y)

        if angle < 0:
            angle += 2 * math.pi

        edges_and_angles.append([edge, angle])

    reverse = not clockwise

    edges_and_angles.sort(key=lambda x: x[1], reverse=reverse)
    angle_sorted_edges = [edge for edge, _ in edges_and_angles]

    return angle_sorted_edges


def get_intersections_incoming_edges(net_xml, intersection_ids='ALL', _sorted=True):

    if intersection_ids == 'ALL':
        intersection_ids = get_intersection_ids(net_xml)

    single_output = False
    if isinstance(intersection_ids, str):
        intersection_ids = [intersection_ids]
        single_output = True

    intersection_incoming_edges = []
    for intersection_id in intersection_ids:

        incoming_edges = set()

        intersection = net_xml.find('.//junction[@id="' + intersection_id + '"]')
        inc_lanes = intersection.get('incLanes').split()

        for lane in inc_lanes:
            edge = net_xml.find('.//edge[@priority]/lane[@id="' + lane + '"]/..')

            if edge is not None:
                incoming_edges.add(edge)

        if _sorted:
            incoming_edges = sort_edges_by_angle(incoming_edges)
        else:
            incoming_edges = list(incoming_edges)

        intersection_incoming_edges.append(incoming_edges)

    if single_output:
        return intersection_incoming_edges[0]
    else:
        return intersection_incoming_edges


def get_intersection_connections(net_xml, intersection_id):

    intersections_incoming_edges = get_intersections_incoming_edges(net_xml, intersection_id)

    connections = []
    for edge in intersections_incoming_edges:
        edge_connections = get_connections(net_xml, from_edge=edge.get('id'))
        connections += edge_connections

    return connections


def get_intersection_edge_ids(net_xml, intersection_ids='ALL', _sorted=True):

    if intersection_ids == 'ALL':
        intersection_ids = get_intersection_ids(net_xml)

    single_output = False
    if isinstance(intersection_ids, str):
        intersection_ids = [intersection_ids]
        single_output = True

    entering_edges = [set() for _ in range(len(intersection_ids))]
    exiting_edges = [set() for _ in range(len(intersection_ids))]

    for intersection_index, intersection_id in enumerate(intersection_ids):

        connections = get_intersection_connections(net_xml, intersection_id)

        for connection in connections:

            connection_from = connection.get('from')
            connection_to = connection.get('to')

            from_edge = net_xml.find('.//edge[@id="' + connection_from + '"]')
            to_edge = net_xml.find('.//edge[@id="' + connection_to + '"]')

            entering_edges[intersection_index].add(from_edge)
            exiting_edges[intersection_index].add(to_edge)

        if _sorted:
            entering_edges[intersection_index] = sort_edges_by_angle(entering_edges[intersection_index])
            exiting_edges[intersection_index] = sort_edges_by_angle(exiting_edges[intersection_index], incoming=False)
        else:
            entering_edges[intersection_index] = list(entering_edges[intersection_index])
            exiting_edges[intersection_index] = list(exiting_edges[intersection_index])

    if single_output:
        return entering_edges[0], exiting_edges[0]
    else:
        return entering_edges, exiting_edges


def get_movement_traffic_light_controller(movement_to_connection):

    movement_to_traffic_light_index_mapping = {}
    for movement, connections in movement_to_connection.items():

        traffic_light_indices = set([])
        for connection in connections:
            traffic_light_index = connection.get('linkIndex')
            traffic_light_indices.add(int(traffic_light_index))

        movement_to_traffic_light_index_mapping[movement] = list(traffic_light_indices)

    return movement_to_traffic_light_index_mapping


def adjusts_intersection_position(junctions, edges, x_spacing=0, y_spacing=0):
    for junction in junctions:
        junction.set('x', str(float(junction.get('x')) + x_spacing))
        junction.set('y', str(float(junction.get('y')) + y_spacing))

        if 'shape' in junction.attrib:
            junction.set('shape', math_util.translate_polyline(junction.get('shape'), x=x_spacing, y=y_spacing))

    for edge in edges:
        for lane in edge:
            lane.set('shape', math_util.translate_polyline(lane.get('shape'), x=x_spacing, y=y_spacing))


def map_connection_direction(connection):
    direction = connection.get('dir').lower()

    if direction == 'l':
        direction = 'left_turn'
    elif direction == 's':
        direction = 'straight'
    elif direction == 'r':
        direction = 'right_turn'

    return direction


def get_average_duration_statistic(output_file):

    with open(output_file, 'r') as handle:
        content = handle.read()

    statistics_begin_index = content.find('Statistics')

    if statistics_begin_index == -1:
        raise Exception('Sumo statistics output is not set')

    statistics_end_index = content.find('DepartDelay:') + content[content.find('DepartDelay:'):].find('\n')

    statistics = content[statistics_begin_index:statistics_end_index]

    duration_begin_index = statistics.find('Duration:') + len('Duration:')
    duration_end_index = duration_begin_index + statistics[duration_begin_index:].find('\n')

    duration = float(statistics[duration_begin_index:duration_end_index])

    return duration


def movement_comparator(m1, m2):
    direction_sort_order = {'L': 0, 'S': 1, 'R': 2}

    m1_0 = int(m1[0])
    m2_0 = int(m2[0])

    if m1_0 > m2_0:
        return 1
    if m1_0 < m2_0:
        return -1

    m1_1 = m1[1]
    m2_1 = m2[1]

    if direction_sort_order[m1_1] > direction_sort_order[m2_1]:
        return 1
    if direction_sort_order[m1_1] < direction_sort_order[m2_1]:
        return -1

    m1_len = len(m1)
    m2_len = len(m2)

    if m1_len < m2_len:
        if m1_1 == "R":
            return 1
        else:
            return -1

    if m1_len > m2_len:
        if m1_1 == "R":
            return -1
        else:
            return 1

    if m1_len > 2:

        m1_end = int(m1[2:])
        m2_end = int(m2[2:])

        if m1_end > m2_end:
            return 1
        if m1_end < m2_end:
            return -1

    return 0


def phase_comparator(p1, p2):

    p1_split = p1.split('_')
    p2_split = p2.split('_')

    p1_split_len = len(p1_split)
    p2_split_len = len(p2_split)

    if p1_split_len < p2_split_len:
        return 1
    if p1_split_len > p2_split_len:
        return -1

    for i in range(0, p1_split_len):
        p1_mi = p1_split[i]
        p2_mi = p2_split[i]

        movement_comparison = movement_comparator(p1_mi, p2_mi)

        if movement_comparison != 0:
            return movement_comparison

    return 0


def detect_movements(net_xml, intersection_ids, is_right_on_red=True):

    intersections_incoming_edges = get_intersections_incoming_edges(net_xml, intersection_ids)

    movement_set = set()
    movements_list = [[] for _ in range(len(intersections_incoming_edges))]
    connection_to_movement = [bidict() for _ in range(len(intersections_incoming_edges))]

    for intersection_index, incoming_edges in enumerate(intersections_incoming_edges):

        for edge_index, edge in enumerate(incoming_edges):

            edge_id = edge.get('id')

            connections = get_connections(net_xml, from_edge=edge_id)

            # new sort order: from left to right
            sorted_connections = list(reversed(connections))

            to_edge_to_direction = bidict()
            for connection in sorted_connections:

                to_edge = connection.get('to')
                direction = connection.get('dir').lower()

                if to_edge in to_edge_to_direction:
                    assert direction == to_edge_to_direction[to_edge]
                else:
                    to_edge_to_direction[to_edge] = direction

            to_edge_to_direction_label = bidict()
            for direction, to_edges in to_edge_to_direction.inverse.items():

                if direction == 'r':
                    to_edges = list(reversed(to_edges))

                for to_edge in to_edges:

                    if len(to_edges) == 1 or to_edges.index(to_edge) == 0:
                        direction_label = direction.upper()
                    else:
                        direction_label = direction.upper() + str(to_edges.index(to_edge))

                    to_edge_to_direction_label[to_edge] = direction_label

            for connection in sorted_connections:

                to_edge = connection.get('to')

                direction_label = to_edge_to_direction_label[to_edge]

                movement = str(edge_index) + direction_label

                if movement not in movements_list[intersection_index]:
                    if not (is_right_on_red and direction_label[0] == 'R'):
                        movements_list[intersection_index].append(movement)
                        movement_set.add(movement)

                connection_to_movement[intersection_index][connection] = movement

    unique_movements = sorted(movement_set, key=cmp_to_key(movement_comparator))

    return unique_movements, movements_list, connection_to_movement


def get_connection_requests(net_xml, intersection_id):

    requests = net_xml.findall('.//junction[@id="' + intersection_id + '"]/request')

    connection_requests = {}
    for request in requests:
        request_index = request.get('index')
        connection_requests[request_index] = request

    return connection_requests


def detect_same_lane_origin_movements(net_xml, intersection_ids, connection_to_movement_list):

    intersections_incoming_edges_list = get_intersections_incoming_edges(net_xml, intersection_ids)

    same_lane_origin_movements_list = [{} for _ in range(len(intersections_incoming_edges_list))]

    for intersection_index, intersection_incoming_edges in enumerate(intersections_incoming_edges_list):

        connection_to_movement = connection_to_movement_list[intersection_index]

        connections = [connection
                       for edge in intersection_incoming_edges
                       for connection in get_connections(net_xml, from_edge=edge.get('id'))]

        same_lane_origin_movements = same_lane_origin_movements_list[intersection_index]

        for index_1, connection_1 in enumerate(connections):
            for index_2, connection_2 in enumerate(connections[index_1 + 1:]):

                movement_1 = connection_to_movement[connection_1]
                movement_2 = connection_to_movement[connection_2]

                if movement_1 == movement_2:
                    continue

                connection_1_from_lane = connection_1.get('from') + '_' + connection_1.get('fromLane')
                connection_2_from_lane = connection_2.get('from') + '_' + connection_2.get('fromLane')

                movements = [movement_1, movement_2]
                for movement in movements:
                    if movement not in same_lane_origin_movements:
                        same_lane_origin_movements[movement] = []

                if connection_1_from_lane == connection_2_from_lane:
                    if movement_2 not in same_lane_origin_movements[movement_1]:
                        same_lane_origin_movements[movement_1].append(movement_2)
                    if movement_1 not in same_lane_origin_movements[movement_2]:
                        same_lane_origin_movements[movement_2].append(movement_1)

        original_same_lane_origin_movements = copy.deepcopy(same_lane_origin_movements)
        for key, values in original_same_lane_origin_movements.items():
            movement_same_lane_origin_movements = set(values)

            unchecked_same_origin_movements = copy.deepcopy(values)
            while unchecked_same_origin_movements:
                movement = unchecked_same_origin_movements.pop(0)
                inherited_same_lane_origin_movements = set(original_same_lane_origin_movements[movement])

                difference = inherited_same_lane_origin_movements.difference(movement_same_lane_origin_movements)
                difference.discard(key)

                movement_same_lane_origin_movements.update(difference)
                unchecked_same_origin_movements.extend(difference)

            same_lane_origin_movements[key] = list(movement_same_lane_origin_movements)

    return same_lane_origin_movements_list


def detect_movements_major_and_minor_links(net_xml, intersection_ids, connection_to_movement_list):

    intersections_incoming_edges_list = get_intersections_incoming_edges(net_xml, intersection_ids)

    link_states_list = [{} for _ in range(len(intersections_incoming_edges_list))]

    for intersection_index, intersection_incoming_edges in enumerate(intersections_incoming_edges_list):

        connection_to_movement = connection_to_movement_list[intersection_index]

        link_states = link_states_list[intersection_index]

        connections = [connection
                       for edge in intersection_incoming_edges
                       for connection in get_connections(net_xml, from_edge=edge.get('id'))]

        intersection_id = intersection_ids[intersection_index]
        internal_lane_connections = get_internal_lane_connections(net_xml, intersection_id)

        for connection in connections:
            movement = connection_to_movement[connection]

            connection_internal_lane = connection.get('via')
            connection_link_state = internal_lane_connections[connection_internal_lane].get('state')

            if movement in link_states:
                assert connection_link_state == link_states[movement]
            else:
                assert connection_link_state == 'M' or connection_link_state == 'm'
                link_states[movement] = connection_link_state

    return link_states_list


def detect_movement_conflicts(net_xml, intersection_ids, connection_to_movement_list, same_lane_origin_movements_list):

    intersections_incoming_edges_list = get_intersections_incoming_edges(net_xml, intersection_ids)

    conflicts_list = [{} for _ in range(len(intersections_incoming_edges_list))]

    for intersection_index, intersection_incoming_edges in enumerate(intersections_incoming_edges_list):

        connection_to_movement = connection_to_movement_list[intersection_index]
        same_lane_origin_movements = same_lane_origin_movements_list[intersection_index]

        conflicts = conflicts_list[intersection_index]

        connections = [connection
                       for edge in intersection_incoming_edges
                       for connection in get_connections(net_xml, from_edge=edge.get('id'))]

        intersection_id = intersection_ids[intersection_index]

        connection_requests = get_connection_requests(net_xml, intersection_id)

        link_index_to_movement = {}
        for connection in connections:
            link_index = connection.get('linkIndex')
            movement = connection_to_movement[connection]
            link_index_to_movement[int(link_index)] = movement

        for connection in connections:

            movement = connection_to_movement[connection]

            movement_link_index = connection.get('linkIndex')
            connection_request = connection_requests[movement_link_index]
            conflict_indicators = connection_request.get('foes')[::-1]

            conflicts[movement] = []
            for link_index, conflict_indicator in enumerate(conflict_indicators):

                if conflict_indicator == '1':
                    other_movement = link_index_to_movement[link_index]

                    if other_movement not in conflicts[movement]:
                        conflicts[movement].append(other_movement)

        original_conflicts = copy.deepcopy(conflicts)
        for key, values in same_lane_origin_movements.items():

            new_conflicts = set([])
            for value in values:

                inherited_conflicts = set(original_conflicts[value])
                new_conflicts.update(inherited_conflicts)

                for inherited_conflict in inherited_conflicts:
                    new_conflicts.update(set(same_lane_origin_movements[inherited_conflict]))

            for new_conflict in new_conflicts:
                new_conflict_set = set(conflicts[new_conflict])
                new_conflict_set.add(key)
                conflicts[new_conflict] = list(new_conflict_set)

            movement_conflicts = set(conflicts[key])
            movement_conflicts.update(new_conflicts)
            conflicts[key] = list(movement_conflicts)

    return conflicts_list


def detect_movements_preferences(net_xml, intersection_ids, connection_to_movement_list):

    intersections_incoming_edges_list = get_intersections_incoming_edges(net_xml, intersection_ids)

    movement_to_give_preference_to_list = [{} for _ in range(len(intersections_incoming_edges_list))]

    for intersection_index, intersection_incoming_edges in enumerate(intersections_incoming_edges_list):

        connection_to_movement = connection_to_movement_list[intersection_index]

        movement_to_give_preference_to = movement_to_give_preference_to_list[intersection_index]

        connections = [connection
                       for edge in intersection_incoming_edges
                       for connection in get_connections(net_xml, from_edge=edge.get('id'))]

        intersection_id = intersection_ids[intersection_index]

        connection_requests = get_connection_requests(net_xml, intersection_id)

        link_index_to_movement = {}
        for connection in connections:
            link_index = connection.get('linkIndex')
            movement = connection_to_movement[connection]
            link_index_to_movement[int(link_index)] = movement

        for connection in connections:

            movement = connection_to_movement[connection]

            movement_link_index = connection.get('linkIndex')
            connection_request = connection_requests[movement_link_index]
            give_preference_to_indicators = connection_request.get('response')[::-1]

            movement_to_give_preference_to[movement] = []
            for link_index, give_preference_to_indicator in enumerate(give_preference_to_indicators):

                if give_preference_to_indicator == '1':
                    other_movement = link_index_to_movement[link_index]

                    if other_movement not in movement_to_give_preference_to[movement]:
                        movement_to_give_preference_to[movement].append(other_movement)

    return movement_to_give_preference_to_list


def detect_minor_conflicts(net_xml, intersection_ids, conflicts_list, link_states_list,
                           movement_to_give_preference_to_list):

    intersections_incoming_edges_list = get_intersections_incoming_edges(net_xml, intersection_ids)

    minor_conflicts_list = [{} for _ in range(len(intersections_incoming_edges_list))]

    for intersection_index, intersection_incoming_edges in enumerate(intersections_incoming_edges_list):
        conflicts = conflicts_list[intersection_index]
        link_states = link_states_list[intersection_index]
        movement_to_give_preference_to = movement_to_give_preference_to_list[intersection_index]

        minor_conflicts = minor_conflicts_list[intersection_index]

        for movement, conflicting_movements in conflicts.items():

            minor_conflicts[movement] = []

            movement_link_state = link_states[movement]

            for conflicting_movement in conflicting_movements:

                conflicting_movement_link_state = link_states[conflicting_movement]

                if movement_link_state != conflicting_movement_link_state:

                    if conflicting_movement in movement_to_give_preference_to[movement] and \
                            movement in movement_to_give_preference_to[conflicting_movement]:
                        pass
                    elif movement_link_state == 'm' and \
                            conflicting_movement in movement_to_give_preference_to[movement]:
                        # minor conflict
                        minor_conflicts[movement].append(conflicting_movement)
                    elif movement_link_state == 'M' and \
                            movement in movement_to_give_preference_to[conflicting_movement]:
                        # minor conflict
                        minor_conflicts[movement].append(conflicting_movement)

    return minor_conflicts_list


def detect_phases(movements_list, conflicts_list, link_states_list, minor_conflicts_list,
                  major_conflicts_only=False, dedicated_minor_links_phases=True):

    phases_final_set = set()
    phases_list = [[] for _ in range(len(movements_list))]

    for intersection_index, movements in enumerate(movements_list):

        original_conflicts = conflicts_list[intersection_index]
        minor_conflicts = minor_conflicts_list[intersection_index]

        if major_conflicts_only:
            conflicts = {}
            for movement, conflicting_movements in original_conflicts.items():
                minor_conflicting_movements = minor_conflicts[movement]
                conflicts[movement] = list(set(conflicting_movements).difference(minor_conflicting_movements))
        else:
            conflicts = original_conflicts

        phases = []

        depth_first_search_tracking = [copy.deepcopy(movements)]
        movements_left_list = [movements]
        elements_tracking = []

        i = [-1]
        while len(depth_first_search_tracking) != 0:

            while len(depth_first_search_tracking[0]) != 0:
                i[0] += 1

                element = depth_first_search_tracking[0].pop(0)
                elements_tracking.append(element)

                movements_left = [movement for movement in movements[i[0]+1:]
                                  if movement not in conflicts[element] and
                                  movement != element and
                                  movement in movements_left_list[-1]]

                movements_left_list.append(movements_left)

                if movements_left:
                    depth_first_search_tracking = [movements_left] + depth_first_search_tracking
                    i = [i[0]] + i
                else:
                    phases.append('_'.join(elements_tracking))
                    elements_tracking.pop()
                    movements_left_list.pop()

            depth_first_search_tracking.pop(0)
            if elements_tracking:
                elements_tracking.pop()
                i.pop(0)
            movements_left_list.pop()

        phase_sets = [set(phase.split('_')) for phase in phases]

        indices_to_remove = set()
        for i in range(0, len(phase_sets)):
            for j in range(i+1, len(phase_sets)):

                phase_i = phase_sets[i]
                phase_j = phase_sets[j]

                if phase_i.issubset(phase_j):
                    indices_to_remove.add(i)
                elif phase_j.issubset(phase_i):
                    indices_to_remove.add(j)

        indices_to_remove = sorted(indices_to_remove, reverse=True)
        for index_to_remove in indices_to_remove:
            phases.pop(index_to_remove)
            phase_sets.pop(index_to_remove)

        if major_conflicts_only and dedicated_minor_links_phases:
            link_states = link_states_list[intersection_index]
            minor_link_phases = []
            for phase in phases:
                phase_movements = phase.split('_')

                minor_phase_movements = []
                for movement_1 in phase_movements:

                    movement_1_link = link_states[movement_1]

                    if movement_1_link == 'M':
                        continue

                    movement_1_conflicts = original_conflicts[movement_1]

                    for movement_2 in phase_movements:
                        if movement_2 in movement_1_conflicts:
                            minor_phase_movements.append(movement_1)
                            break

                minor_link_phase = '_'.join(minor_phase_movements)
                if minor_link_phase:
                    minor_link_phases.append(minor_link_phase)

            phases.extend(minor_link_phases)

        phases_list[intersection_index] = phases

        for phase in phases:
            phases_final_set.add(phase)

    unique_phases = sorted(phases_final_set, key=cmp_to_key(phase_comparator))

    return unique_phases, phases_list


def detect_existing_phases():

    return None, None


def simplify_existing_phases(net_xml, intersection_ids, phases_list):
    return None, None


def build_phase_expansions(unique_movements, unique_phases):

    phase_expansions = {}
    for i, phase in enumerate(unique_phases):
        phase_movements = phase.split("_")
        phase_expansion = [0] * len(unique_movements)

        for phase_movement in phase_movements:
            phase_expansion[unique_movements.index(phase_movement)] = 1

        phase_expansions[i] = phase_expansion

    phase_expansions[-1] = [0] * len(unique_movements)

    return phase_expansions


def match_ordered_movements_to_phases(ordered_movements, phases):

    movements = copy.deepcopy(ordered_movements)
    number_of_movements_per_phase = max(map(lambda x: len(x.split('_')), phases))

    phase_sets = [set(phase.split('_')) for phase in phases]
    phase_sets_tracking = copy.deepcopy(phase_sets)

    selected_phases_indices = []

    done = False
    while not done:

        done = True

        combination_generator = itertools.combinations(movements, number_of_movements_per_phase)

        for phase_movements in combination_generator:

            possible_phase = set(phase_movements)
            if possible_phase in phase_sets_tracking:
                for movement in phase_movements:
                    movements.remove(movement)

                selected_phases_indices.append(phase_sets.index(possible_phase))

                phase_sets_tracking.remove(possible_phase)

                done = False
                break

    i = len(movements)
    while len(movements) > 0:

        combination_generator = itertools.combinations(movements, i)

        for phase_movements in combination_generator:

            possible_phase = set(phase_movements)
            for phase in phase_sets_tracking:
                if possible_phase.issubset(phase):
                    for movement in phase_movements:
                        movements.remove(movement)

                    selected_phases_indices.append(phase_sets.index(phase))
                    
                    phase_sets_tracking.remove(phase)
                    break

        i -= 1

    selected_phases = np.array(phases)[selected_phases_indices]

    return selected_phases


def get_internal_edges(net_xml, intersection_id):

    all_connections_from_lane = {connection.get('from') + '_' + connection.get('fromLane'): connection
                                 for connection in net_xml.findall(".//connection")}

    connections = get_intersection_connections(net_xml, intersection_id)
    internal_edges = []
    for connection in connections:
        via_lane = connection.get('via')

        while via_lane is not None:
            edge = net_xml.find('.//edge[@function="internal"]/lane[@id="' + via_lane + '"]/..')
            internal_edges.append(edge)

            via_lane = all_connections_from_lane[via_lane].get('via')

    return internal_edges


def get_internal_lanes(net_xml, intersection_id):

    all_connections_from_lane = {connection.get('from') + '_' + connection.get('fromLane'): connection
                                 for connection in net_xml.findall(".//connection")}

    connections = get_intersection_connections(net_xml, intersection_id)
    internal_lanes = []
    for connection in connections:
        via_lane = connection.get('via')

        while via_lane is not None:
            lane = net_xml.find('.//edge[@function="internal"]/lane[@id="' + via_lane + '"]')
            internal_lanes.append(lane)

            via_lane = all_connections_from_lane[via_lane].get('via')

    return internal_lanes


def get_internal_lane_connections(net_xml, intersection_id):

    all_connections_from_lane = {connection.get('from') + '_' + connection.get('fromLane'): connection
                                 for connection in net_xml.findall(".//connection")}

    connections = get_intersection_connections(net_xml, intersection_id)
    internal_lanes_connections = {}
    for connection in connections:
        via_lane = connection.get('via')

        while via_lane is not None:
            via_edge = via_lane.rsplit('_', 1)[0]
            connection = net_xml.find('.//connection[@from="' + via_edge + '"]')
            internal_lanes_connections[via_lane] = connection

            via_lane = all_connections_from_lane[via_lane].get('via')

    return internal_lanes_connections


def get_internal_lane_paths(net_xml, intersection_id, internal_lanes):
    
    lanes_by_id = {lane.get('id'): lane for lane in internal_lanes}

    connections = get_intersection_connections(net_xml, intersection_id)

    all_connections_from_lane = {connection.get('from') + '_' + connection.get('fromLane'): connection
                                 for connection in net_xml.findall(".//connection")}

    lane_path = {}
    for connection in connections:
        via_lane = connection.get('via')

        polyline_lane_ids = []
        polylines = []
        while via_lane is not None:

            polyline_lane_ids.append(via_lane)
            polylines.append([])
            
            shape = lanes_by_id[via_lane].get('shape')

            via_lane_polyline = [list(map(float, point.split(','))) for point in shape.split()]
            
            for polyline in polylines:

                if polyline:
                    polyline_extension_start_index = 1
                else:
                    polyline_extension_start_index = 0
                
                polyline.extend(via_lane_polyline[polyline_extension_start_index:])

            via_lane = all_connections_from_lane[via_lane].get('via')

        for index, lane_id in enumerate(polyline_lane_ids):
            lane_path[lane_id] = polylines[index]

    return lane_path


def convert_sumo_angle_to_canonical_angle(sumo_angle):

    if sumo_angle <= 90:
        canonical_angle = 90 - sumo_angle
    elif sumo_angle > 90:
        canonical_angle = 90 - (sumo_angle - 360)

    return canonical_angle


# Sumo 1.7.0 only
def convert_flows_to_trips(route_file):
    
    route_xml = xml_util.get_xml(route_file)
    flows = route_xml.findall('.//flow')

    root = route_xml.getroot()

    rng = np.random.Generator(np.random.MT19937(23423))

    trips = []
    for flow in flows:

        id_ = flow.get('id')

        begin = int(flow.get('begin'))
        end = int(flow.get('end'))

        probability = float(flow.get('probability'))

        from_edge = flow.get('from')
        to_edge = flow.get('to')
        depart_lane = flow.get('departLane')
        depart_speed = flow.get('departSpeed')

        depart_times = np.where(rng.binomial(1, probability, end - begin) == 1)[0]

        vehicle_count = 0

        for depart_time in depart_times:

            vehicle_count += 1

            attributes = {
                'id': id_ + '.' + str(vehicle_count),
                'from': from_edge,
                'to': to_edge,
                'depart': str(depart_time),
                'departLane': depart_lane,
                'departSpeed': depart_speed
            }

            trip = etree.Element('trip', attributes)
            trips.append(trip)

        root.remove(flow)

    trips.sort(key=lambda x: float(x.get('depart')))

    root.extend(trips)

    with open(route_file, 'wb') as handle:
        route_xml.write(handle, pretty_print=True)


# Sumo 1.7.0 only
def fix_save_state_stops(net_xml, save_state, time):

    save_state_xml = xml_util.get_xml(save_state)
    save_state_vehicles = list(set(save_state_xml.findall('.//stop/..')))

    stops_to_issue = []
    for save_state_vehicle in save_state_vehicles:

        stop_elements = save_state_vehicle.findall('.//stop')

        if len(stop_elements) > 1:

            for stop_element in stop_elements[0:-1]:
                save_state_vehicle.remove(stop_element)

        stop_element = stop_elements[-1]

        if stop_element is not None:

            vehicle_id = save_state_vehicle.get('id')
            lane = stop_element.get('lane')
            vehicle_lane_position = save_state_vehicle.get('pos')

            start_pos = stop_element.get('startPos')
            end_pos = stop_element.get('endPos')
            depart = stop_element.get('depart')

            if end_pos is not None:
                if start_pos is not None:
                    start_pos = float(start_pos)
                    end_pos = float(end_pos)

                    if start_pos == end_pos:
                        lane_definition = net_xml.find('.//edge/lane[@id="' + lane + '"]')
                        lane_length = float(lane_definition.get('length'))

                        if end_pos + 0.15 > lane_length:
                            end_pos = lane_length
                            start_pos = lane_length - 0.15
                        else:
                            end_pos += 0.15

                        stop_element.attrib['startPos'] = str(start_pos)
                        stop_element.attrib['endPos'] = str(end_pos)

            elif depart is None:
                actual_arrival = stop_element.get('actualArrival')
                duration = stop_element.get('duration')
                edge, lane_index = stop_element.get('lane').rsplit('_', 1)

                if duration is None and actual_arrival is not None:
                    duration = time - float(actual_arrival)

                if duration is not None:
                    duration = float(duration)

                    stops_to_issue.append(
                        {
                            'vehID': vehicle_id,
                            'edgeID': edge,
                            'pos': vehicle_lane_position,
                            'laneIndex': lane_index,
                            'duration': duration
                        }
                    )

                    save_state_vehicle.remove(stop_element)

    with open(save_state, 'wb') as handle:
        save_state_xml.write(handle, pretty_print=True)

    return stops_to_issue

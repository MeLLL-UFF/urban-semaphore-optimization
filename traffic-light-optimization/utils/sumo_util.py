import os
import ast
import copy
import itertools

import numpy as np
from sumolib import checkBinary
import lxml.etree as etree
from sympy.geometry.line import Point
from sympy.functions.elementary.trigonometric import atan2
from sympy.core.numbers import pi
from sympy import Point2D, Segment2D

from utils.bidict import bidict
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


def configure_sumo_traffic_light_parameters(experiment, traffic_light_parameters={}):

    global net_xml_copy

    configurer = traffic_light_configurer_instances.get(experiment.strategy, None)

    if not configurer:
        return

    configurer.set_parameters(traffic_light_parameters)

    scenario = experiment.scenario
    net_filename = os.path.abspath(get_network_file_path(scenario))

    parser = etree.XMLParser(remove_blank_text=True)

    net_xml = etree.parse(net_filename, parser)
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

    net_xml.write(net_filename, pretty_print=True)


def configure_sumo_flow_parameters(scenario):
    global route_xml_copy

    parser = etree.XMLParser(remove_blank_text=True)

    route_filename = os.path.abspath(get_route_file_path(scenario))
    route_xml = etree.parse(route_filename, parser)
    route_xml_copy = copy.deepcopy(route_xml)

    flows = route_xml.findall(".//flow")

    configurer = FlowConfigurer()
    configurer.set_scenario(scenario)

    for flow in flows:
        configurer.configure_flow(flow)

    route_xml.write(route_filename, pretty_print=True)


def fix_flow_route_association(scenario):

    parser = etree.XMLParser(remove_blank_text=True)

    route_filename = os.path.abspath(get_route_file_path(scenario))

    route_xml = etree.parse(route_filename, parser)

    flows = route_xml.findall(".//flow")

    for flow in flows:

        if not flow.get('route'):
            flow_id = flow.get('id')
            flow.set('route', flow_id.replace('flow_', ''))

    route_xml.write(route_filename, pretty_print=True)


def reset_sumo_traffic_light_parameters(scenario):
    net_filename = os.path.abspath(get_network_file_path(scenario))

    if net_xml_copy is not None:
        net_xml_copy.write(net_filename, pretty_print=True)


def reset_sumo_flow_parameters(scenario):
    route_filename = os.path.abspath(get_route_file_path(scenario))

    if route_xml_copy is not None:
        route_xml_copy.write(route_filename, pretty_print=True)


def get_intersections_ids(net_xml):
    intersections = net_xml.findall(".//junction[@type]")

    intersection_ids = [intersection.get('id')
                        for intersection in intersections
                        if intersection.get('type') != 'dead_end']

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


def sort_edges_by_angle(net_xml, edge_ids, incoming=True, clockwise=True):
    
    all_edges = net_xml.findall(".//edge[@priority]")

    ids_and_angles = []
    for edge in all_edges:

        edge_id = edge.get('id')
        if edge_id in edge_ids:
            lane = edge[0]
            polyline = lane.get('shape')
            polyline_points = polyline.split()

            first_point = Point(polyline_points[0].split(','))
            last_point = Point(polyline_points[-1].split(','))

            if incoming:
                first_point, last_point = last_point, first_point

            normalized_point = last_point - first_point

            angle = atan2(normalized_point.x, normalized_point.y)

            if angle < 0:
                angle += 2 * pi

            ids_and_angles.append([edge_id, angle])

    reverse = not clockwise

    ids_and_angles.sort(key=lambda x: x[1], reverse=reverse)

    angle_sorted_ids = [_id for _id, angle in ids_and_angles]

    return angle_sorted_ids


def get_intersection_edge_ids(net_xml, from_edge='ALL', to_edge='ALL', _sorted=True):

    connections = get_connections(net_xml, from_edge=from_edge, to_edge=to_edge)

    incoming_edges = set()
    outgoing_edges = set()

    for connection in connections:
        connection_from = connection.get('from')
        connection_to = connection.get('to')

        incoming_edges.add(connection_from)
        outgoing_edges.add(connection_to)

    if _sorted:
        incoming_edges = sort_edges_by_angle(net_xml, incoming_edges)
        outgoing_edges = sort_edges_by_angle(net_xml, outgoing_edges, incoming=False)
    else:
        incoming_edges = list(incoming_edges)
        outgoing_edges = list(outgoing_edges)

    return incoming_edges, outgoing_edges


def get_lane_traffic_light_controller(net_xml, lanes_ids):

    connections = get_connections(net_xml)

    lane_to_traffic_light_index_mapping = {}
    for connection in connections:
        from_edge = connection.get('from')
        from_lane = connection.get('fromLane')

        lane_id = from_edge + '_' + from_lane
        if lane_id in lanes_ids:
            traffic_light_index = connection.get('linkIndex')
            lane_to_traffic_light_index_mapping[lane_id] = traffic_light_index

    return lane_to_traffic_light_index_mapping


def translate_polyline(polyline, x=0, y=0):

    polyline_points = polyline.split()

    translated_polyline = []
    for point in polyline_points:
        coordinates = point.split(',')
        coordinates[0] = str(float(coordinates[0]) + x)
        coordinates[1] = str(float(coordinates[1]) + y)
        translated_polyline.append(coordinates[0] + ',' + coordinates[1])

    return ' '.join(translated_polyline)


def adjusts_intersection_position(junctions, edges, x_spacing=0, y_spacing=0):
    for junction in junctions:
        junction.set('x', str(float(junction.get('x')) + x_spacing))
        junction.set('y', str(float(junction.get('y')) + y_spacing))

        if 'shape' in junction.attrib:
            junction.set('shape', translate_polyline(junction.get('shape'), x=x_spacing, y=y_spacing))

    for edge in edges:
        for lane in edge:
            lane.set('shape', translate_polyline(lane.get('shape'), x=x_spacing, y=y_spacing))


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


def detect_movements(net_xml, use_sumo_directions=False, is_right_on_red=True):

    incoming_edges, _ = get_intersection_edge_ids(net_xml)

    movement_to_connection = bidict()

    movements = []
    for edge_index, edge in enumerate(incoming_edges):

        connections = get_connections(net_xml, from_edge=edge)

        sorted_connections = list(reversed(connections))

        if use_sumo_directions:
            direction_to_from_lane = {}
            for connection in sorted_connections:

                from_lane = connection.get('fromLane')
                direction = connection.get('dir').lower()
                if direction in direction_to_from_lane:
                    direction_to_from_lane[direction].append(from_lane)
                else:
                    direction_to_from_lane[direction] = [from_lane]

            for connection in sorted_connections:

                direction = connection.get('dir').lower()
                from_lane = connection.get('fromLane')

                direction_from_lane = direction_to_from_lane[direction]
                if len(direction_from_lane) == 1:
                    direction_label = direction.upper()
                else:
                    direction_label = direction.upper() + str(direction_to_from_lane[dir].index(from_lane) + 1)

                movement = str(edge_index) + direction_label

                movements.append(movement)

                movement_to_connection[movement] = connection

        else:
            direction_labels = [None]*len(sorted_connections)
            if sorted_connections[0].get('dir').lower() == 'l':
                direction_labels[0] = 'L'
            if sorted_connections[len(sorted_connections) - 1].get('dir').lower() == 'r':
                direction_labels[len(sorted_connections) - 1] = 'R'
            count = 0
            for index, direction_label in enumerate(direction_labels):
                if direction_label is None:
                    if count == 0:
                        direction_labels[index] = 'S'
                    else:
                        direction_labels[index] = 'S' + str(count)
                    count += 1

            for index, connection in enumerate(sorted_connections):

                movement = str(edge_index) + direction_labels[index]

                if is_right_on_red and direction_labels[index] != 'R':
                    movements.append(movement)

                movement_to_connection[movement] = connection

    return movements, movement_to_connection


def detect_movement_conflicts(net_xml, movement_to_connection):

    conflicts = {}

    incoming_edges, outgoing_edges = get_intersection_edge_ids(net_xml)
    connections = get_connections(net_xml)

    all_edges = incoming_edges + outgoing_edges

    intersection_id = get_intersections_ids(net_xml)[0]
    intersection = net_xml.find(".//junction[@id='" + intersection_id + "']")
    intersection_point = Point2D(intersection.get('x'), intersection.get('y'))

    lane_to_movement_point = {}

    for edge in all_edges:

        lanes = net_xml.findall(".//edge[@id='" + edge + "']/lane")

        for lane in lanes:

            lane_id = lane.get('id')

            lane_points = lane.get('shape').split()
            first_lane_point = Point2D(lane_points[0].split(','))
            last_lane_point = Point2D(lane_points[-1].split(','))

            if intersection_point.distance(first_lane_point) < intersection_point.distance(last_lane_point):
                movement_lane_point = first_lane_point
            else:
                movement_lane_point = last_lane_point

            lane_to_movement_point[lane_id] = movement_lane_point

    same_lane_origin_movements = {}
    for index_1 in range(0, len(connections)):
        for index_2 in range(index_1 + 1, len(connections)):

            connection_1 = connections[index_1]
            connection_2 = connections[index_2]

            connection_1_from_lane = connection_1.get('from') + '_' + connection_1.get('fromLane')
            connection_1_to_lane = connection_1.get('to') + '_' + connection_1.get('toLane')

            connection_2_from_lane = connection_2.get('from') + '_' + connection_2.get('fromLane')
            connection_2_to_lane = connection_2.get('to') + '_' + connection_2.get('toLane')

            connection_1_line = \
                Segment2D(lane_to_movement_point[connection_1_from_lane], lane_to_movement_point[connection_1_to_lane])

            connection_2_line = \
                Segment2D(lane_to_movement_point[connection_2_from_lane], lane_to_movement_point[connection_2_to_lane])

            line_intersections = connection_1_line.intersection(connection_2_line)

            movement_1 = movement_to_connection.inverse[connection_1][0]
            movement_2 = movement_to_connection.inverse[connection_2][0]
            if connection_1_line.p1 == connection_2_line.p1:
                if movement_1 in same_lane_origin_movements:
                    same_lane_origin_movements[movement_1].append(movement_2)
                else:
                    same_lane_origin_movements[movement_1] = [movement_2]

                if movement_2 in same_lane_origin_movements:
                    same_lane_origin_movements[movement_2].append(movement_1)
                else:
                    same_lane_origin_movements[movement_2] = [movement_1]

            elif len(line_intersections) > 0:
                if movement_1 in conflicts:
                    conflicts[movement_1].append(movement_2)
                else:
                    conflicts[movement_1] = [movement_2]

                if movement_2 in conflicts:
                    conflicts[movement_2].append(movement_1)
                else:
                    conflicts[movement_2] = [movement_1]
            else:
                if movement_1 not in conflicts:
                    conflicts[movement_1] = []

                if movement_2 not in conflicts:
                    conflicts[movement_2] = []

    for key, values in same_lane_origin_movements.items():
        original_conflicts = set(conflicts[key])

        for value in values:

            inherited_conflicts = conflicts[value]

            original_conflicts.update(set(inherited_conflicts))

            for inherited_conflict in inherited_conflicts:
                original_conflicts.update(set(same_lane_origin_movements[inherited_conflict]))

        conflicts[key] = list(original_conflicts)

    return conflicts


def detect_phases(movements, conflicts, is_right_on_red=True):

    if is_right_on_red:
        movements = [movement for movement in movements if 'R' not in movement]

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
                              if movement not in conflicts[element] + [element] and
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

    return phases


def build_phase_expansions(movements, phases):

    phase_expansion = {}
    for i, phase in enumerate(phases):
        phase_movements = phase.split("_")
        zeros = [0] * len(movements)

        for phase_movement in phase_movements:
            zeros[movements.index(phase_movement)] = 1

        phase_expansion[i + 1] = zeros

    phase_expansion[-1] = [0] * len(movements)

    return phase_expansion

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

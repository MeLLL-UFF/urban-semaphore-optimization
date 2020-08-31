import os
import ast
import copy
import optparse

from sumolib import checkBinary
import lxml.etree as etree
from sympy.geometry.line import Point
from sympy.functions.elementary.trigonometric import atan2
from sympy.core.numbers import pi

from city.flow.configurer.flow_configurer import FlowConfigurer
from city.traffic_light_system.traffic_light.configurer.traffic_light_configurer_factory import traffic_light_configurer_instances
from definitions import get_network_file_path, get_route_file_path


def get_sumo_binary(gui=False):
    options = _get_options()

    if gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    return sumoBinary

def _get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=True, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


net_xml_copy = None

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

    tlLogics = net_xml.findall(".//tlLogic")

    for tlLogic in tlLogics:

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

def reset_sumo_traffic_light_parameters(scenario):

    net_filename = os.path.abspath(get_network_file_path(scenario))

    net_xml_copy.write(net_filename, pretty_print=True)


route_xml_copy = None

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

def reset_sumo_flow_parameters(scenario):

    route_filename = os.path.abspath(get_route_file_path(scenario))

    route_xml_copy.write(route_filename, pretty_print=True)

def get_intersections_ids(net_xml):
    intersections = net_xml.findall(".//junction[@type]")

    intersection_ids = [intersection.get('id') for intersection in intersections if intersection.get('type') != 'dead_end']

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
                angle += 2*pi

            ids_and_angles.append([edge_id, angle])

    reverse = not clockwise

    ids_and_angles.sort(key=lambda x: x[1], reverse=reverse)

    angle_sorted_ids = [id for id, angle in ids_and_angles]

    return angle_sorted_ids

def get_intersection_edge_ids(net_xml, from_edge='ALL', to_edge='ALL', sorted=True):

    connections = get_connections(net_xml, from_edge=from_edge, to_edge=to_edge)

    incoming_edges = set()
    outgoing_edges = set()

    for connection in connections:
        connection_from = connection.get('from')
        connection_to = connection.get('to')

        incoming_edges.add(connection_from)
        outgoing_edges.add(connection_to)

    if sorted:
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

import os
import copy
import sys

sys.path.append('traffic-light-optimization')

import lxml.etree as etree

import definitions
from utils.traffic_util import generate_all_traffic_level_configurations
from utils.sumo_util import get_intersection_edge_ids, adjusts_intersection_position
from utils.xml_util import rename_xml_string


test_i_folder = definitions.ROOT_DIR + '/scenario/test_i/'
test_iii_folder = definitions.ROOT_DIR + '/scenario/test_iii/'

experiment_iii_network_name = 'disconnected_intersections'


def _build_experiment_iii_network(type='regular'):

    parser = etree.XMLParser(remove_blank_text=True)

    test_i_scenarios = os.listdir(test_i_folder)

    net_attributes = {
        'version': '1.3',
        'junctionCornerDetail': '5',
        'limitTurnSpeed': '5.50'
    }

    root = etree.Element('net', net_attributes)

    SPACING = 100

    x_spacing = 0

    merged_net_x1 = 0
    merged_net_y1 = 0
    merged_net_x2 = 0
    merged_net_y2 = 0

    for index, scenario in enumerate(test_i_scenarios):

        name = scenario

        net_xml = etree.parse(test_i_folder + scenario + '/' + name + '__' + type + '.net.xml', parser)

        edges = net_xml.findall(".//edge")
        tlLogics = net_xml.findall(".//tlLogic")
        junctions = net_xml.findall(".//junction")
        connections = net_xml.findall(".//connection")

        location_tag = net_xml.getroot()[0]
        net_boundary = location_tag.attrib['convBoundary'].split(',')
        x1 = float(net_boundary[0])
        y1 = float(net_boundary[1])
        x2 = float(net_boundary[2])
        y2 = float(net_boundary[3])

        adjusts_intersection_position(junctions, edges, x_spacing=x_spacing - x1, y_spacing=-y2)

        x_spacing += x2 - x1 + SPACING


        number_of_incoming_streets = len(get_intersection_edge_ids(net_xml)[0])
        traffic_level_configurations_generator = generate_all_traffic_level_configurations(
            number_of_incoming_streets)
        number_of_traffic_level_configurations = len(list(traffic_level_configurations_generator))


        y_spacing = 0

        for i in range(number_of_traffic_level_configurations):

            adjusts_intersection_position(junctions, edges, y_spacing=-y_spacing)

            if y_spacing == 0:
                y_spacing = y2 - y1 + SPACING

            elements = []
            elements.extend(copy.deepcopy(edges))
            elements.extend(copy.deepcopy(tlLogics))
            elements.extend(copy.deepcopy(junctions))
            elements.extend(copy.deepcopy(connections))

            temporary_root = etree.Element('routes')
            temporary_root.extend(elements)

            old_id_prefix = 'gne'
            new_id_prefix = name + '_' + str(i) + '_' + 'gne'
            renamed_temporary_root = rename_xml_string(temporary_root, old_id_prefix, new_id_prefix, parser)

            root.append(etree.Comment('Extracted from ' + name + '__' + type + ' iteration: ' + str(i)))
            root.extend(renamed_temporary_root)

        merged_net_x2 = max(merged_net_x2, x_spacing - SPACING)
        merged_net_y2 = min(merged_net_y2, -(number_of_traffic_level_configurations * y_spacing - SPACING))

    location_attributes = {
        'netOffset': "0.00,0.00",
        'convBoundary': f'{merged_net_x1},{merged_net_y1},{merged_net_x2},{merged_net_y2}',
        'origBoundary': '-10000000000.00,-10000000000.00,10000000000.00,10000000000.00',
        'projParameter': '!'
    }

    location_element = etree.Element('location', location_attributes)
    root.insert(0, location_element)

    if not os.path.isdir(test_iii_folder):
        os.makedirs(test_iii_folder)

    merged_net_xml = etree.ElementTree(root)

    with open(test_iii_folder + experiment_iii_network_name + '__' + type + '.net.xml', 'wb') as handle:
        merged_net_xml.write(handle, pretty_print=True)

def _build_experiment_iii_routes():

    parser = etree.XMLParser(remove_blank_text=True)

    test_i_scenarios = os.listdir(test_i_folder)

    root = etree.Element('routes')

    for index, scenario in enumerate(test_i_scenarios):

        name = scenario

        net_xml = etree.parse(test_i_folder + scenario + '/' + name + '.net.xml', parser)
        route_xml = etree.parse(test_i_folder + scenario + '/' + name + '.rou.xml', parser)

        flows = route_xml.findall(".//flow")

        number_of_incoming_streets = len(get_intersection_edge_ids(net_xml)[0])
        traffic_level_configurations_generator = generate_all_traffic_level_configurations(
            number_of_incoming_streets)
        number_of_traffic_level_configurations = len(list(traffic_level_configurations_generator))

        for i in range(number_of_traffic_level_configurations):

            elements = []
            elements.extend(copy.deepcopy(flows))

            temporary_root = etree.Element('routes')
            temporary_root.extend(elements)

            old_id_prefix = 'gne'
            new_id_prefix = name + '_' + str(i) + '_' + 'gne'
            renamed_temporary_root = rename_xml_string(temporary_root, old_id_prefix, new_id_prefix, parser)

            root.append(etree.Comment('Extracted from ' + name + ' iteration: ' + str(i)))
            root.extend(renamed_temporary_root)

    if not os.path.isdir(test_iii_folder):
        os.makedirs(test_iii_folder)

    merged_net_xml = etree.ElementTree(root)

    with open(test_iii_folder + experiment_iii_network_name + '.rou.xml', 'wb') as handle:
        merged_net_xml.write(handle, pretty_print=True)


_build_experiment_iii_network('regular')
_build_experiment_iii_network('right_on_red')
_build_experiment_iii_network('unregulated')
_build_experiment_iii_routes()
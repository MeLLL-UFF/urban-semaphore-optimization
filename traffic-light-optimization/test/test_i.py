#print('Enabling attach')
#import ptvsd
#ptvsd.enable_attach(address=('0.0.0.0', 5678))
#print('Waiting for attach')
#ptvsd.wait_for_attach()

import os
import sys
import time

sys.path.append('traffic-light-optimization')

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import lxml.etree as etree

from algorithm.frap.frap import Frap
from algorithm.sumo_based.sumo_based import SumoBased
import definitions
from utils.traffic_util import generate_unique_traffic_level_configurations
from utils.sumo_util import get_intersection_edge_ids, get_connections, map_connection_direction, \
    sort_edges_by_angle
from utils.process_util import NoDaemonPool
from config import Config as config


car_turning_policy_dict = {'left_turn': 0.1, 'straight': 0.6, 'right_turn': 0.3}

traffic_level_mapping = {
    'light': 0.3,
    'medium': 0.5,
    'heavy': 0.7
}

test_i_folder = definitions.ROOT_DIR + config.SCENARIO_PATH + '/test_i'

NUMBER_OF_PROCESSES = 4

def _build_experiment_i_routes():

    parser = etree.XMLParser(remove_blank_text=True)

    test_i_scenarios = os.listdir(test_i_folder)

    for scenario in test_i_scenarios:

        name = scenario
        _type = 'regular'

        scenario_folder = test_i_folder + '/' + scenario

        net_xml = etree.parse(scenario_folder + '/' + name + '__' + _type + '.net.xml', parser)

        connections = get_connections(net_xml)

        from_to_edge_mapping = {}
        from_edge_connection_mapping = {}
        from_edge_lane_used_by_mapping = {}
        for connection in connections:
            from_edge = connection.attrib['from']
            to_edge = connection.attrib['to']

            if from_edge in from_to_edge_mapping:
                from_to_edge_mapping[from_edge].append(to_edge)
            else:
                from_to_edge_mapping[from_edge] = [to_edge]

            from_edge_connection_mapping[from_edge + '__' + to_edge] = connection

            from_edge_lane = connection.attrib['fromLane']
            if from_edge + '__' + from_edge_lane in from_edge_lane_used_by_mapping:
                from_edge_lane_used_by_mapping[from_edge + '__' + from_edge_lane] += 1
            else:
                from_edge_lane_used_by_mapping[from_edge + '__' + from_edge_lane] = 1

        incoming_edge_ids, _ = get_intersection_edge_ids(net_xml)
        number_of_incoming_streets = len(incoming_edge_ids)
        traffic_level_configurations_generator = generate_unique_traffic_level_configurations(
            number_of_incoming_streets)

        clockwise_sorted_edges = sort_edges_by_angle(net_xml, incoming_edge_ids)
        traffic_level_configuration = next(traffic_level_configurations_generator)

        edge_traffic_levels = dict(zip(clockwise_sorted_edges, traffic_level_configuration))

        root = etree.Element('routes')

        begin = 0
        end = 3600

        for from_edge, to_edges in from_to_edge_mapping.items():

            for to_edge in to_edges:

                connection = from_edge_connection_mapping[from_edge + '__' + to_edge]
                direction = map_connection_direction(connection)
                edge_traffic_level = edge_traffic_levels[from_edge]
                traffic_level_value = traffic_level_mapping[edge_traffic_level]
                car_turning_policy = car_turning_policy_dict[direction]

                from_edge_lane = connection.attrib['fromLane']
                lane_used_by = from_edge_lane_used_by_mapping[from_edge + '__' + from_edge_lane]

                attributes = {
                    'id': from_edge + '__' + to_edge + '__' + edge_traffic_level,
                    'from': from_edge,
                    'to': to_edge,
                    'begin': str(begin),
                    'end': str(end),
                    'probability': str((traffic_level_value * car_turning_policy)/lane_used_by),
                    'departLane': 'best',
                    'departSpeed': 'max'
                }

                flow_element = etree.Element('flow', attributes)
                root.append(flow_element)

        routes_xml = etree.ElementTree(root)

        with open(scenario_folder + '/' + name + '.rou.xml', 'wb') as handle:
            routes_xml.write(handle, pretty_print=True)

def _configure_scenario_routes(scenario, traffic_level_configuration):

    parser = etree.XMLParser(remove_blank_text=True)

    name = scenario
    _type = 'regular'

    scenario_folder = test_i_folder + '/' + scenario

    net_xml = etree.parse(scenario_folder + '/' + name + '__' + _type + '.net.xml', parser)

    connections = get_connections(net_xml)

    from_edge_connection_mapping = {}
    from_edge_lane_used_by_mapping = {}
    for connection in connections:
        from_edge = connection.attrib['from']
        to_edge = connection.attrib['to']

        from_edge_connection_mapping[from_edge + '__' + to_edge] = connection

        from_edge_lane = connection.attrib['fromLane']
        if from_edge + '__' + from_edge_lane in from_edge_lane_used_by_mapping:
            from_edge_lane_used_by_mapping[from_edge + '__' + from_edge_lane] += 1
        else:
            from_edge_lane_used_by_mapping[from_edge + '__' + from_edge_lane] = 1

    incoming_edge_ids, _ = get_intersection_edge_ids(net_xml)
    clockwise_sorted_edges = sort_edges_by_angle(net_xml, incoming_edge_ids)

    edge_traffic_levels = dict(zip(clockwise_sorted_edges, traffic_level_configuration))

    routes_xml = etree.parse(scenario_folder + '/' + name + '.rou.xml', parser)
    root = routes_xml.getroot()

    flows = list(root)

    for flow in flows:

        from_edge = flow.attrib['from']
        to_edge = flow.attrib['to']

        connection = from_edge_connection_mapping[from_edge + '__' + to_edge]
        direction = map_connection_direction(connection)
        edge_traffic_level = edge_traffic_levels[from_edge]
        traffic_level_value = traffic_level_mapping[edge_traffic_level]
        car_turning_policy = car_turning_policy_dict[direction]

        from_edge_lane = connection.attrib['fromLane']
        lane_used_by = from_edge_lane_used_by_mapping[from_edge + '__' + from_edge_lane]

        flow.attrib['id'] = from_edge + '__' + to_edge + '__' + edge_traffic_level
        flow.attrib['probability'] = str((traffic_level_value * car_turning_policy)/lane_used_by)

    temporary_route_folder = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/'
    if not os.path.isdir(temporary_route_folder):
        os.makedirs(temporary_route_folder)

    route_file_path = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/' + name + '_' + \
                      '_'.join(traffic_level_configuration) + '.rou.xml'

    with open(route_file_path, 'wb') as handle:
        routes_xml.write(handle, pretty_print=True)

    return route_file_path

def create_experiment_generator(_type='regular', algorithm=None):
    # _type : regular, right_on_red, unregulated

    parser = etree.XMLParser(remove_blank_text=True)

    test_i_scenarios = sorted(next(os.walk(test_i_folder))[1])

    for scenario in test_i_scenarios:

        scenario_folder = test_i_folder + '/' + scenario

        net_xml = etree.parse(scenario_folder + '/' + scenario + '__' + _type + '.net.xml', parser)

        incoming_edge_ids, _ = get_intersection_edge_ids(net_xml)
        traffic_level_configurations = generate_unique_traffic_level_configurations(
            len(incoming_edge_ids))

        net_file = scenario_folder + '/' + scenario + '__' + _type + '.net.xml'
        sumocfg_file = scenario_folder + '/' + scenario + '__' + _type + '.sumocfg'
        base_output_folder = scenario_folder + '/' + 'output' + '/' + str(algorithm) + '/' + _type + '/'

        for traffic_level_configuration in traffic_level_configurations:

            experiment_name = scenario + '__' + _type + '__' + '_'.join(traffic_level_configuration)

            output_folder = base_output_folder + experiment_name + '/'

            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

            yield scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder, \
                  sumocfg_file, output_folder

def run_experiment(arguments):

    scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder, sumocfg_file, \
        output_folder = arguments

    route_file = _configure_scenario_routes(scenario, traffic_level_configuration)
    #route_file = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/' + scenario + '_' + '_'.join(traffic_level_configuration) + '.rou.xml'
    output_file = output_folder + experiment_name + '.out.txt'

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if algorithm == 'FRAP':

        begin = time.time()

        Frap.run(net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

        end = time.time()
        timing = end - begin

        with open(scenario_folder + '/' + 'frap_timing.txt', 'a') as handle:
            handle.write(str(timing))
    else:

        sumo = SumoBased(net_file, route_file, output_file, scenario, _type, traffic_level_configuration)
        sumo.run()

    sys.stdout.flush()

    os.remove(route_file)

def continue_experiment(arguments):

    scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder, sumocfg_file, \
        output_folder, experiment = arguments

    route_file = _configure_scenario_routes(scenario, traffic_level_configuration)
    #route_file = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/' + scenario + '_' + '_'.join(traffic_level_configuration) + '.rou.xml'
    output_file = output_folder + experiment_name + '.out.txt'

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if algorithm == 'FRAP':
        Frap.continue_(experiment, net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

    else:
        raise ValueError('please specify algorithm that can be continued')

    sys.stdout.flush()

    os.remove(route_file)

def _run(_type='regular', algorithm=None, experiment=None):
    
    experiment_generator = create_experiment_generator(_type=_type, algorithm=algorithm)

    global NUMBER_OF_PROCESSES

    if algorithm == 'FRAP':
        NUMBER_OF_PROCESSES = 4
    else:
        NUMBER_OF_PROCESSES = 32

    if experiment is None:

        with NoDaemonPool(processes=NUMBER_OF_PROCESSES) as pool:
            pool.map(run_experiment, experiment_generator)

    else:
        experiment_arguments = [
            (scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder,  
                sumocfg_file, output_folder, experiment)]

        with NoDaemonPool(processes=NUMBER_OF_PROCESSES) as pool:
            pool.map(continue_experiment, experiment_arguments)

def run():
    #'OFF', STATIC, and FRAP
    #_run(_type='unregulated')
    #_run(_type='right_on_red')
    _run(_type='right_on_red', algorithm='FRAP')


if __name__ == "__main__":
    #_build_experiment_i_routes()
    run()

    #_run(_type='right_on_red')
    #_run(_type='unregulated')

    '''
    Frap.summary('0_regular-intersection__right_on_red__custom_4_street_traffic___10_06_15_35_28_10__a05adcb1-1e85-4639-bffa-5f97e40f92a9',
                 plots='summary_only', baseline_comparison=True, scenario='0_regular-intersection',
                 traffic_level_configuration='custom_4_street_traffic')

    Frap.summary('0_regular-intersection__right_on_red__custom_4_street_traffic___10_06_15_35_28_10__a05adcb1-1e85-4639-bffa-5f97e40f92a9',
                 plots='records_only', _round=265, baseline_comparison=True, scenario='0_regular-intersection',
                 traffic_level_configuration='custom_4_street_traffic')
    '''

    '''
    _round = 'worst_time_loss'
    #_round = 0

    Frap.visualize_policy_behavior(
        scenario='0_regular-intersection', _type='right_on_red', traffic_level_configuration='custom_4_street_traffic',
        experiment=experiment, _round=_round)
    '''
    '''
    for _dir, folders, files in os.walk('/home/marcelo/code/urban-semaphore-optimization/scenario/test_i/0_regular-intersection/output/FRAP/right_on_red/'):

        for folder in folders:
            if '___' not in folder:
                continue
            
            Frap._consolidate_output_file(_dir + '/' + folder, folder)
    '''
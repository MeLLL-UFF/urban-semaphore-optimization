#print('Enabling attach')
#import ptvsd
#ptvsd.enable_attach(address=('0.0.0.0', 5678))
#print('Waiting for attach')
#ptvsd.wait_for_attach()

import os
import sys
import subprocess

sys.path.append('traffic-light-optimization')

from algorithm.frap_pub import synchronization_util

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import lxml.etree as etree
import sumolib

from algorithm.experiment import Experiment
import definitions
from utils.traffic_util import generate_unique_traffic_level_configurations
from utils import sumo_util, xml_util
from utils.process_util import NoDaemonPool
from config import Config as config


car_turning_policy_dict = {'left_turn': 0.1, 'straight': 0.6, 'right_turn': 0.3}

traffic_level_mapping = {
    'light': 0.3,
    'heavy': 0.7
}

#test_i_folder = definitions.ROOT_DIR + config.SCENARIO_PATH + '/experimental/Bologna_small-0.29.0'

#test_i_folder = definitions.ROOT_DIR + config.SCENARIO_PATH + '/experimental'
test_i_folder = definitions.ROOT_DIR + config.SCENARIO_PATH + '/test_i'

NUMBER_OF_PROCESSES = 4


def _build_experiment_i_routes():

    test_i_scenarios = os.listdir(test_i_folder)

    for scenario in test_i_scenarios:

        name = scenario
        _type = 'right_on_red'

        scenario_folder = test_i_folder + '/' + scenario

        net_file = scenario_folder + '/' + name + '__' + _type + '.net.xml'
        net_xml = xml_util.get_xml(net_file)

        intersection_ids = sumo_util.get_traffic_lights(net_xml)
        junction_to_network_incoming_edges_mapping, _ = sumo_util.get_network_border_edges(net_xml)

        sorted_edges_id = []
        for intersection_id in intersection_ids:
            edges = junction_to_network_incoming_edges_mapping[intersection_id]
            sorted_edges_partial = sumo_util.sort_edges_by_angle(edges)

            for edge in sorted_edges_partial:
                sorted_edges_id.append(edge.get('id'))

        traffic_level_configurations_generator = generate_unique_traffic_level_configurations(
            len(sorted_edges_id))
        traffic_level_configuration = next(traffic_level_configurations_generator)

        edge_traffic_levels = dict(zip(sorted_edges_id, traffic_level_configuration))

        root = etree.Element('routes')

        begin = 0
        end = 3600

        for from_edge_id in sorted_edges_id:

            edge_traffic_level = edge_traffic_levels[from_edge_id]
            traffic_level_value = traffic_level_mapping[edge_traffic_level]

            attributes = {
                'id': from_edge_id + '__' + edge_traffic_level,
                'from': from_edge_id,
                'begin': str(begin),
                'end': str(end),
                'probability': str(traffic_level_value),
                'departLane': 'best',
                'departSpeed': 'max'
            }

            flow_element = etree.Element('flow', attributes)
            root.append(flow_element)

        routes_xml = etree.ElementTree(root)

        base_route_file = scenario_folder + '/' + name + '.base.rou.xml'

        with open(base_route_file, 'wb') as handle:
            routes_xml.write(handle, pretty_print=True)

        turn_default = ','.join([
            str(car_turning_policy_dict['right_turn']),
            str(car_turning_policy_dict['straight']),
            str(car_turning_policy_dict['left_turn']),
            '0'  # necessary padding
        ])

        route_file = scenario_folder + '/' + name + '.rou.xml'

        JTRROUTER = sumolib.checkBinary('jtrrouter')
        args = [JTRROUTER,
                '-n', net_file,
                '--turn-defaults', turn_default,
                '--route-files', base_route_file,
                '--accept-all-destinations',
                '--max-edges-factor', str(0.5),
                '-o', route_file]

        subprocess.call(args)


def _configure_scenario_routes(scenario, traffic_level_configuration):

    name = scenario
    _type = 'regular'

    scenario_folder = test_i_folder + '/' + scenario

    net_file = scenario_folder + '/' + name + '__' + _type + '.net.xml'
    net_xml = xml_util.get_xml(net_file)

    intersection_ids = sumo_util.get_traffic_lights(net_xml)
    junction_to_network_incoming_edges_mapping, _ = sumo_util.get_network_border_edges(net_xml)

    sorted_edges_id = []
    for intersection_id in intersection_ids:
        edges = junction_to_network_incoming_edges_mapping[intersection_id]
        sorted_edges_partial = sumo_util.sort_edges_by_angle(edges)

        for edge in sorted_edges_partial:
            sorted_edges_id.append(edge.get('id'))

    traffic_level_configurations_generator = generate_unique_traffic_level_configurations(
        len(sorted_edges_id))
    traffic_level_configuration = next(traffic_level_configurations_generator)

    edge_traffic_levels = dict(zip(sorted_edges_id, traffic_level_configuration))

    routes_file = scenario_folder + '/' + name + '.rou.xml'
    routes_xml = xml_util.get_xml(routes_file)
    root = routes_xml.getroot()

    flows = list(root)

    for flow in flows:

        from_edge_id = flow.get('from')

        edge_traffic_level = edge_traffic_levels[from_edge_id]
        traffic_level_value = traffic_level_mapping[edge_traffic_level]

        flow.attrib['id'] = from_edge_id + '__' + edge_traffic_level
        flow.attrib['probability'] = str(traffic_level_value)

    temporary_route_folder = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/'
    if not os.path.isdir(temporary_route_folder):
        os.makedirs(temporary_route_folder)

    base_route_file = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/' + name + '_' + \
                      '_'.join(traffic_level_configuration) + 'base.rou.xml'

    with open(base_route_file, 'wb') as handle:
        routes_xml.write(handle, pretty_print=True)

    turn_default = ','.join([
        str(int(car_turning_policy_dict['right_turn'] * 100)),
        str(int(car_turning_policy_dict['straight'] * 100)),
        str(int(car_turning_policy_dict['left_turn'] * 100))
    ])

    route_file = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/' + name + '_' + \
                      '_'.join(traffic_level_configuration) + '.rou.xml'

    JTRROUTER = sumolib.checkBinary('jtrrouter')
    args = [JTRROUTER,
            '-n', net_file,
            '--turn-defaults', turn_default,
            '--route-files', base_route_file,
            '--accept-all-destinations',
            '--max-edges-factor', str(0.5),
            '-o', route_file]

    subprocess.call(args)

    return route_file


def create_experiment_generator(_type='regular', algorithm=None):
    # _type : regular, right_on_red, unregulated

    test_i_scenarios = sorted(next(os.walk(test_i_folder))[1])

    for scenario in test_i_scenarios:

        scenario_folder = test_i_folder + '/' + scenario

        net_file = scenario_folder + '/' + scenario + '__' + _type + '.net.xml'
        net_xml = xml_util.get_xml(net_file)

        intersection_ids = sumo_util.get_traffic_lights(net_xml)
        junction_to_network_incoming_edges_mapping, _ = sumo_util.get_network_border_edges(net_xml)

        sorted_edges_id = []
        for intersection_id in intersection_ids:
            edges = junction_to_network_incoming_edges_mapping[intersection_id]
            sorted_edges_partial = sumo_util.sort_edges_by_angle(edges)

            for edge in sorted_edges_partial:
                sorted_edges_id.append(edge.get('id'))

        traffic_level_configurations = generate_unique_traffic_level_configurations(
            len(sorted_edges_id))

        sumocfg_file = scenario_folder + '/' + scenario + '__' + _type + '.sumocfg'

        for traffic_level_configuration in traffic_level_configurations:

            experiment_name = scenario + '__' + _type + '__' + '_'.join(traffic_level_configuration)

            yield scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder, \
                  sumocfg_file


def run_experiment(arguments):

    scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder, sumocfg_file \
        = arguments

    #route_file = _configure_scenario_routes(scenario, traffic_level_configuration)
    route_file = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/' + scenario + '_' + '_'.join(traffic_level_configuration) + '.rou.xml'

    route_files = [route_file]
    additional_files = []
    environment_additional_files = []
    traffic_light_file = net_file

    if scenario == 'acosta' or scenario == 'pasubio' or scenario == 'joined':
        route_files = [
            scenario_folder + '/' + scenario + ".rou.xml",
            scenario_folder + '/' + scenario + "_busses.rou.xml"
        ]
        additional_files = [
            scenario_folder + '/' + scenario + "_bus_stops.add.xml",
            scenario_folder + '/' + scenario + "_detectors.add.xml",
            scenario_folder + '/' + scenario + "_vtypes.add.xml",
            scenario_folder + '/' + scenario + "_tls.add.xml",
        ]
        environment_additional_files = [
            scenario_folder + '/' + scenario + "__multi_intersection_traffic_light.json"
        ]

        traffic_light_file = scenario_folder + '/' + scenario + "_tls.add.xml"

    Experiment.run(scenario, net_file, route_files, sumocfg_file, traffic_level_configuration,
                   additional_files, environment_additional_files, traffic_light_file)

    sys.stdout.flush()

    #os.remove(route_file)


def replay_experiment(arguments):

    scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder, sumocfg_file, \
        experiment, _round = arguments

    #route_file = _configure_scenario_routes(scenario, traffic_level_configuration)
    route_file = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/' + scenario + '_' + '_'.join(traffic_level_configuration) + '.rou.xml'

    route_files = [route_file]
    additional_files = []
    environment_additional_files = []
    traffic_light_file = net_file

    if scenario == 'acosta' or scenario == 'pasubio' or scenario == 'joined':

        route_files = [
            scenario_folder + '/' + scenario + ".rou.xml",
            scenario_folder + '/' + scenario + "_busses.rou.xml"
        ]
        additional_files = [
            scenario_folder + '/' + scenario + "_bus_stops.add.xml",
            scenario_folder + '/' + scenario + "_detectors.add.xml",
            scenario_folder + '/' + scenario + "_vtypes.add.xml",
            scenario_folder + '/' + scenario + "_tls.add.xml",
        ]
        environment_additional_files = [
            scenario_folder + '/' + scenario + "__multi_intersection_traffic_light.json"
        ]

        traffic_light_file = scenario_folder + '/' + scenario + "_tls.add.xml"

    Experiment.visualize_policy_behavior(scenario, experiment, net_file, route_files, sumocfg_file,
                                         traffic_level_configuration, additional_files,
                                         environment_additional_files,
                                         traffic_light_file, _type, 'Frap', _round=_round)

    sys.stdout.flush()

    #os.remove(route_file)


def continue_experiment(arguments):

    scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder, sumocfg_file, \
        experiment = arguments

    #route_file = _configure_scenario_routes(scenario, traffic_level_configuration)
    route_file = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/' + scenario + '_' + '_'.join(traffic_level_configuration) + '.rou.xml'

    route_files = [route_file]
    additional_files = []
    environment_additional_files = []
    traffic_light_file = net_file

    if scenario == 'acosta' or scenario == 'pasubio' or scenario == 'joined':

        route_files = [
            scenario_folder + '/' + scenario + ".rou.xml",
            scenario_folder + '/' + scenario + "_busses.rou.xml"
        ]
        additional_files = [
            scenario_folder + '/' + scenario + "_bus_stops.add.xml",
            scenario_folder + '/' + scenario + "_detectors.add.xml",
            scenario_folder + '/' + scenario + "_vtypes.add.xml",
            scenario_folder + '/' + scenario + "_tls.add.xml",
        ]
        environment_additional_files = [
            scenario_folder + '/' + scenario + "__multi_intersection_traffic_light.json"
        ]

        traffic_light_file = scenario_folder + '/' + scenario + "_tls.add.xml"

    if algorithm == 'FRAP':
        Experiment.continue_(scenario, experiment, net_file, route_files, sumocfg_file,
                             traffic_level_configuration, additional_files, environment_additional_files,
                             traffic_light_file)

    else:
        raise ValueError('please specify algorithm that can be continued')

    sys.stdout.flush()

    #os.remove(route_file)


def re_run(arguments):

    scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder, sumocfg_file, \
        experiment = arguments

    #route_file = _configure_scenario_routes(scenario, traffic_level_configuration)
    route_file = scenario_folder + '/' + 'temp' + '/' + 'routes' + '/' + scenario + '_' + '_'.join(traffic_level_configuration) + '.rou.xml'

    route_files = [route_file]
    additional_files = []

    if algorithm == 'FRAP':
        Experiment.retrain(
            scenario=scenario,
            experiment='0_regular-intersection__right_on_red__custom_4_street_traffic___10_10_07_54_05_10__74168e65-8cce-41de-927d-1a64cbe6b929', 
            _round=3, 
            net_file=net_file, 
            route_files=route_files,
            sumocfg_file=sumocfg_file, 
            traffic_level_configuration=traffic_level_configuration)
    else:
        raise ValueError('please specify algorithm that can be retrained')

    sys.stdout.flush()

    #os.remove(route_file)


def _run(_type='regular', algorithm=None, experiment=None, _round=None, replay=False):
    
    #experiment_generator = create_experiment_generator(_type=_type, algorithm=algorithm)

    # scenario = 'acosta'
    # test_i_folder = definitions.ROOT_DIR + config.SCENARIO_PATH + '/experimental/Bologna_small-0.29.0'
    # traffic_level_configuration = tuple(['native'])
    # experiment_name = 'Bologna_small' + '__' + scenario
    # scenario_folder = test_i_folder + '/' + scenario
    # #net_file = scenario_folder + '/' + scenario + '_' + 'buslanes' +'.net.xml'
    # net_file = scenario_folder + '/' + scenario + '_' + 'buslanes' + '__' + 'unregulated' + '.net.xml'
    # sumocfg_file = scenario_folder + '/' + 'run' + '.sumo.cfg'

    # scenario = 'multi_intersection'
    # test_i_folder = definitions.ROOT_DIR + config.SCENARIO_PATH + '/experimental'
    # traffic_level_configuration = tuple(['custom_4_street_traffic'])

    scenario = '0_regular-intersection'
    traffic_level_configuration = tuple(['custom_4_street_traffic'])
    experiment_name = scenario + '__' + _type + '__' + '_'.join(traffic_level_configuration)
    scenario_folder = test_i_folder + '/' + scenario
    net_file = scenario_folder + '/' + scenario + '__' + _type + '.net.xml'
    sumocfg_file = scenario_folder + '/' + scenario + '__' + _type + '.sumocfg'

    experiment_generator = [(scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file,
                             scenario_folder, sumocfg_file)]


    global NUMBER_OF_PROCESSES

    if algorithm == 'FRAP':
        NUMBER_OF_PROCESSES = 4
    else:
        NUMBER_OF_PROCESSES = 32

    if experiment is None:

        with NoDaemonPool(processes=NUMBER_OF_PROCESSES) as pool:
            pool.map(run_experiment, experiment_generator)

    else:

        if replay:
            experiment_arguments = [
                (scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder,
                 sumocfg_file, experiment, _round)]

            with NoDaemonPool(processes=NUMBER_OF_PROCESSES) as pool:
                pool.map(replay_experiment, experiment_arguments)

        else:
            experiment_arguments = [
                (scenario, traffic_level_configuration, experiment_name, _type, algorithm, net_file, scenario_folder,
                    sumocfg_file, experiment)]

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

    #_run(_type='right_on_red', algorithm='FRAP')
    #_run(_type='unregulated', algorithm='FRAP')

    # Experiment.summary(
    #     ['Frap', '0', 'tab:blue', 'acosta_buslanes__native___04_10_02_53_49__82acedfc-9ba3-4e50-acf3-1792b4251d14'],
    #     plots='summary_only', baseline_comparison=False,
    #     baseline_experiments=[
    #         ['Sumo', '0_regular-intersection__right_on_red__custom_4_street_traffic___02_15_09_15_55__89976a1e-3dde-48ef-aaa3-9c44d9267b56', 0, 'r', 'static'],
    #         ['Sumo', '0_regular-intersection__unregulated__custom_4_street_traffic___02_15_09_56_04__56bfd179-d3dc-461d-a487-1fbb0ae9cfe8', 0, 'g', 'unregulated']
    #     ])

    '''
    Experiment.summary(
        'multi_intersection__unregulated__custom_4_street_traffic___01_29_10_42_55__75339974-6257-4fb7-8379-daf7f8db4ead',
        memo='Frap', plots='summary_only', baseline_comparison=True,
        baseline_experiments=[
            ['Sumo', '0_regular-intersection__right_on_red__custom_4_street_traffic___02_15_09_15_55__89976a1e-3dde-48ef-aaa3-9c44d9267b56', 0, 'r', 'right on red'],
            ['Sumo', '0_regular-intersection__unregulated__custom_4_street_traffic___02_15_09_56_04__56bfd179-d3dc-461d-a487-1fbb0ae9cfe8', 0, 'g', 'unregulated']
        ])
    Experiment.summary(
        'multi_intersection__unregulated__custom_4_street_traffic___01_29_10_42_55__75339974-6257-4fb7-8379-daf7f8db4ead',
        memo='Frap', _round=0, plots='records_only', baseline_comparison=True,
        baseline_experiments=[
            ['Sumo', '0_regular-intersection__right_on_red__custom_4_street_traffic___02_15_09_15_55__89976a1e-3dde-48ef-aaa3-9c44d9267b56', 0, 'r', 'right on red'],
            ['Sumo', '0_regular-intersection__unregulated__custom_4_street_traffic___02_15_09_56_04__56bfd179-d3dc-461d-a487-1fbb0ae9cfe8', 0, 'g', 'unregulated']
        ])
    '''

    '''
    Experiment.summary(
        [['Sumo', 'label_1', 'k', 'joined_buslanes__native___03_27_13_58_30__24dd0839-784d-4b67-96e6-2079634e73fc'],
         ['Sumo', 'label_2', 'k', 'joined_buslanes__unregulated__native___03_27_15_03_11__de9c74ae-12d8-4167-8eaf-314a75cd67d8']],
        plots='summary_only', baseline_comparison=True,
        baseline_experiments=[
            ['Sumo', '0_regular-intersection__right_on_red__custom_4_street_traffic___02_15_09_15_55__89976a1e-3dde-48ef-aaa3-9c44d9267b56', 0, 'r', 'static'],
            ['Sumo', '0_regular-intersection__unregulated__custom_4_street_traffic___02_15_09_56_04__56bfd179-d3dc-461d-a487-1fbb0ae9cfe8', 0, 'g', 'unregulated']
        ])
    '''

    '''
    Experiment.summary(
        'multi_intersection__unregulated__custom_4_street_traffic___01_29_10_42_55__75339974-6257-4fb7-8379-daf7f8db4ead',
        memo='Frap', plots='summary_only', baseline_comparison=True,
        baseline_experiments=[
            ['Sumo', 'multi_intersection__right_on_red__custom_4_street_traffic___02_15_09_00_21__9503f26d-5453-4504-b819-41a52f9b039a', 0, 'r', 'right on red'],
            ['Sumo', 'multi_intersection__unregulated__custom_4_street_traffic___02_15_09_58_24__7a5c1750-9118-40a5-bb91-183f780eaf43', 0, 'g', 'unregulated']
        ])
    Experiment.summary(
        'multi_intersection__unregulated__custom_4_street_traffic___01_29_10_42_55__75339974-6257-4fb7-8379-daf7f8db4ead',
        memo='Frap', _round=0, plots='records_only', baseline_comparison=True,
        baseline_experiments=[
            ['Sumo', 'multi_intersection__right_on_red__custom_4_street_traffic___02_15_09_00_21__9503f26d-5453-4504-b819-41a52f9b039a', 0, 'r', 'right on red'],
            ['Sumo', 'multi_intersection__unregulated__custom_4_street_traffic___02_15_09_58_24__7a5c1750-9118-40a5-bb91-183f780eaf43', 0, 'g', 'unregulated']
        ])
    '''
    '''
    experiment = '0_regular-intersection__right_on_red__custom_4_street_traffic___11_25_21_45_20__5048bb77-0cf8-4472-8b6f-5f78e1cd4989'

    #_round = 'worst_time_loss'
    _round = 0

    Experiment.visualize_policy_behavior(
        scenario='0_regular-intersection', _type='right_on_red', traffic_level_configuration='custom_4_street_traffic',
        memo='Frap', experiment=experiment, _round=_round)

    '''
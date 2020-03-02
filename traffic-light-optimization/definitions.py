from os import path


ROOT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
SCENARIO_DIR = ROOT_DIR + '/scenario'
OUTPUT_DIR = ROOT_DIR + '/output'


def get_tripinfo_dir(experiment):
    return OUTPUT_DIR + '/' + experiment.path + '/tripinfo'

def get_tripinfo_filename(experiment, execution):
    return 'tripinfo' + '_' + experiment.name + '_' + str(execution) + '.xml'

def get_tripinfo_file_path(experiment, execution):
    return get_tripinfo_dir(experiment) + '/' + get_tripinfo_filename(experiment, execution)



def get_scenario_dir(scenario):
    return SCENARIO_DIR + '/' + scenario.name


def get_route_filename(scenario):
    return scenario.name + '.rou.xml'

def get_route_file_path(scenario):
    return get_scenario_dir(scenario) + '/' + get_route_filename(scenario)


def get_network_filename(scenario):
    return scenario.name + '.net.xml'

def get_network_file_path(scenario):
    return get_scenario_dir(scenario) + '/' + get_network_filename(scenario)


def get_sumo_configuration_filename(scenario):
    return scenario.name + '.sumocfg'

def get_sumo_configuration_file_path(scenario):
    return get_scenario_dir(scenario) + '/' + get_sumo_configuration_filename(scenario)


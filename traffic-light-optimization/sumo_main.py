import os
from distutils.dir_util import copy_tree

import lxml.etree as etree

from experiment.experiment import Experiment
from city.simulation.simulation_factory import get_simulation
from utils.sumo_util import configure_sumo_traffic_light_parameters, reset_sumo_traffic_light_parameters, \
    fix_flow_route_association, configure_sumo_flow_parameters, reset_sumo_flow_parameters
from definitions import ROOT_DIR, get_tripinfo_dir, get_scenario_dir, get_sumo_configuration_file_path
from city.scenario.scenario_factory import get_scenario
from city.scenario.inga_small.inga_small_scenario import INGA_SMALL
from experiment.objective import TIME_LOSS
from experiment.strategy import OFF
from config import Config as config

def copy_scenario_files(experiment):

    scenario = experiment.scenario

    if not os.path.isdir(get_scenario_dir(scenario)):
        os.makedirs(get_scenario_dir(scenario))

        copy_tree(config.REGIONS_PATH + '/' + scenario.name, get_scenario_dir(scenario))

        for root, dirs, files in os.walk(get_scenario_dir(scenario)):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o777)


        fix_flow_route_association(scenario)

        remove_sumocfg_output_configuration(experiment)

def prepare_output_folder(experiment):
    if not os.path.isdir(get_tripinfo_dir(experiment)):
        os.makedirs(get_tripinfo_dir(experiment))

        for root, dirs, files in os.walk(get_tripinfo_dir(experiment)):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o777)



def remove_sumocfg_output_configuration(experiment):

    parser = etree.XMLParser(remove_blank_text=True)

    scenario = experiment.scenario
    sumocfg_filename = os.path.abspath(get_sumo_configuration_file_path(scenario))

    sumocfg_xml = etree.parse(sumocfg_filename, parser)

    output = sumocfg_xml.find(".//output")

    if output:
        sumocfg_xml.getroot().remove(output)

    sumocfg_xml.write(sumocfg_filename, pretty_print=True)

def prepare_scenario(experiment, parameters):

    copy_scenario_files(experiment)

    configure_sumo_traffic_light_parameters(experiment, parameters)
    configure_sumo_flow_parameters(experiment.scenario)


def reset_scenario(experiment):

    reset_sumo_traffic_light_parameters(experiment.scenario)
    reset_sumo_flow_parameters(experiment.scenario)

def main(experiment, parameters=None, execution=''):

    if parameters is None:
        parameters = {}

    prepare_scenario(experiment, parameters)

    prepare_output_folder(experiment)

    Simulation = get_simulation(experiment.strategy)
    simulation = Simulation(experiment, parameters, execution=execution)
    simulation.run()
    simulation.shutdown()

    reset_scenario(experiment)

if __name__ == "__main__":

    scenario = get_scenario(INGA_SMALL)

    traffic_configuration = scenario.create_traffic_configuration(10, 10)

    experiment = Experiment(scenario, traffic_configuration, TIME_LOSS, OFF)

    main(experiment)

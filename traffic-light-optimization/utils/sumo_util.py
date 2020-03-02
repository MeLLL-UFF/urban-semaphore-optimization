import os
import ast
import copy
import optparse

from sumolib import checkBinary
import lxml.etree as etree

from city.flow.configurer.flow_configurer import FlowConfigurer
from city.traffic_light_system.traffic_light.configurer.traffic_light_configurer_factory import traffic_light_configurer_instances
from definitions import get_network_file_path, get_route_file_path


def get_sumo_binary():
    options = _get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

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
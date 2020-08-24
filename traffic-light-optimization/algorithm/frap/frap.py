import os
import shutil

import lxml.etree as etree

from algorithm.frap.internal.frap_pub import run_batch
from algorithm.frap.internal.frap_pub import definitions

from utils.sumo_util import get_intersections_ids, get_intersection_edge_ids

class Frap:

    def run(self, net_file, route_file, sumocfg_file, output_file):
        input_data_path = os.path.join(definitions.ROOT_DIR, 'data', 'template_ls')

        net_file_name = net_file.rsplit('/', 1)[1]
        route_file_name = route_file.rsplit('/', 1)[1]
        sumocfg_file_name = sumocfg_file.rsplit('/', 1)[1]

        shutil.copy2(net_file, input_data_path)
        shutil.copy2(route_file, input_data_path)
        shutil.copy2(sumocfg_file, input_data_path)

        external_configurations = {}
        external_configurations['TRAFFIC_FILE_LIST'] = [
            route_file_name
        ]
        external_configurations['SUMOCFG_FILE'] = sumocfg_file_name
        external_configurations['ROADNET_FILE'] = net_file_name
        external_configurations['_LIST_SUMO_FILES'] = [
            external_configurations['SUMOCFG_FILE'],
            external_configurations['ROADNET_FILE']
        ]

        parser = etree.XMLParser(remove_blank_text=True)
        net_xml = etree.parse(net_file, parser)
        intersection_id = get_intersections_ids(net_xml)[0]
        external_configurations['NODE_LIGHT'] = intersection_id

        incoming_edges, _ = get_intersection_edge_ids(net_xml)
        external_configurations['N_LEG'] = len(incoming_edges)

        external_configurations['NUMBER_OF_LEGS_NETWORK_COMPATIBILITY'] = 5

        external_configurations['USE_SUMO_DIRECTIONS_IN_MOVEMENT_DETECTION'] = False

        external_configurations['SUMOCFG_PARAMETERS'] = {
            '-n': net_file,
            '-r': route_file,
            '--tripinfo-output': output_file,
            '--time-to-teleport': -1,
            '--collision.stoptime': 10,
            '--collision.mingap-factor': 0,
            '--collision.action': 'warn',
            '--collision.check-junctions': True,
            '--device.rerouting.threads': 4
        }

        run_batch.run(external_configurations)
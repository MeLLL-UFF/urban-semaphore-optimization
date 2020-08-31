import os
import shutil
import uuid

import lxml.etree as etree
import pandas as pd

from algorithm.frap.internal.frap_pub import run_batch, replay
from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR as FRAP_ROOT_DIR

from utils.sumo_util import get_intersections_ids, get_intersection_edge_ids, get_average_duration_statistic, get_sumo_binary
from config import Config as config


class Frap:

    def run(self, net_file, route_file, sumocfg_file, output_file, traffic_level_configuration):

        external_configurations = self._create_external_configurations_dict(
            net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

        experiment_name = run_batch.run(external_configurations)

        self._consolidate_output_file(output_file, experiment_name)

        return experiment_name

    def visualize_policy_behavior(self, scenario, _type, traffic_level_configuration, experiment='last', _round='best'):
        # experiment: 'last' or actual name
        # _round: 'last', 'best', or number

        output_folder_base = os.path.join(config.SCENARIO_PATH, 'test_i', scenario, 'output', 'FRAP', _type, 
            scenario + '__' + _type + '__' + traffic_level_configuration)

        if experiment == 'last':
            experiment = os.listdir(output_folder_base)[-1]

        output_folder = os.path.join(output_folder_base, experiment)
        frap_records_folder = os.path.join(FRAP_ROOT_DIR, 'records', 'TransferDQN', experiment)

        if _round == 'best':
            result_df = pd.read_csv(os.path.join(output_folder, experiment + '_result.csv'))
            result_df.set_index(result_df.columns[0], inplace=True)

            column_label = 'test'
            
            _round = result_df[column_label].idxmin()

        elif _round == 'last':
            round_folders = os.listdir(os.path.join(frap_records_folder, 'test_round'))
            last_round = round_folders.sort(key=lambda x: int(x.split('_')[1]))[-1]
            _round = last_round.split('_')[1]

        execution_name = 'replay' + '_' + 'test_round' + '_' + 'round' + '_' + str(_round)

        net_file = os.path.join(config.SCENARIO_PATH, 'test_i', scenario, scenario + '__' + _type + '.net.xml')
        route_file = os.path.join(config.SCENARIO_PATH, 'test_i', scenario, 
            'temp', 'routes', scenario + '_' + traffic_level_configuration + '.rou.xml')

        if not os.path.isfile(route_file):
            raise ValueError("Route file does not exist")

        sumocfg_file = os.path.join(config.SCENARIO_PATH, 'test_i', scenario, scenario + '__' + _type + '.sumocfg')
        output_file = os.path.join(output_folder, '')

        external_configurations = self._create_external_configurations_dict(
            net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

        replay.run(
            os.path.join('TransferDQN', experiment), 
            round_number=_round, 
            run_cnt=3600,
            execution_name=execution_name,
            if_gui=True,
            external_configurations=external_configurations)

    def _consolidate_output_file(self, output_file, experiment_name):
        
        output_folder = output_file.rsplit('/', 1)[0] + '/' + experiment_name
        
        duration_df = pd.DataFrame()
        for _file in os.listdir(output_folder):

            duration = get_average_duration_statistic(os.path.join(output_folder,  _file))

            round_split = _file.split('_round_')
            round_number = int(round_split[1].split('.')[0])
            
            if '_train_' in _file:
                generator_number = round_split[0].split('_generator_')[1]
                duration_df.loc[round_number, 'train_generator' + '_' + generator_number] = duration
            elif '_test_' in _file:
                duration_df.loc[round_number, 'test'] = duration          

        duration_df = duration_df.reindex(sorted(duration_df.index))
        duration_df = duration_df.reindex(sorted(duration_df.columns), axis=1)

        duration_df.to_csv(os.path.join(output_folder, experiment_name + '_' + 'result' + '.csv'))

    def _create_external_configurations_dict(self, net_file, route_file, sumocfg_file, output_file,
                                             traffic_level_configuration):

        input_data_path = os.path.join(FRAP_ROOT_DIR, 'data', 'template_ls')

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

        external_configurations['TRAFFIC_LEVEL_CONFIGURATION'] = traffic_level_configuration

        parser = etree.XMLParser(remove_blank_text=True)
        net_xml = etree.parse(net_file, parser)
        intersection_id = get_intersections_ids(net_xml)[0]
        external_configurations['NODE_LIGHT'] = intersection_id

        incoming_edges, _ = get_intersection_edge_ids(net_xml)
        external_configurations['N_LEG'] = len(incoming_edges)

        external_configurations['NUMBER_OF_LEGS_NETWORK_COMPATIBILITY'] = 5

        external_configurations['USE_SUMO_DIRECTIONS_IN_MOVEMENT_DETECTION'] = False
        external_configurations['UNIQUE_ID'] = str(uuid.uuid4())

        external_configurations['SUMOCFG_PARAMETERS'] = {
            '-n': net_file,
            '-r': route_file,
            '--log': output_file,
            '--duration-log.statistics': True,
            '--time-to-teleport': -1,
            '--collision.stoptime': 10,
            '--collision.mingap-factor': 0,
            '--collision.action': 'warn',
            '--collision.check-junctions': True,
            '--device.rerouting.threads': 4
        }

        return external_configurations

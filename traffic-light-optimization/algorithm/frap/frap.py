import os
import shutil
import uuid

import numpy as np
import lxml.etree as etree
import pandas as pd
import matplotlib as mlp
mlp.use("agg")
import matplotlib.pyplot as plt

from algorithm.frap.internal.frap_pub import run_batch, replay, summary
from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR as FRAP_ROOT_DIR

from utils.sumo_util import get_intersections_ids, get_intersection_edge_ids, get_average_duration_statistic, get_sumo_binary
from config import Config as config
from definitions import ROOT_DIR


class Frap:

    def run(self, net_file, route_file, sumocfg_file, output_file, traffic_level_configuration):

        external_configurations = self._create_external_configurations_dict(
            net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

        experiment_name = run_batch.run(external_configurations)

        output_folder = output_file.rsplit('/', 1)[0] + '/' + experiment_name        

        self._consolidate_output_file(output_folder, experiment_name)

        return experiment_name

    def visualize_policy_behavior(self, scenario, _type, traffic_level_configuration, experiment='last', _round='best_time_loss'):
        # experiment: 'last' or actual name
        # _round: 'last', 'best_time_loss', 'best_average_trip_duration', 'best_reward' or number

        output_folder_base = ROOT_DIR + os.path.join(config.SCENARIO_PATH, 'test_i', scenario, 'output', 'FRAP', _type, 
            scenario + '__' + _type + '__' + traffic_level_configuration)

        if experiment == 'last':
            experiment = sorted(os.listdir(output_folder_base))[-1]

        output_folder = os.path.join(output_folder_base, experiment)
        frap_records_folder = os.path.join(FRAP_ROOT_DIR, 'records', 'TransferDQN', experiment)
        frap_summary_folder = os.path.join(FRAP_ROOT_DIR, 'summary', 'TransferDQN', experiment)

        if _round == 'best_time_loss':
            result_df = pd.read_csv(os.path.join(frap_summary_folder, experiment + '-test-time_loss.csv'))
            result_df.set_index(result_df.columns[0], inplace=True)

            column_label = 'time_loss'
            
            _round = result_df[column_label].idxmin()

        elif _round == 'best_average_trip_duration':
            result_df = pd.read_csv(os.path.join(output_folder, experiment + '_result.csv'))
            result_df.set_index(result_df.columns[0], inplace=True)

            column_label = 'test'
            
            _round = result_df[column_label].idxmin()
        
        elif _round == 'best_reward':
            result_df = pd.read_csv(os.path.join(frap_summary_folder, experiment + '-test-reward.csv'))
            result_df.set_index(result_df.columns[0], inplace=True)

            column_label = 'reward'
            
            _round = result_df[column_label].idxmax()

        elif _round == 'last':
            round_folders = next(os.walk(os.path.join(frap_records_folder, 'test_round')))[1]
            round_folders.sort(key=lambda x: int(x.split('_')[1]))
            last_round = round_folders[-1]
            _round = int(last_round.split('_')[1])

        execution_name = 'replay' + '_' + 'test_round' + '_' + 'round' + '_' + str(_round)

        net_file = os.path.join(frap_records_folder, scenario + '__' + _type + '.net.xml')
        route_file = os.path.join(frap_records_folder, scenario + '_' + traffic_level_configuration + '.rou.xml')
        sumocfg_file = os.path.join(frap_records_folder, scenario + '__' + _type + '.sumocfg')
        output_file = os.path.join(output_folder, '')

        if not os.path.isfile(route_file):
            raise ValueError("Route file does not exist")

        external_configurations = self._create_external_configurations_dict(
            net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

        replay.run(
            os.path.join('TransferDQN', experiment), 
            round_number=_round, 
            run_cnt=3600,
            execution_name=execution_name,
            if_gui=True,
            external_configurations=external_configurations)

    def _consolidate_output_file(self, output_folder, experiment_name):
        
        duration_df = pd.DataFrame()
        for _file in os.listdir(output_folder):

            if '.out' not in _file:
                continue
            
            try:
                duration = get_average_duration_statistic(os.path.join(output_folder,  _file))
            except:
                duration = np.NaN

            round_split = _file.split('_round_')
            round_number = int(round_split[1].split('.')[0])
            
            if '_train_' in _file:
                generator_number = round_split[0].split('_generator_')[1]
                duration_df.loc[round_number, 'train_generator' + '_' + generator_number] = duration
            elif '_test_' in _file:
                duration_df.loc[round_number, 'test'] = duration          

        duration_df = duration_df.reindex(sorted(duration_df.index))
        duration_df = duration_df.reindex(sorted(duration_df.columns), axis=1)

        duration_df.to_csv(os.path.join(output_folder, experiment_name + '_' + 'avg_trip_duration_result' + '.csv'))

        split_experiment_name = experiment_name.split('__')
        scenario = split_experiment_name[0]
        traffic_level_configuration = split_experiment_name[2]

        if 'test' in duration_df:
            self._plot_consolidate_output(output_folder, experiment_name, duration_df['test'], 
                scenario, traffic_level_configuration)


    def _plot_consolidate_output(self, output_folder, experiment_name, duration_list,
                                 scenario, traffic_level_configuration):
        
        num_rounds = len(duration_list)
        NAN_LABEL = -1

        min_duration = float('inf')
        min_duration_id = 0

        validation_duration_length = 10
        minimum_round = 50 if num_rounds > 50 else 0
        duration_list = np.array(duration_list)

        nan_count = len(np.where(duration_list == NAN_LABEL)[0])
        validation_duration = duration_list[-validation_duration_length:]
        final_duration = np.round(np.mean(validation_duration[validation_duration > 0]), decimals=2)

        if nan_count == 0:
            convergence = {1.2: len(duration_list) - 1, 1.1: len(duration_list) - 1}
            for j in range(minimum_round, len(duration_list)):
                for level in [1.2, 1.1]:
                    if max(duration_list[j:]) <= level * final_duration:
                        if convergence[level] > j:
                            convergence[level] = j
            conv_12 = convergence[1.2]
            conv_11 = convergence[1.1]
        else:
            conv_12, conv_11 = 0, 0

        
        right_on_red_output_folder = ROOT_DIR + os.path.join(
            config.SCENARIO_PATH, 'test_i', scenario, 'output', 'None', 'right_on_red',
            scenario + '__' + 'right_on_red' + '__' + traffic_level_configuration)
        unregulated_output_folder = ROOT_DIR + os.path.join(
            config.SCENARIO_PATH, 'test_i', scenario, 'output', 'None', 'unregulated',
            scenario + '__' + 'unregulated' + '__' + traffic_level_configuration)

        right_on_red_result_file = right_on_red_output_folder + '/' + scenario + '__' + 'right_on_red' + '__' + \
                                   traffic_level_configuration + '_result.csv'

        unregulated_result_file = unregulated_output_folder + '/' + scenario + '__' + 'unregulated' + '__' + \
                                  traffic_level_configuration + '_result.csv'

        # simple plot for each training instance
        f, ax = plt.subplots(1, 1, figsize=(20, 9), dpi=100)
        ax.plot(duration_list, linewidth=2, color='k')
        ax.plot([0, len(duration_list)], [final_duration, final_duration], linewidth=2, color="g")
        ax.plot([conv_12, conv_12], [duration_list[conv_12], duration_list[conv_12] * 3], linewidth=2, color="b")
        ax.plot([conv_11, conv_11], [duration_list[conv_11], duration_list[conv_11] * 3], linewidth=2, color="b")
        ax.plot([0, len(duration_list)], [min_duration, min_duration], linewidth=2, color="r")
        ax.plot([min_duration_id, min_duration_id], [min_duration, min_duration * 3], linewidth=2, color="r")

        if os.path.isfile(right_on_red_result_file):
            right_on_red_df = pd.read_csv(right_on_red_result_file)
            ax.plot([0, len(duration_list)], [right_on_red_df['test'], right_on_red_df['test']],
                    linewidth=2, linestyle=':', color='r')
        
        if os.path.isfile(unregulated_result_file):
            unregulated_df = pd.read_csv(unregulated_result_file)
            ax.plot([0, len(duration_list)], [unregulated_df['test'], unregulated_df['test']],
                    linewidth=2, linestyle=':', color='g')

        ax.set_title(experiment_name.split('___')[0]  + "-" + str(final_duration))
        plt.savefig(os.path.join(output_folder, experiment_name + '_' + 'result' + '.png'))
        plt.close()

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


    def summary(self, experiment, plots='all', _round=None):
        summary.single_experiment_summary('TransferDQN', 'records/TransferDQN/' + experiment, plots=plots, _round=_round)

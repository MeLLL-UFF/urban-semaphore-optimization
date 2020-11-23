import os
import shutil
import uuid

import numpy as np
import lxml.etree as etree
import pandas as pd
import matplotlib as mlp
mlp.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, MaxNLocator, FormatStrFormatter)

from algorithm.frap_pub import run_batch, replay, summary
from algorithm.frap_pub.definitions import ROOT_DIR as FRAP_ROOT_DIR

from utils import sumo_util
from config import Config as config
from definitions import ROOT_DIR


class Experiment:

    @staticmethod
    def run(net_file, route_file, sumocfg_file, output_file, traffic_level_configuration):

        external_configurations = Experiment._create_external_configurations_dict(
            net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

        experiment_name = run_batch.run(external_configurations)

        output_folder = output_file.rsplit('/', 1)[0] + '/' + experiment_name        

        Experiment._consolidate_output_file(output_folder, experiment_name)

        return experiment_name

    @staticmethod
    def continue_(experiment, net_file, route_file, sumocfg_file, output_file, traffic_level_configuration):

        external_configurations = Experiment._create_external_configurations_dict(
            net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

        experiment_name = run_batch.continue_(experiment, external_configurations)

        return experiment_name

    @staticmethod
    def visualize_policy_behavior(scenario, _type, traffic_level_configuration, experiment='last',
                                  _round='best_time_loss'):
        # experiment: 'last' or actual name
        # _round: 'last', 'best_time_loss', 'best_average_trip_duration', 'best_reward' or number

        output_folder_base = ROOT_DIR + os.path.join(config.SCENARIO_PATH, 'test_i', scenario, 'output', 'FRAP', _type, 
            scenario + '__' + _type + '__' + traffic_level_configuration)

        if experiment == 'last':
            experiment = sorted(os.listdir(output_folder_base))[-1]

        output_folder = os.path.join(output_folder_base, experiment)
        frap_records_folder = os.path.join(FRAP_ROOT_DIR, 'records', 'Frap', experiment)
        frap_summary_folder = os.path.join(FRAP_ROOT_DIR, 'summary', 'Frap', experiment)

        if _round == 'best_time_loss':
            result_df = pd.read_csv(os.path.join(frap_summary_folder, experiment + '-test-time_loss.csv'))
            result_df.set_index(result_df.columns[0], inplace=True)

            column_label = 'time_loss'
            
            _round = result_df[column_label].idxmin()
        
        elif _round == 'worst_time_loss':
            result_df = pd.read_csv(os.path.join(frap_summary_folder, experiment + '-test-time_loss.csv'))
            result_df.set_index(result_df.columns[0], inplace=True)

            column_label = 'time_loss'
            
            _round = result_df[column_label].idxmax()

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

        external_configurations = Experiment._create_external_configurations_dict(
            net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

        replay.run(
            os.path.join('Frap', experiment),
            round_number=_round, 
            run_cnt=3600,
            execution_name=execution_name,
            if_gui=True,
            rewrite_mode=False,
            external_configurations=external_configurations)


    @staticmethod
    def retrain(experiment, _round, net_file, route_file, sumocfg_file, output_file, traffic_level_configuration):

        external_configurations = Experiment._create_external_configurations_dict(
            net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

        experiment_name = run_batch.continue_(experiment, _round, external_configurations)

        return experiment_name


    @staticmethod
    def summary(experiment, memo, plots='all', _round=None, baseline_comparison=True, baseline_experiments=None):
        summary.single_experiment_summary(memo, 'records/' + memo + '/' + experiment,
                                          plots, _round, baseline_comparison, baseline_experiments)

    @staticmethod
    def _consolidate_output_file(output_folder, experiment_name):
        
        duration_df = pd.DataFrame()
        for _file in os.listdir(output_folder):

            if '.out' not in _file:
                continue
            
            try:
                duration = sumo_util.get_average_duration_statistic(os.path.join(output_folder,  _file))
            except:
                duration = np.NaN

            round_split = _file.split('_round_')
            round_number = int(round_split[1].split('__')[0])
            
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
            Experiment._plot_consolidate_output(output_folder, experiment_name, duration_df['test'],
                                                scenario, traffic_level_configuration)


    @staticmethod
    def _plot_consolidate_output(output_folder, experiment_name, duration_list,
                                 scenario, traffic_level_configuration):
        
        duration_list = np.array(duration_list)

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

        tail_length = 10
        duration_tail = duration_list[-tail_length:]
        final_duration = np.round(np.mean(duration_tail[duration_tail > 0]), decimals=2)

        # simple plot for each training instance
        f, ax = plt.subplots(1, 1, figsize=(20, 9), dpi=100)
        ax.plot(duration_list, linewidth=2, color='k', label='frap' + ' ' + '(' + str(final_duration) + ')')

        if os.path.isfile(right_on_red_result_file):
            right_on_red_df = pd.read_csv(right_on_red_result_file)
            ax.plot([0, len(duration_list)], [right_on_red_df['test'], right_on_red_df['test']],
                    linewidth=2, linestyle=':', color='r',
                    label='right on red' + ' ' + '(' + str(right_on_red_df['test'][0]) + ')')
        
        if os.path.isfile(unregulated_result_file):
            unregulated_df = pd.read_csv(unregulated_result_file)
            ax.plot([0, len(duration_list)], [unregulated_df['test'], unregulated_df['test']],
                    linewidth=2, linestyle=':', color='g',
                    label='unregulated' + ' ' + '(' + str(unregulated_df['test'][0]) + ')')

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_minor_locator(MultipleLocator(10))

        ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(10))

        ax.set_axisbelow(True)
        ax.grid(color='gray', linestyle='dashed', alpha=0.5, which='both')

        ax.legend()

        ax.set_title('average trip duration')
        plt.savefig(os.path.join(output_folder, experiment_name + '_' + 'result' + '.png'))
        plt.close()

    @staticmethod
    def _create_external_configurations_dict(net_file, route_file, sumocfg_file, output_file,
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
        intersection_id = sumo_util.get_intersections_ids(net_xml)[0]
        external_configurations['NODE_LIGHT'] = intersection_id

        incoming_edges, _ = sumo_util.get_intersection_edge_ids(net_xml)
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
            '--device.rerouting.threads': 4,
            '--ignore-junction-blocker': 10  # Currently not working
        }

        return external_configurations

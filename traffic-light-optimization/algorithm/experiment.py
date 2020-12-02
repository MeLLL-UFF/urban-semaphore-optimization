import os
import shutil
import uuid

import pandas as pd
import matplotlib as mlp
mlp.use("agg")

from algorithm.frap_pub import run_batch, replay, summary
from algorithm.frap_pub.definitions import ROOT_DIR as FRAP_ROOT_DIR

from config import Config as config
from definitions import ROOT_DIR


class Experiment:

    @staticmethod
    def run(net_file, route_file, sumocfg_file, output_file, traffic_level_configuration):

        external_configurations = Experiment._create_external_configurations_dict(
            net_file, route_file, sumocfg_file, output_file, traffic_level_configuration)

        experiment_name = run_batch.run(external_configurations)

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
        external_configurations['NET_FILE'] = net_file_name
        external_configurations['_LIST_SUMO_FILES'] = [
            external_configurations['SUMOCFG_FILE'],
            external_configurations['NET_FILE']
        ]

        external_configurations['TRAFFIC_LEVEL_CONFIGURATION'] = traffic_level_configuration

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
            '--ignore-junction-blocker': 10  # working in Sumo 1.8.0
        }

        return external_configurations

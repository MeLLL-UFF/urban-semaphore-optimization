import os
import pickle as pkl
import copy
import json

import pandas as pd

import matplotlib as mlp
mlp.use("agg")

from utils import summary_util

from algorithm.frap_pub.definitions import ROOT_DIR

font = {"family": "Times New Roman", 'size': 24}
mlp.rc('font', **font)

def single_experiment_intersection_summary(memo, records_dir, intersection_id, single_round=0, plots='all',
                                           baseline_comparison=False, baseline_experiments=None):
    # plots: None, 'records_only'

    traffic_env_conf = open(os.path.join(ROOT_DIR, records_dir, "traffic_env.conf"), 'r')
    dic_traffic_env_conf = json.load(traffic_env_conf)

    test_round_dir = os.path.join(records_dir, "test_round")
    try:
        round_files = os.listdir(ROOT_DIR + '/' + test_round_dir)
    except:
        print("no test round in {}".format(records_dir))
        return
    round_files = [f for f in round_files if "round" in f]
    round_files.sort(key=lambda x: int(x[6:]))

    round_ = round_files[single_round]

    traffic_folder = records_dir.rsplit('/', 1)[1]
    traffic_name = traffic_folder
    mode_name = 'test'
    name_base = traffic_name + '-' + mode_name

    intersection_index = dic_traffic_env_conf['INTERSECTION_ID'].index(intersection_id)
    movements = dic_traffic_env_conf['MOVEMENT'][intersection_index]
    movement_to_connection = dic_traffic_env_conf['MOVEMENT_TO_CONNECTION'][intersection_index]
    movement_to_traffic_light_index = dic_traffic_env_conf['MOVEMENT_TO_TRAFFIC_LIGHT_LINK_INDEX'][intersection_index]

    algorithm_label = memo

    round_dir = os.path.join(test_round_dir, round_)

    reward_each_step = []
    time_loss_each_step = []
    traffic_light_each_step = []
    relative_occupancy_each_step = []
    relative_mean_speed_each_step = []
    absolute_number_of_vehicles_each_step = []

    # summary items (queue_length) from pickle
    f = open(os.path.join(ROOT_DIR, round_dir, "inter_{0}_detailed.pkl".format(intersection_id)), "rb")
    samples = pkl.load(f)
    f.close()
    for sample in samples:
        reward_each_step.append(sample['reward'])
        time_loss_each_step.append(sample['extra']['time_loss'])
        if plots is not None:
            traffic_light_each_step.append(sample['extra']['traffic_light'])
            relative_occupancy_each_step.append(sample['extra']['relative_occupancy'])
            relative_mean_speed_each_step.append(sample['extra']['relative_mean_speed'])
            absolute_number_of_vehicles_each_step.append(sample['extra']['absolute_number_of_vehicles'])

    save_path = ROOT_DIR + '/' + round_dir

    if plots is not None:

        summary_util.consolidate_time_loss(
            time_loss_each_step,
            save_path,
            name_base,
            algorithm_label,
            baseline_comparison=baseline_comparison,
            baseline_experiments=baseline_experiments)
        summary_util.consolidate_reward(reward_each_step, save_path, name_base)

        summary_util.consolidate_occupancy_and_speed_inflow_outflow(
            relative_occupancy_each_step,
            relative_mean_speed_each_step,
            movements,
            movement_to_connection,
            save_path,
            name_base)

        summary_util.consolidate_phase_and_demand(
            absolute_number_of_vehicles_each_step,
            traffic_light_each_step,
            movements,
            movement_to_connection,
            movement_to_traffic_light_index,
            save_path,
            name_base)


def single_experiment_network_summary(memo, records_dir, single_round=None, plots='all',
                                      baseline_comparison=False, baseline_experiments=None):
    # plots: None, 'records_only' 'summary_only', 'all'

    if single_round is not None:
        if plots == 'all':
            print('Changing plots to "records only"')
            plots = 'records_only'
        elif plots == 'summary_only':
            raise ValueError('It is not possible to specify a round and choose summary_only at the same time')

    test_round_dir = os.path.join(records_dir, "test_round")
    try:
        round_files = os.listdir(ROOT_DIR + '/' + test_round_dir)
    except:
        print("no test round in {}".format(records_dir))
        return
    round_files = [f for f in round_files if "round" in f]
    round_files.sort(key=lambda x: int(x[6:]))

    if single_round is not None:
        round_files = [round_files[single_round]]

    average_reward_each_round = []
    percentage_time_loss_per_driver_each_round = pd.DataFrame()
    cumulative_time_loss_per_driver_each_round = pd.DataFrame()

    traffic_folder = records_dir.rsplit('/', 1)[1]
    traffic_name = traffic_folder
    mode_name = 'test'
    name_base = traffic_name + '-' + mode_name

    algorithm_label = memo

    for round_ in round_files:

        round_dir = os.path.join(test_round_dir, round_)

        cumulative_time_loss_per_driver_each_step = []
        percentage_time_loss_per_driver_each_step = []

        if not os.path.isfile(os.path.join(ROOT_DIR, round_dir, "network_detailed.pkl")):
            continue

        # summary items (queue_length) from pickle
        f = open(os.path.join(ROOT_DIR, round_dir, "network_detailed.pkl"), "rb")
        samples = pkl.load(f)
        f.close()
        for sample in samples:
            cumulative_time_loss_per_driver_each_step.append(sample['extra']['cumulative_time_loss_per_driver'])
            percentage_time_loss_per_driver_each_step.append(sample['extra']['percentage_time_loss_per_driver'])

        cumulative_time_loss_per_driver_each_step = pd.DataFrame(cumulative_time_loss_per_driver_each_step)
        percentage_time_loss_per_driver_each_step = pd.DataFrame(percentage_time_loss_per_driver_each_step)

        save_path = ROOT_DIR + '/' + round_dir

        if plots is not None and plots != 'summary_only':

            summary_util.plot_cumulative_time_loss_per_driver(
                cumulative_time_loss_per_driver_each_step,
                save_path,
                name_base,
                algorithm_label,
                baseline_comparison=baseline_comparison,
                baseline_experiments=baseline_experiments)

            summary_util.plot_percentage_time_loss_per_driver(
                percentage_time_loss_per_driver_each_step,
                save_path,
                name_base,
                algorithm_label,
                baseline_comparison=baseline_comparison,
                baseline_experiments=baseline_experiments)

        if plots is not None and plots != 'records_only':

            cumulative_time_loss_per_driver_each_round = \
                cumulative_time_loss_per_driver_each_round.append(
                    cumulative_time_loss_per_driver_each_step.iloc[-1], ignore_index=True)
            percentage_time_loss_per_driver_each_round = \
                percentage_time_loss_per_driver_each_round.append(
                    percentage_time_loss_per_driver_each_step.mean(), ignore_index=True)

    if single_round is not None:
        return

    traffic_folder = records_dir.rsplit('/', 1)[1]
    result_dir = os.path.join("summary", memo, traffic_folder)
    if not os.path.exists(ROOT_DIR + '/' + result_dir):
        os.makedirs(ROOT_DIR + '/' + result_dir)

    save_path = ROOT_DIR + '/' + result_dir

    if plots is not None and plots != 'records_only':

        cumulative_time_loss_per_driver_each_round.to_csv(
            save_path + "/" + name_base + "-" + 'cumulative_time_loss_per_driver' + ".csv")

        summary_util.plot_cumulative_time_loss_per_driver(
            cumulative_time_loss_per_driver_each_round,
            save_path,
            name_base,
            algorithm_label,
            baseline_comparison=baseline_comparison,
            baseline_experiments=baseline_experiments,
            is_summary=True)

        percentage_time_loss_per_driver_each_round.to_csv(
            save_path + "/" + name_base + "-" + 'percentage_time_loss_per_driver' + ".csv")

        summary_util.plot_percentage_time_loss_per_driver(
            percentage_time_loss_per_driver_each_round,
            save_path,
            name_base,
            algorithm_label,
            baseline_comparison=baseline_comparison,
            baseline_experiments=baseline_experiments,
            is_summary=True)

        summary_util.consolidate_reward(average_reward_each_round, save_path, name_base)

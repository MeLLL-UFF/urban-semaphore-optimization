import pickle as pkl
import os
import pandas as pd
import numpy as np
import json
import copy
from math import isnan
from lxml import etree
import matplotlib as mlp
from algorithm.frap.internal.frap_pub.script import  *
from algorithm.frap.internal.frap_pub.sumo_env import get_connections, get_lane_traffic_light_controller

mlp.use("agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, MaxNLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.lines import Line2D

from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR

font = {'size': 24}
mlp.rc('font', **font)

NAN_LABEL = -1


def get_metrics(duration_list, queue_length_list, min_duration, min_duration_id, min_queue_length, min_queue_length_id,
                traffic_name, total_summary, mode_name, save_path, num_rounds, min_duration2=None):
    validation_duration_length = 10
    minimum_round = 50 if num_rounds > 50 else 0
    duration_list = np.array(duration_list)
    queue_length_list = np.array(queue_length_list)

    # min_duration, min_duration_id = np.min(duration_list), np.argmin(duration_list)
    # min_queue_length, min_queue_length_id = np.min(queue_length_list), np.argmin(queue_length_list)

    nan_count = len(np.where(duration_list == NAN_LABEL)[0])
    validation_duration = duration_list[-validation_duration_length:]
    final_duration = np.round(np.mean(validation_duration[validation_duration > 0]), decimals=2)
    final_duration_std = np.round(np.std(validation_duration[validation_duration > 0]), decimals=2)

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

    # simple plot for each training instance
    f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
    ax.plot(duration_list, linewidth=2, color='k')
    ax.plot([0, len(duration_list)], [final_duration, final_duration], linewidth=2, color="g")
    ax.plot([conv_12, conv_12], [duration_list[conv_12], duration_list[conv_12] * 3], linewidth=2, color="b")
    ax.plot([conv_11, conv_11], [duration_list[conv_11], duration_list[conv_11] * 3], linewidth=2, color="b")
    ax.plot([0, len(duration_list)], [min_duration, min_duration], linewidth=2, color="r")
    ax.plot([min_duration_id, min_duration_id], [min_duration, min_duration * 3], linewidth=2, color="r")
    ax.set_title(traffic_name + "-" + str(final_duration))
    plt.savefig(ROOT_DIR + '/' + save_path + "/" + traffic_name + "-" + mode_name + ".png")
    plt.close()

    total_summary["traffic_file"].append(traffic_name)
    total_summary["traffic"].append(traffic_name.split("___")[0])
    total_summary["min_queue_length"].append(min_queue_length)
    total_summary["min_queue_length_round"].append(min_queue_length_id)
    total_summary["min_duration"].append(min_duration)
    total_summary["min_duration_round"].append(min_duration_id)
    total_summary["final_duration"].append(final_duration)
    total_summary["final_duration_std"].append(final_duration_std)
    total_summary["convergence_1.2"].append(conv_12)
    total_summary["convergence_1.1"].append(conv_11)
    total_summary["nan_count"].append(nan_count)
    total_summary["min_duration2"].append(min_duration2)

    return total_summary


def summary_plot(traffic_performance, figure_dir, mode_name, num_rounds):
    minimum_round = 50 if num_rounds > 50 else 0
    validation_duration_length = 10
    anomaly_threshold = 1.3

    for traffic_name in traffic_performance:
        f, ax = plt.subplots(2, 1, figsize=(12, 9), dpi=100)
        performance_tmp = []
        check_list = []
        for ti in range(len(traffic_performance[traffic_name])):
            ax[0].plot(traffic_performance[traffic_name][ti][0], linewidth=2)
            validation_duration = traffic_performance[traffic_name][ti][0][-validation_duration_length:]
            final_duration = np.round(np.mean(validation_duration), decimals=2)
            if len(np.where(traffic_performance[traffic_name][ti][0] == NAN_LABEL)[0]) == 0:
                # and len(traffic_performance[traffic_name][ti][0]) == num_rounds:
                tmp = traffic_performance[traffic_name][ti][0]
                if len(tmp) < num_rounds:
                    tmp.extend([float("nan")] * (num_rounds - len(traffic_performance[traffic_name][ti][0])))
                performance_tmp.append(tmp)
                check_list.append(final_duration)
            else:
                print("the length of traffic {} is shorter than {}".format(traffic_name, num_rounds))
        check_list = np.array(check_list)
        for ci in np.where(check_list > anomaly_threshold * np.mean(check_list))[0][::-1]:
            del performance_tmp[ci]
            print("anomaly traffic_name:{} id:{} err:{}".format(traffic_name, ci, check_list[ci] - np.mean(check_list)))
        if len(performance_tmp) == 0:
            print("The result of {} is not enough for analysis.".format(traffic_name))
            continue
        try:
            performance_summary = np.array(performance_tmp)
            print(traffic_name, performance_summary.shape)
            ax[1].errorbar(x=range(len(traffic_performance[traffic_name][0][0])),
                           y=np.mean(performance_summary, axis=0),
                           yerr=np.std(performance_summary, axis=0))

            psm = np.mean(performance_summary, axis=0)
            validation_duration = psm[-validation_duration_length:]
            final_duration = np.round(np.mean(validation_duration), decimals=2)

            convergence = {1.2: len(psm) - 1, 1.1: len(psm) - 1}
            for j in range(minimum_round, len(psm)):
                for level in [1.2, 1.1]:
                    if max(psm[j:]) <= level * final_duration:
                        if convergence[level] > j:
                            convergence[level] = j
            ax[1].plot([0, len(psm)], [final_duration, final_duration], linewidth=2, color="g")
            ax[1].text(len(psm), final_duration * 2, "final-" + str(final_duration))
            ax[1].plot([convergence[1.2], convergence[1.2]], [psm[convergence[1.2]], psm[convergence[1.2]] * 3],
                       linewidth=2, color="b")
            ax[1].text(convergence[1.2], psm[convergence[1.2]] * 2, "conv 1.2-" + str(convergence[1.2]))
            ax[1].plot([convergence[1.1], convergence[1.1]], [psm[convergence[1.1]], psm[convergence[1.1]] * 3],
                       linewidth=2, color="b")
            ax[1].text(convergence[1.1], psm[convergence[1.1]] * 2, "conv 1.1-" + str(convergence[1.1]))
            ax[1].set_title(traffic_name)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
            plt.savefig(ROOT_DIR + '/' + figure_dir + "/" + traffic_name + "-" + mode_name + ".png")
            plt.close()
        except:
            print("plot error")


def plot_segment_duration(round_summary, path, mode_name):
    save_path = os.path.join(path, "segments")
    if not os.path.exists(ROOT_DIR + '/' + save_path):
        os.makedirs(ROOT_DIR + '/' + save_path)
    for key in round_summary.keys():
        if "duration" in key:
            f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
            ax.plot(round_summary[key], linewidth=2, color='k')
            ax.set_title(key)
            plt.savefig(ROOT_DIR + '/' + save_path + "/" + key + "-" + mode_name + ".png")
            plt.close()


def padding_duration(performance_duration):
    for traffic_name in performance_duration.keys():
        max_duration_length = max([len(x[0]) for x in performance_duration[traffic_name]])
        for i, ti in enumerate(performance_duration[traffic_name]):
            performance_duration[traffic_name][i][0].extend((max_duration_length - len(ti[0]))*[ti[0][-1]])

    return performance_duration


def performance_at_min_duration_round_plot(performance_at_min_duration_round, figure_dir, mode_name):
    for traffic_name in performance_at_min_duration_round:
        f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
        for ti in range(len(performance_at_min_duration_round[traffic_name])):
            ax.plot(performance_at_min_duration_round[traffic_name][ti][0], linewidth=2)
        plt.savefig(ROOT_DIR + '/' + figure_dir + "/" + "min_duration_round" + "-" + mode_name + ".png")
        plt.close()


def summary_detail_train(memo, total_summary):
    # each_round_train_duration

    performance_duration = {}
    performance_at_min_duration_round = {}
    records_dir = os.path.join("records", memo)
    for traffic_file in os.listdir(ROOT_DIR + '/' + records_dir):
        if ".xml" not in traffic_file and ".json" not in traffic_file:
            continue
        print(traffic_file)

        min_queue_length = min_duration = float('inf')
        min_queue_length_id = min_duration_ind = 0

        # get run_counts to calculate the queue_length each second
        exp_conf = open(os.path.join(ROOT_DIR, records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        run_counts = dic_exp_conf["RUN_COUNTS"]
        num_rounds = dic_exp_conf["NUM_ROUNDS"]
        num_seg = run_counts // 3600

        traffic_vol = get_total_traffic_volume(dic_exp_conf["TRAFFIC_FILE"][0])
        nan_thres = 120

        duration_each_round_list = []
        queue_length_each_round_list = []

        train_round_dir = os.path.join(records_dir, traffic_file, "train_round")
        round_files = os.listdir(ROOT_DIR + '/' + train_round_dir)
        round_files = [f for f in round_files if "round" in f]
        round_files.sort(key=lambda x: int(x[6:]))
        round_summary = {"round": list(range(num_rounds))}
        for round in round_files:
            try:
                round_dir = os.path.join(train_round_dir, round)

                duration_gens = 0
                queue_length_gens = 0
                cnt_gen = 0

                list_duration_seg = [float('inf')] * num_seg
                list_queue_length_seg = [float('inf')] * num_seg
                list_queue_length_id_seg = [0] * num_seg
                list_duration_id_seg = [0] * num_seg
                for gen in os.listdir(ROOT_DIR + '/' + round_dir):
                    if "generator" not in gen:
                        continue

                    # summary items (queue_length) from pickle
                    gen_dir = os.path.join(records_dir, traffic_file, "train_round", round, gen)
                    f = open(os.path.join(ROOT_DIR, gen_dir, "inter_0.pkl"), "rb")
                    samples = pkl.load(f)

                    for sample in samples:
                        queue_length_gens += sum(sample['state']['lane_queue_length'])
                    sample_num = len(samples)
                    f.close()

                    # summary items (duration) from csv
                    df_vehicle_inter_0 = pd.read_csv(os.path.join(ROOT_DIR, round_dir, gen, "vehicle_inter_0.csv"),
                                                     sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                     names=["vehicle_id", "enter_time", "leave_time"])

                    duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
                    ave_duration = np.mean([time for time in duration if not isnan(time)])
                    real_traffic_vol = 0
                    nan_num = 0
                    for time in duration:
                        if not isnan(time):
                            real_traffic_vol += 1
                        else:
                            nan_num += 1
                    # print(ave_duration)

                    cnt_gen += 1
                    duration_gens += ave_duration

                    for i, interval in enumerate(range(0, run_counts, 3600)):
                        did = np.bitwise_and(df_vehicle_inter_0["enter_time"].values < interval + 3600,
                                             df_vehicle_inter_0["enter_time"].values > interval)
                        # vehicle_in_seg = sum([int(x) for x in (df_vehicle_inter_0["enter_time"][did].values > 0)])
                        # vehicle_out_seg = sum([int(x) for x in (df_vehicle_inter_0["leave_time"][did].values > 0)])
                        duration_seg = df_vehicle_inter_0["leave_time"][did].values - df_vehicle_inter_0["enter_time"][
                            did].values
                        ave_duration_seg = np.mean([time for time in duration_seg if not isnan(time)])
                        # print(traffic_file, round, i, ave_duration)
                        real_traffic_vol_seg = 0
                        nan_num_seg = 0
                        for time in duration_seg:
                            if not isnan(time):
                                real_traffic_vol_seg += 1
                            else:
                                nan_num_seg += 1

                        # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)

                        if nan_num_seg < nan_thres:
                            # if min_duration[i] > ave_duration and ave_duration > 24:
                            list_duration_seg[i] = ave_duration_seg
                            list_duration_id_seg[i] = int(round[6:])


                list_duration_seg = np.array(list_duration_seg)/cnt_gen
                for j in range(num_seg):
                    key = "min_duration-" + str(j)
                    if key not in round_summary.keys():
                        round_summary[key] = [list_duration_seg[j]]
                    else:
                        round_summary[key].append(list_duration_seg[j])


                duration_each_round_list.append(duration_gens / cnt_gen)
                queue_length_each_round_list.append(queue_length_gens / cnt_gen / sample_num)

                # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)
                if min_queue_length > queue_length_gens / cnt_gen / sample_num:
                    min_queue_length = queue_length_gens / cnt_gen / sample_num
                    min_queue_length_id = int(round[6:])

                valid_flag = json.load(open(os.path.join(ROOT_DIR, gen_dir, "valid_flag.json")))
                if valid_flag['0']: # temporary for one intersection
                    if min_duration > duration_gens / cnt_gen:
                        min_duration = duration_gens / cnt_gen
                        min_duration_ind = int(round[6:])
                #print(nan_num, nan_thres)

            except:
                # change anomaly label from nan to -1000 for the convenience of following computation
                duration_each_round_list.append(NAN_LABEL)
                queue_length_each_round_list.append(NAN_LABEL)

        result_dir = os.path.join("summary", memo, traffic_file)
        if not os.path.exists(ROOT_DIR + '/' + result_dir):
            os.makedirs(ROOT_DIR + '/' + result_dir)

        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(ROOT_DIR, result_dir, "train_results.csv"))
        if num_seg > 1:
            round_result = pd.DataFrame(round_summary)
            round_result.to_csv(os.path.join(ROOT_DIR, result_dir, "train_seg_results.csv"), index=False)
            # plot duration segment
            plot_segment_duration(round_summary, result_dir, mode_name="train")
            duration_each_segment_list = round_result.iloc[min_duration_ind][1:].values
            if ".xml" in traffic_file:
                traffic_name, traffic_time = traffic_file.split(".xml")
            elif ".json" in traffic_file:
                traffic_name, traffic_time = traffic_file.split(".json")
            if traffic_name not in performance_at_min_duration_round:
                performance_at_min_duration_round[traffic_name] = [(duration_each_segment_list, traffic_time)]
            else:
                performance_at_min_duration_round[traffic_name].append((duration_each_segment_list, traffic_time))



        # total_summary
        total_summary = get_metrics(duration_each_round_list, queue_length_each_round_list,
                                    min_duration, min_duration_ind, min_queue_length, min_queue_length_id,
                                    traffic_file, total_summary,
                                    mode_name="train", save_path=result_dir, num_rounds=num_rounds)

        if ".xml" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".xml")
        elif ".json" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".json")
        if traffic_name not in performance_duration:
            performance_duration[traffic_name] = [(duration_each_round_list, traffic_time)]
        else:
            performance_duration[traffic_name].append((duration_each_round_list, traffic_time))

    figure_dir = os.path.join("summary", memo, "figures")
    if not os.path.exists(ROOT_DIR + '/' + figure_dir):
        os.makedirs(ROOT_DIR + '/' + figure_dir)
    if dic_exp_conf["EARLY_STOP"]:
        performance_duration = padding_duration(performance_duration)
    summary_plot(performance_duration, figure_dir, mode_name="train", num_rounds=num_rounds)
    total_result = pd.DataFrame(total_summary)
    total_result.to_csv(os.path.join(ROOT_DIR, "summary", memo, "total_train_results.csv"))
    performance_at_min_duration_round_plot(performance_at_min_duration_round, figure_dir, mode_name="train")


def summary_detail_test(memo, total_summary):
    # each_round_train_duration

    performance_duration = {}
    performance_at_min_duration_round = {}
    records_dir = os.path.join("records", memo)
    for traffic_file in os.listdir(ROOT_DIR + '/' + records_dir):
        if ".xml" not in traffic_file and ".json" not in traffic_file:
            continue

        #if "cross.2phases_rou01_equal_700.xml_12_11_08_16_00" != traffic_file:
        #    continue
        print(traffic_file)

        min_queue_length = min_duration = min_duration2 = float('inf')
        min_queue_length_id = min_duration_ind = 0

        # get run_counts to calculate the queue_length each second
        exp_conf = open(os.path.join(ROOT_DIR, records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        run_counts = dic_exp_conf["RUN_COUNTS"]
        num_rounds = dic_exp_conf["NUM_ROUNDS"]
        num_seg = run_counts//3600

        traffic_vol = get_total_traffic_volume(dic_exp_conf["TRAFFIC_FILE"][0])
        nan_thres = 120

        duration_each_round_list = []
        duration_each_round_list2 = []
        queue_length_each_round_list = []
        num_of_vehicle_in = []
        num_of_vehicle_out = []

        train_round_dir = os.path.join(records_dir, traffic_file, "test_round")
        try:
            round_files = os.listdir(ROOT_DIR + '/' + train_round_dir)
        except:
            print("no test round in {}".format(traffic_file))
            continue
        round_files = [f for f in round_files if "round" in f]
        round_files.sort(key=lambda x: int(x[6:]))
        round_summary = {"round": list(range(num_rounds))}
        for round in round_files:

            try:
                round_dir = os.path.join(train_round_dir, round)

                list_duration_seg = [float('inf')] * num_seg
                list_queue_length_seg = [float('inf')] * num_seg
                list_queue_length_id_seg = [0] * num_seg
                list_duration_id_seg = [0] * num_seg

                # summary items (queue_length) from pickle
                f = open(os.path.join(ROOT_DIR, round_dir, "inter_0.pkl"), "rb")
                samples = pkl.load(f)
                queue_length_each_round = 0
                for sample in samples:
                    queue_length_each_round += sum(sample['state']['lane_queue_length'])
                sample_num = len(samples)
                f.close()

                # summary items (duration) from csv
                df_vehicle_inter_0 = pd.read_csv(os.path.join(ROOT_DIR + '/' + round_dir, "vehicle_inter_0.csv"),
                                                 sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                 names=["vehicle_id", "enter_time", "leave_time"])

                vehicle_in = sum([int(x) for x in (df_vehicle_inter_0["enter_time"].values > 0)])
                vehicle_out = sum([int(x) for x in (df_vehicle_inter_0["leave_time"].values > 0)])
                total_vol = get_total_traffic_volume(traffic_file)
                duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
                ave_duration = np.mean([time for time in duration if not isnan(time)])
                # print(ave_duration)

                if "peak" in traffic_file:
                    did1 = df_vehicle_inter_0["enter_time"].values <= run_counts / 2
                    duration = df_vehicle_inter_0["leave_time"][did1].values - df_vehicle_inter_0["enter_time"][
                        did1].values
                    ave_duration = np.mean([time for time in duration if not isnan(time)])

                    did2 = df_vehicle_inter_0["enter_time"].values > run_counts / 2
                    duration2 = df_vehicle_inter_0["leave_time"][did2].values - df_vehicle_inter_0["enter_time"][
                        did2].values
                    ave_duration2 = np.mean([time for time in duration2 if not isnan(time)])
                    duration_each_round_list2.append(ave_duration2)

                    real_traffic_vol2 = 0
                    nan_num2 = 0
                    for time in duration2:
                        if not isnan(time):
                            real_traffic_vol2 += 1
                        else:
                            nan_num2 += 1

                    if nan_num2 < nan_thres:
                        if min_duration2 > ave_duration2 and ave_duration2 > 24:
                            min_duration2 = ave_duration2
                            min_duration_ind2 = int(round[6:])

                real_traffic_vol = 0
                nan_num = 0
                for time in duration:
                    if not isnan(time):
                        real_traffic_vol += 1
                    else:
                        nan_num += 1

                duration_each_round_list.append(ave_duration)
                queue_length_each_round_list.append(queue_length_each_round / sample_num)
                num_of_vehicle_in.append(vehicle_in)
                num_of_vehicle_out.append(vehicle_out)

                # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)
                if min_queue_length > queue_length_each_round / sample_num:
                    min_queue_length = queue_length_each_round / sample_num
                    min_queue_length_id = int(round[6:])

                valid_flag = json.load(open(os.path.join(ROOT_DIR, round_dir, "valid_flag.json")))
                #if valid_flag['0']:  # temporary for one intersection
                if total_vol is not None and vehicle_out > total_vol * 0.9:
                    if min_duration > ave_duration and ave_duration > 24:
                        print(">", traffic_file)
                        print(">>>", ave_duration, vehicle_out, total_vol)
                        min_duration = ave_duration
                        min_duration_ind = int(round[6:])

                if num_seg > 1:
                    for i, interval in enumerate(range(0, run_counts, 3600)):
                        did = np.bitwise_and(df_vehicle_inter_0["enter_time"].values < interval + 3600,
                                             df_vehicle_inter_0["enter_time"].values > interval)
                        #vehicle_in_seg = sum([int(x) for x in (df_vehicle_inter_0["enter_time"][did].values > 0)])
                        #vehicle_out_seg = sum([int(x) for x in (df_vehicle_inter_0["leave_time"][did].values > 0)])
                        duration_seg = df_vehicle_inter_0["leave_time"][did].values - df_vehicle_inter_0["enter_time"][
                            did].values
                        ave_duration_seg = np.mean([time for time in duration_seg if not isnan(time)])
                        # print(traffic_file, round, i, ave_duration)
                        real_traffic_vol_seg = 0
                        nan_num_seg = 0
                        for time in duration_seg:
                            if not isnan(time):
                                real_traffic_vol_seg += 1
                            else:
                                nan_num_seg += 1

                        # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)

                        if nan_num_seg < nan_thres:
                            # if min_duration[i] > ave_duration and ave_duration > 24:
                            list_duration_seg[i] = ave_duration_seg
                            list_duration_id_seg[i] = int(round[6:])

                        #round_summary = {}
                    for j in range(num_seg):
                        key = "min_duration-" + str(j)
                        if key not in round_summary.keys():
                            round_summary[key] = [list_duration_seg[j]]
                        else:
                            round_summary[key].append(list_duration_seg[j])
                    #round_result_dir = os.path.join("summary", memo, traffic_file)
                    #if not os.path.exists(round_result_dir):
                    #    os.makedirs(round_result_dir)

            except:
                duration_each_round_list.append(NAN_LABEL)
                queue_length_each_round_list.append(NAN_LABEL)
                num_of_vehicle_in.append(NAN_LABEL)
                num_of_vehicle_out.append(NAN_LABEL)
                if "peak" in traffic_file:
                    duration_each_round_list2.append(NAN_LABEL)

        # result_dir = os.path.join(records_dir, traffic_file)
        result_dir = os.path.join("summary", memo, traffic_file)
        if not os.path.exists(ROOT_DIR + '/' + result_dir):
            os.makedirs(ROOT_DIR + '/' + result_dir)
        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list,
            "vehicle_in": num_of_vehicle_in,
            "vehicle_out": num_of_vehicle_out
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(ROOT_DIR, result_dir, "test_results.csv"))
        if num_seg > 1:
            round_result = pd.DataFrame(round_summary)
            round_result.to_csv(os.path.join(ROOT_DIR + '/' + result_dir, "test_seg_results.csv"), index=False)
            plot_segment_duration(round_summary, result_dir, mode_name="test")
            duration_each_segment_list = round_result.iloc[min_duration_ind][1:].values

            traffic_name, traffic_time = traffic_file.split(".xml")
            if traffic_name not in performance_at_min_duration_round:
                performance_at_min_duration_round[traffic_name] = [(duration_each_segment_list, traffic_time)]
            else:
                performance_at_min_duration_round[traffic_name].append((duration_each_segment_list, traffic_time))


        # print(os.path.join(result_dir, "test_results.csv"))

        # total_summary
        total_summary = get_metrics(duration_each_round_list, queue_length_each_round_list,
                                    min_duration, min_duration_ind, min_queue_length, min_queue_length_id,
                                    traffic_file, total_summary,
                                    mode_name="test", save_path=result_dir, num_rounds=num_rounds,
                                    min_duration2=None if "peak" not in traffic_file else min_duration2)

        if ".xml" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".xml")
        elif ".json" in traffic_file:
            traffic_name, traffic_time = traffic_file.split(".json")
        if traffic_name not in performance_duration:
            performance_duration[traffic_name] = [(duration_each_round_list, traffic_time)]
        else:
            performance_duration[traffic_name].append((duration_each_round_list, traffic_time))

    total_result = pd.DataFrame(total_summary)
    if not os.path.exists(ROOT_DIR + '/' + "summary" + '/' + memo):
        os.makedirs(ROOT_DIR + '/' + "summary" + '/' + memo)
    total_result.to_csv(os.path.join(ROOT_DIR, "summary", memo, "total_test_results.csv"))
    figure_dir = os.path.join("summary", memo, "figures")
    if not os.path.exists(ROOT_DIR + '/' + figure_dir):
        os.makedirs(ROOT_DIR + '/' + figure_dir)
    if dic_exp_conf["EARLY_STOP"]:
        performance_duration = padding_duration(performance_duration)
    summary_plot(performance_duration, figure_dir, mode_name="test", num_rounds=num_rounds)
    performance_at_min_duration_round_plot(performance_at_min_duration_round, figure_dir, mode_name="test")


def summary_detail_baseline(memo):
    # each_round_train_duration
    total_summary = {
        "traffic": [],
        "min_queue_length": [],
        "min_queue_length_round": [],
        "min_duration": [],
        "min_duration_round": []
    }

    records_dir = os.path.join("records", memo)
    for traffic_file in os.listdir(ROOT_DIR + '/' + records_dir):
        if ".xml" not in traffic_file and ".json" not in traffic_file:
            continue
        print(traffic_file)

        # if "650" not in traffic_file:
        #    continue

        # get run_counts to calculate the queue_length each second
        exp_conf = open(os.path.join(ROOT_DIR, records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        run_counts = dic_exp_conf["RUN_COUNTS"]

        duration_each_round_list = []
        queue_length_each_round_list = []

        train_dir = os.path.join(records_dir, traffic_file)

        # summary items (queue_length) from pickle
        f = open(os.path.join(ROOT_DIR, train_dir, "inter_0.pkl"), "rb")
        try:
            samples = pkl.load(f)
        except:
            continue
        for sample in samples:
            queue_length_each_round = sum(sample['state']['lane_queue_length'])
        f.close()

        # summary items (duration) from csv
        df_vehicle_inter_0 = pd.read_csv(os.path.join(ROOT_DIR, train_dir, "vehicle_inter_0.csv"),
                                         sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                         names=["vehicle_id", "enter_time", "leave_time"])

        duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
        ave_duration = np.mean([time for time in duration if not isnan(time)])
        # print(ave_duration)

        duration_each_round_list.append(ave_duration)
        ql = queue_length_each_round / len(samples)
        queue_length_each_round_list.append(ql)

        # result_dir = os.path.join(records_dir, traffic_file)
        result_dir = os.path.join("summary", memo, traffic_file)
        if not os.path.exists(ROOT_DIR + '/' + result_dir):
            os.makedirs(ROOT_DIR + '/' + result_dir)
        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(ROOT_DIR, result_dir, "test_results.csv"))
        # print(os.path.join(result_dir, "test_results.csv"))

        total_summary["traffic"].append(traffic_file)
        total_summary["min_queue_length"].append(ql)
        total_summary["min_queue_length_round"].append(0)
        total_summary["min_duration"].append(ave_duration)
        total_summary["min_duration_round"].append(0)

    total_result = pd.DataFrame(total_summary)
    total_result.to_csv(os.path.join(ROOT_DIR, "summary", memo, "total_baseline_test_results.csv"))


def consolidate(object_label, object_list, save_path, traffic_name, mode_name):

    objects_df = pd.DataFrame({object_label: object_list})

    objects_df.to_csv(ROOT_DIR + '/' + save_path + "/" + traffic_name + "-" + mode_name + "-" + object_label + ".csv")

    plot(object_label, object_list, save_path, traffic_name, mode_name)


def plot(object_label, object_list, save_path, traffic_name, mode_name):

    f, ax = plt.subplots(1, 1, figsize=(20, 9), dpi=100)
    ax.margins(0)
    ax.plot(object_list, linewidth=2, color='k')
    ax.set_title(traffic_name + "-" + str(np.mean(object_list)))
    plt.savefig(ROOT_DIR + '/' + save_path + "/" + traffic_name + "-" + mode_name + "-" + object_label + ".png")
    plt.close()

def consolidate_occupancy_and_speed_inflow_outflow(relative_occupancy_each_step, relative_mean_speed_each_step, 
        dic_traffic_env_conf, save_path, traffic_name, mode_name):

    relative_occupancy_df = pd.DataFrame(relative_occupancy_each_step)
    relative_mean_speed_df = pd.DataFrame(relative_mean_speed_each_step)

    movements = dic_traffic_env_conf['list_lane_order']
    movement_to_connection = dic_traffic_env_conf['movement_to_connection']

    from_lane_set = set()
    to_lane_set = set()

    per_movement_df = pd.DataFrame()
    for movement in movements:

        connection = movement_to_connection[movement]

        from_lane = connection['from'] + '_' + connection['fromLane']
        to_lane = connection['to'] + '_' + connection['toLane']

        per_movement_df.loc[:, movement + '_' + 'inflow' + '_' + 'relative_occupancy'] = relative_occupancy_df.loc[:, from_lane]
        per_movement_df.loc[:, movement + '_' + 'outflow' + '_' + 'relative_occupancy'] = relative_occupancy_df.loc[:, to_lane]
        per_movement_df.loc[:, movement + '_' + 'inflow' + '_' + 'relative_mean_speed'] = relative_mean_speed_df.loc[:, from_lane]
        per_movement_df.loc[:, movement + '_' + 'outflow' + '_' + 'relative_mean_speed'] = relative_mean_speed_df.loc[:, to_lane]

        from_lane_set.add(from_lane)
        to_lane_set.add(to_lane)

    per_movement_df.to_csv(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'occupancy_and_speed_inflow_outflow' + '-' + 'per_movement' + '.csv')


    edge_dict = {}
    lanes = relative_occupancy_df.columns.tolist()
    for lane in lanes:
        edge = lane.split('_')[0]

        if edge in edge_dict:
            edge_dict[edge].append(lane)
        else:
            edge_dict[edge] = [lane]

    per_edge_df = pd.DataFrame()   
    edges = list(edge_dict.keys()) 
    for edge in edges:
        per_edge_df.loc[:, edge + '_' + 'relative_occupancy'] = relative_occupancy_df.loc[:, edge_dict[edge]].mean(axis=1)
        per_edge_df.loc[:, edge + '_' + 'relative_mean_speed'] = relative_mean_speed_df.loc[:, edge_dict[edge]].mean(axis=1)

    per_edge_df.to_csv(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'occupancy_and_speed_inflow_outflow' + '-' + 'per_edge' + '.csv')


    from_lanes_relative_occupancy_df = relative_occupancy_df.loc[:, from_lane_set].mean(axis=1)
    to_lanes_relative_occupancy_df = relative_occupancy_df.loc[:, to_lane_set].mean(axis=1)
    from_lanes_relative_mean_speed_df = relative_mean_speed_df.loc[:, from_lane_set].mean(axis=1)
    to_lanes_relative_mean_speed_df = relative_mean_speed_df.loc[:, to_lane_set].mean(axis=1)

    all_lanes_df = pd.concat(
        [
            from_lanes_relative_occupancy_df, 
            to_lanes_relative_occupancy_df,
            from_lanes_relative_mean_speed_df, 
            to_lanes_relative_mean_speed_df
        ], axis=1, sort=False)

    all_lanes_df.columns = [
        'from_lane_relative_occupancy', 
        'to_lane_relative_occupancy', 
        'from_lane_relative_mean_speed',
        'to_lane_relative_mean_speed'
    ]

    all_lanes_df.to_csv(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'occupancy_and_speed_inflow_outflow' + '-' + 'all_lanes' + '.csv')

    
    f, axs = plt.subplots(len(edges), 2, figsize=(120, 18*len(edges)), dpi=100)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.05, hspace=0.05)

    for i in range(0, len(edges)):

        axs[i][0].margins(0)
        axs[i][1].margins(0)

        edge = edges[i]

        axs[i][0].plot(per_edge_df.iloc[:, i*2+0:i*2+1], linewidth=2, color='k')
        axs[i][1].plot(per_edge_df.iloc[:, i*2+1:i*2+2], linewidth=2, color='k')
        axs[i][0].set_title(edge + ' ' + 'relative_occupancy (%)')
        axs[i][1].set_title(edge + ' ' + 'relative_speed (%)')
        axs[i][0].set_ylim(0, 1)
        axs[i][1].set_ylim(0, 1)

    plt.savefig(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'occupancy_and_speed_inflow_outflow' + '-' + 'per_edge' + '.png')
    plt.close()


    f, axs = plt.subplots(len(movements), 2, figsize=(120, 18*len(movements)), dpi=100)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.05, hspace=0.05)

    for i in range(0, len(movements)):

        movement = movements[i]

        axs[i][0].margins(0)
        axs[i][1].margins(0)

        axs[i][0].plot(per_movement_df.iloc[:, i*4+0:i*4+1], linewidth=2, color='b')
        axs[i][0].plot(per_movement_df.iloc[:, i*4+1:i*4+2], linewidth=2, color='r')
        axs[i][1].plot(per_movement_df.iloc[:, i*4+2:i*4+3], linewidth=2, color='b')
        axs[i][1].plot(per_movement_df.iloc[:, i*4+3:i*4+4], linewidth=2, color='r')
        axs[i][0].set_title(movement + ' ' + 'relative_occupancy (%)')
        axs[i][1].set_title(movement + ' ' + 'relative_speed (%)')
        axs[i][0].set_ylim(0, 1)
        axs[i][1].set_ylim(0, 1)
        axs[i][0].legend(['inflow', 'outflow'])
        axs[i][1].legend(['inflow', 'outflow'])

    plt.savefig(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'occupancy_and_speed_inflow_outflow' + '-' + 'per_movement' + '.png')
    plt.close()


    f, axs = plt.subplots(1, 2, figsize=(120, 18), dpi=100)
    plt.subplots_adjust(left=0.02, right=0.98, wspace=0.05, hspace=0.05)

    axs[0].margins(0)
    axs[1].margins(0)

    axs[0].plot(all_lanes_df.iloc[:, 0:1], linewidth=2, color='b')
    axs[0].plot(all_lanes_df.iloc[:, 1:2], linewidth=2, color='r')
    axs[1].plot(all_lanes_df.iloc[:, 2:3], linewidth=2, color='b')
    axs[1].plot(all_lanes_df.iloc[:, 3:4], linewidth=2, color='r')
    axs[0].set_title('relative_occupancy (%)')
    axs[1].set_title('relative_speed (%)')
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].legend(['inflow', 'outflow'])
    axs[1].legend(['inflow', 'outflow'])

    plt.savefig(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'occupancy_and_speed_inflow_outflow' + '-' + 'all_lanes' + '.png')
    plt.close()


def consolidate_phase_and_demand(absolute_number_of_cars_each_step, traffic_light_each_step, 
        dic_traffic_env_conf, records_dir, save_path, traffic_name, mode_name):

    absolute_number_of_cars_df = pd.DataFrame(absolute_number_of_cars_each_step)
    traffic_light_df = pd.DataFrame(traffic_light_each_step)

    movements = dic_traffic_env_conf['list_lane_order']
    movement_to_connection = dic_traffic_env_conf['movement_to_connection']

    absolute_number_of_cars_per_movement_df = pd.DataFrame()
    for movement in movements:

        connection = movement_to_connection[movement]
        from_lane = connection['from'] + '_' + connection['fromLane']

        absolute_number_of_cars_per_movement_df.loc[:, movement] = absolute_number_of_cars_df.loc[:, from_lane]

    def percent(row):

        new_row = []

        row_sum = row.sum()
        for index, element in enumerate(row):
            if row_sum == 0:
                new_row.append(0)
            else:
                new_row.append(element/row_sum)

        row.iloc[:] = new_row

        return row

    percent_number_of_cars_per_movement_df = absolute_number_of_cars_per_movement_df.apply(percent, axis=1)

    absolute_number_of_cars_per_movement_df.to_csv(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'phase_and_demand' + '-' + 'absolute' + '-' + 'per_movement' + '.csv')

    percent_number_of_cars_per_movement_df.to_csv(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'phase_and_demand' + '-' + 'percent' + '-' + 'per_movement' + '.csv')
    

    traffic_light_color_mapping = {
        'r': (1, 0, 0),             # red
        'y': (1, 1, 0),             # yellow
        'g': (0, 0.702, 0),         # dark green
        'G': (0, 1, 0),             # green
        's': (0.502, 0, 0.502),     # purple
        'u': (1, 0.502, 0),         # orange
        'o': (0.502, 0.251, 0),     # brown
        'O': (0, 1, 1)              # cyan
    }


    net_file = ROOT_DIR + '/' + records_dir + '/' + dic_traffic_env_conf['ROADNET_FILE']
    parser = etree.XMLParser(remove_blank_text=True)
    net_file_xml = etree.parse(net_file, parser)
    connections = get_connections(net_file_xml)
    entering_lanes = [connection.get('from') + '_' + connection.get('fromLane') for connection in connections]
    lane_to_traffic_light_index_mapping = get_lane_traffic_light_controller(net_file_xml, entering_lanes)

    movement_to_traffic_light_index_mapping = {}
    for movement in movements:

        connection = movement_to_connection[movement]
        from_lane = connection['from'] + '_' + connection['fromLane']

        movement_to_traffic_light_index_mapping[movement] = lane_to_traffic_light_index_mapping[from_lane]

    def strip_traffic_light(traffic_light):

        new_data = []

        for index in movement_to_traffic_light_index_mapping.values():
            new_data.append(traffic_light.iloc[0][int(index)])

        lanes_traffic_light = pd.Series(new_data, movement_to_traffic_light_index_mapping.keys())

        return lanes_traffic_light

    traffic_light_df = traffic_light_df.apply(strip_traffic_light, axis=1)

    traffic_light_df.to_csv(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'phase_and_demand' + '-' + 'traffic_light' + '.csv')

    traffic_light_colors = traffic_light_df.applymap(lambda x: traffic_light_color_mapping[x])


    legend_label_keys_kwargs = {
        'marker': 's', 
        'markersize': 30,
        'linewidth': 0,
    }

    legend_label_keys = [
        Line2D([], [], color=traffic_light_color_mapping['r'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['y'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['g'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['G'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['s'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['u'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['o'], **legend_label_keys_kwargs),
        Line2D([], [], color=traffic_light_color_mapping['O'], **legend_label_keys_kwargs)
    ]

    legend_label_values = [
        'red light', 
        'amber (yellow) light', 
        'green light, no priority', 
        'green light, priority', 
        'right on red light', 
        'red+yellow light',
        'off, blinking',
        'off, no signal'
    ]


    x_len, y_len = percent_number_of_cars_per_movement_df.shape

    f, axs = plt.subplots(y_len, 1, figsize=(60, 3*y_len), dpi=100, sharex=True)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.05, hspace=0)

    x = range(0, x_len)
    for i in range(0, y_len):

        data = percent_number_of_cars_per_movement_df.iloc[:, i]
        
        axs[i].margins(0)
        axs[i].set_ylim(0, 1)

        if i == y_len - 1:
            axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[i].yaxis.set_minor_locator(MultipleLocator(0.05))

        axs[i].xaxis.set_major_locator(MultipleLocator(300))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.set_minor_locator(MultipleLocator(10))

        axs[i].set_axisbelow(True)
        axs[i].grid(color='gray', linestyle='dashed', alpha=0.5, which='both')


        lane_traffic_light_colors = traffic_light_colors.iloc[:, i]

        axs[i].plot(data, linewidth=2, color='k')
        axs[i].bar(x, data, width=1, color=tuple(lane_traffic_light_colors))

        axs[i].set_ylabel(data.name.split('_', 1)[0], labelpad=-100, rotation=0)

    axs[0].legend(legend_label_keys, legend_label_values, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=len(legend_label_values), mode="expand", borderaxespad=0.)

    plt.suptitle('phase and demand distribution (%, in decimal)')

    plt.savefig(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'phase_and_demand' + '-' + 'percent' + '-' + 'per_movement' + '.png')
    plt.close()


    x_len, y_len = absolute_number_of_cars_per_movement_df.shape

    f, axs = plt.subplots(y_len, 1, figsize=(60, 3*y_len), dpi=100, sharex=True)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90, wspace=0.05, hspace=0)

    max_number_of_cars = absolute_number_of_cars_per_movement_df.values.max()

    x = range(0, x_len)
    for i in range(0, y_len):

        data = absolute_number_of_cars_per_movement_df.iloc[:, i]
        
        axs[i].margins(0)
        axs[i].set_ylim(0, max_number_of_cars)

        if i == y_len - 1:
            axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].yaxis.set_minor_locator(MultipleLocator(1))

        axs[i].xaxis.set_major_locator(MultipleLocator(300))
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axs[i].xaxis.set_minor_locator(MultipleLocator(10))

        axs[i].set_axisbelow(True)
        axs[i].grid(color='gray', linestyle='dashed', alpha=0.5, which='both')


        lane_traffic_light_colors = traffic_light_colors.iloc[:, i]

        axs[i].plot(data, linewidth=2, color='k')
        axs[i].bar(x, data, width=1, color=tuple(lane_traffic_light_colors))

        axs[i].set_ylabel(data.name.split('_', 1)[0], labelpad=-100, rotation=0)

    axs[0].legend(legend_label_keys, legend_label_values, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=len(legend_label_values), mode="expand", borderaxespad=0.)

    plt.suptitle('phase and demand distribution (absolute numbers)')

    plt.savefig(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'phase_and_demand' + '-' + 'absolute' + '-' + 'per_movement' + '.png')
    plt.close()


    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                    '%d' % round(height*100) + '%',
                    ha='center', va='bottom')


    f, ax = plt.subplots(1, 1, figsize=(40, 20), dpi=100)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.90, wspace=0.05, hspace=0.05)

    ax.margins(0.05)
    ax.set_ylim(0, 1)

    ax.yaxis.set_major_locator(MaxNLocator(nbins= 10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    x_ticks = []
    width = 1

    sort_order = list(traffic_light_color_mapping.keys())

    accumulated_width = 0
    for i in range(0, traffic_light_df.shape[1]):

        data = percent(traffic_light_df.iloc[:, i].value_counts())
        sorted_index = sorted(list(data.index), key=lambda x: sort_order.index(x))
        data = data.reindex(sorted_index)

        lane_traffic_light_colors = [traffic_light_color_mapping[index] for index in data.index]
        
        base_bar_positions = np.multiply([width]*len(data), range(0, len(data)))
        bar_positions = accumulated_width + base_bar_positions + width/2

        bars = ax.bar(bar_positions, data, width=width, color=tuple(lane_traffic_light_colors))
        autolabel(bars)

        x_ticks.append(accumulated_width + width*len(data)/2)
        
        accumulated_width += width*len(data) + width


    ax.set_xticks(x_ticks)
    ax.set_xticklabels(traffic_light_df.columns)

    ax.set_axisbelow(True)
    ax.grid(axis='y', color='gray', linestyle='dashed', alpha=0.7, which='both')

    ax.legend(legend_label_keys, legend_label_values, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=len(legend_label_values), mode="expand", borderaxespad=0.)

    plt.suptitle('traffic light distribution (%, in decimal)')

    plt.savefig(ROOT_DIR + '/' + save_path + '/' + traffic_name + '-' + mode_name + '-' + 
        'traffic light distribution' + '-' + 'per_movement' + '.png')
    plt.close()    


def single_experiment_summary_detail_test(memo, records_dir, total_summary):
    # each_round_train_duration

    performance_duration = {}
    performance_at_min_duration_round = {}

    min_queue_length = min_duration = min_duration2 = float('inf')
    min_queue_length_id = min_duration_ind = 0

    # get run_counts to calculate the queue_length each second
    exp_conf = open(os.path.join(ROOT_DIR, records_dir, "exp.conf"), 'r')
    dic_exp_conf = json.load(exp_conf)
    run_counts = dic_exp_conf["RUN_COUNTS"]
    num_rounds = dic_exp_conf["NUM_ROUNDS"]
    num_seg = run_counts//3600

    nan_thres = 120

    duration_each_round_list = []
    duration_each_round_list2 = []
    queue_length_each_round_list = []
    num_of_vehicle_in = []
    num_of_vehicle_out = []

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
    round_summary = {"round": list(range(num_rounds))}

    average_reward_each_round = []
    average_time_loss_each_round = []
    average_relative_occupancy_each_round = []
    average_relative_mean_speed_each_round = []
    for round in round_files:

        try:
            round_dir = os.path.join(test_round_dir, round)

            list_duration_seg = [float('inf')] * num_seg
            list_queue_length_seg = [float('inf')] * num_seg
            list_queue_length_id_seg = [0] * num_seg
            list_duration_id_seg = [0] * num_seg

            # summary items (queue_length) from pickle
            f = open(os.path.join(ROOT_DIR, round_dir, "inter_0_detailed.pkl"), "rb")
            samples = pkl.load(f)
            queue_length_each_round = 0
            reward_each_step = []
            time_loss_each_step = []
            traffic_light_each_step = []
            relative_occupancy_each_step = []
            relative_mean_speed_each_step = []
            absolute_number_of_cars_each_step = []

            for sample in samples:
                queue_length_each_round += sum(sample['state']['lane_queue_length'])
                reward_each_step.append(sample['reward'])
                time_loss_each_step.append(sample['extra']['time_loss'])
                traffic_light_each_step.append(sample['extra']['traffic_light'])
                relative_occupancy_each_step.append(sample['extra']['relative_occupancy'])
                relative_mean_speed_each_step.append(sample['extra']['relative_mean_speed'])
                absolute_number_of_cars_each_step.append(sample['extra']['absolute_number_of_cars'])
            sample_num = len(samples)
            f.close()

            traffic_folder = records_dir.rsplit('/', 1)[1]

            consolidate('time_loss', time_loss_each_step, save_path=round_dir, traffic_name=traffic_folder, mode_name='test')
            consolidate('reward', reward_each_step, save_path=round_dir, traffic_name=traffic_folder, mode_name='test')

            consolidate_occupancy_and_speed_inflow_outflow(relative_occupancy_each_step, relative_mean_speed_each_step, 
                dic_traffic_env_conf, save_path=round_dir, traffic_name=traffic_folder, mode_name='test')

            consolidate_phase_and_demand(absolute_number_of_cars_each_step, traffic_light_each_step, 
                dic_traffic_env_conf, records_dir=records_dir, save_path=round_dir, traffic_name=traffic_folder, mode_name='test')

            relative_occupancy_df = pd.DataFrame(relative_occupancy_each_step)
            relative_mean_speed_df = pd.DataFrame(relative_mean_speed_each_step)

            average_reward_each_round.append(np.mean(reward_each_step))
            average_time_loss_each_round.append(np.mean(time_loss_each_step))
            average_relative_occupancy_each_round.append(relative_occupancy_df.mean().to_dict())
            average_relative_mean_speed_each_round.append(relative_mean_speed_df.mean().to_dict())

            # summary items (duration) from csv
            df_vehicle_inter_0 = pd.read_csv(os.path.join(ROOT_DIR + '/' + round_dir, "vehicle_inter_0.csv"),
                                                sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                                names=["vehicle_id", "enter_time", "leave_time"])

            vehicle_in = sum([int(x) for x in (df_vehicle_inter_0["enter_time"].values > 0)])
            vehicle_out = sum([int(x) for x in (df_vehicle_inter_0["leave_time"].values > 0)])
            duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
            ave_duration = np.mean([time for time in duration if not isnan(time)])
            # print(ave_duration)

            real_traffic_vol = 0
            nan_num = 0
            for time in duration:
                if not isnan(time):
                    real_traffic_vol += 1
                else:
                    nan_num += 1

            duration_each_round_list.append(ave_duration)
            queue_length_each_round_list.append(queue_length_each_round / sample_num)
            num_of_vehicle_in.append(vehicle_in)
            num_of_vehicle_out.append(vehicle_out)

            # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)
            if min_queue_length > queue_length_each_round / sample_num:
                min_queue_length = queue_length_each_round / sample_num
                min_queue_length_id = int(round[6:])

            valid_flag = json.load(open(os.path.join(ROOT_DIR, round_dir, "valid_flag.json")))
            #if valid_flag['0']:  # temporary for one intersection

            if num_seg > 1:
                for i, interval in enumerate(range(0, run_counts, 3600)):
                    did = np.bitwise_and(df_vehicle_inter_0["enter_time"].values < interval + 3600,
                                            df_vehicle_inter_0["enter_time"].values > interval)
                    #vehicle_in_seg = sum([int(x) for x in (df_vehicle_inter_0["enter_time"][did].values > 0)])
                    #vehicle_out_seg = sum([int(x) for x in (df_vehicle_inter_0["leave_time"][did].values > 0)])
                    duration_seg = df_vehicle_inter_0["leave_time"][did].values - df_vehicle_inter_0["enter_time"][
                        did].values
                    ave_duration_seg = np.mean([time for time in duration_seg if not isnan(time)])
                    real_traffic_vol_seg = 0
                    nan_num_seg = 0
                    for time in duration_seg:
                        if not isnan(time):
                            real_traffic_vol_seg += 1
                        else:
                            nan_num_seg += 1

                    # print(real_traffic_vol, traffic_vol, traffic_vol - real_traffic_vol, nan_num)

                    if nan_num_seg < nan_thres:
                        # if min_duration[i] > ave_duration and ave_duration > 24:
                        list_duration_seg[i] = ave_duration_seg
                        list_duration_id_seg[i] = int(round[6:])

                    #round_summary = {}
                for j in range(num_seg):
                    key = "min_duration-" + str(j)
                    if key not in round_summary.keys():
                        round_summary[key] = [list_duration_seg[j]]
                    else:
                        round_summary[key].append(list_duration_seg[j])

        except Exception as e:
            duration_each_round_list.append(NAN_LABEL)
            queue_length_each_round_list.append(NAN_LABEL)
            num_of_vehicle_in.append(NAN_LABEL)
            num_of_vehicle_out.append(NAN_LABEL)

        traffic_folder = records_dir.rsplit('/', 1)[1]
        result_dir = os.path.join("summary", memo, traffic_folder)
        if not os.path.exists(ROOT_DIR + '/' + result_dir):
            os.makedirs(ROOT_DIR + '/' + result_dir)
        
        consolidate('time_loss', average_time_loss_each_round, save_path=result_dir, traffic_name=traffic_folder, mode_name='test')
        consolidate('reward', average_reward_each_round, save_path=result_dir, traffic_name=traffic_folder, mode_name='test')

        consolidate_occupancy_and_speed_inflow_outflow(average_relative_occupancy_each_round, average_relative_mean_speed_each_round, 
                dic_traffic_env_conf, save_path=result_dir, traffic_name=traffic_folder, mode_name='test')
        
        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list,
            "vehicle_in": num_of_vehicle_in,
            "vehicle_out": num_of_vehicle_out
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(ROOT_DIR, result_dir, "test_results.csv"))
        if num_seg > 1:
            round_result = pd.DataFrame(round_summary)
            round_result.to_csv(os.path.join(ROOT_DIR + '/' + result_dir, "test_seg_results.csv"), index=False)
            plot_segment_duration(round_summary, result_dir, mode_name="test")
            duration_each_segment_list = round_result.iloc[min_duration_ind][1:].values

            traffic_name, traffic_time = traffic_folder.split('___')
            traffic_time = traffic_time.split('__')
            if traffic_name not in performance_at_min_duration_round:
                performance_at_min_duration_round[traffic_name] = [(duration_each_segment_list, traffic_time)]
            else:
                performance_at_min_duration_round[traffic_name].append((duration_each_segment_list, traffic_time))


        # print(os.path.join(result_dir, "test_results.csv"))

        # total_summary
        total_summary = get_metrics(duration_each_round_list, queue_length_each_round_list,
                                    min_duration, min_duration_ind, min_queue_length, min_queue_length_id,
                                    traffic_folder, total_summary,
                                    mode_name="test", save_path=result_dir, num_rounds=num_rounds,
                                    min_duration2=None)

        traffic_name, traffic_time = traffic_folder.split('___')
        traffic_time = traffic_time.split('__')
        if traffic_name not in performance_duration:
            performance_duration[traffic_name] = [(duration_each_round_list, traffic_time)]
        else:
            performance_duration[traffic_name].append((duration_each_round_list, traffic_time))

    total_result = pd.DataFrame(total_summary)
    if not os.path.exists(ROOT_DIR + '/' + "summary" + '/' + memo):
        os.makedirs(ROOT_DIR + '/' + "summary" + '/' + memo)
    total_result.to_csv(os.path.join(ROOT_DIR, "summary", memo, "total_test_results.csv"))
    figure_dir = os.path.join("summary", memo, "figures")
    if not os.path.exists(ROOT_DIR + '/' + figure_dir):
        os.makedirs(ROOT_DIR + '/' + figure_dir)
    if dic_exp_conf["EARLY_STOP"]:
        performance_duration = padding_duration(performance_duration)
    summary_plot(performance_duration, figure_dir, mode_name="test", num_rounds=num_rounds)
    performance_at_min_duration_round_plot(performance_at_min_duration_round, figure_dir, mode_name="test")


def single_experiment_summary(memo=None, records_dir=None):

    total_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_queue_length": [],
        "min_queue_length_round": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }

    #summary_detail_train(memo, copy.deepcopy(total_summary))
    single_experiment_summary_detail_test(memo, records_dir, copy.deepcopy(total_summary))
    # summary_detail_test_segments(memo, copy.deepcopy(total_summary))

def main(memo=None):
    total_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_queue_length": [],
        "min_queue_length_round": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }
    if not memo:
        memo = "pipeline_500"
    #summary_detail_train(memo, copy.deepcopy(total_summary))
    summary_detail_test(memo, copy.deepcopy(total_summary))
    # summary_detail_test_segments(memo, copy.deepcopy(total_summary))


if __name__ == "__main__":
    total_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_queue_length": [],
        "min_queue_length_round": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }

    memo = "TransferDQN"
    #summary_detail_train(memo, copy.deepcopy(total_summary))
    summary_detail_test(memo, copy.deepcopy(total_summary))
    # summary_detail_baseline(memo)
    #summary_detail_test_segments(memo, copy.deepcopy(total_summary))

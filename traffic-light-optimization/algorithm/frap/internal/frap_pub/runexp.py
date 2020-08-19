import algorithm.frap.internal.frap_pub.config as config
import copy
from algorithm.frap.internal.frap_pub.pipeline import Pipeline
import os
import time
from multiprocessing import Process
import sys
from algorithm.frap.internal.frap_pub.script import get_traffic_volume


def memo_rename(traffic_file_list):
    new_name = ""
    for traffic_file in traffic_file_list:
        if "synthetic" in traffic_file:
            sta = traffic_file.rfind("-") + 1
            print(traffic_file, int(traffic_file[sta:-4]))
            new_name = new_name + "syn" + traffic_file[sta:-4] + "_"
        elif "cross" in traffic_file:
            sta = traffic_file.find("equal_") + len("equal_")
            end = traffic_file.find(".xml")
            new_name = new_name + "uniform" + traffic_file[sta:end] + "_"
        elif "flow" in traffic_file:
            new_name = traffic_file[:-4]
    new_name = new_name[:-1]
    return new_name

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path, external_configurations={}):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf,
                   dic_agent_conf=dic_agent_conf,
                   dic_traffic_env_conf=dic_traffic_env_conf,
                   dic_path=dic_path,
                   external_configurations=external_configurations
                   )
    ppl.run(multi_process=True)

    print("pipeline_wrapper end")
    return


def main(args=None, memo=None, external_configurations={}):

    #traffic_file_list = [
    #    "0_regular-intersection.rou.xml"
    #    #"inter_0_1786.json",
    #]

    traffic_file_list = external_configurations['TRAFFIC_FILE_LIST']
    roadnet_file = external_configurations['ROADNET_FILE']
    number_of_legs = external_configurations['N_LEG']

    process_list = []
    n_workers = args.workers #len(traffic_file_list)

    multi_process = True


    # ind_arg = int(sys.argv[1])

    if not memo:
        memo = "headway_test"

    for traffic_file in traffic_file_list:
        #model_name = "SimpleDQN"
        model_name = args.algorithm
        ratio = 1
        dic_exp_conf_extra = {
            "RUN_COUNTS": args.run_counts,
            "TEST_RUN_COUNTS": args.test_run_counts,
            "MODEL_NAME": model_name,
            "TRAFFIC_FILE": [traffic_file], # here: change to multi_traffic
            #"ROADNET_FILE": "roadnet_1_1.json",
            "ROADNET_FILE": roadnet_file,

            "NUM_ROUNDS": args.run_round,
            "NUM_GENERATORS": 3,

            "MODEL_POOL": False,
            "NUM_BEST_MODEL": 1,

            "PRETRAIN": False,
            "PRETRAIN_NUM_ROUNDS": 20,
            "PRETRAIN_NUM_GENERATORS": 15,

            "AGGREGATE": False,
            "DEBUG": False,
            "EARLY_STOP": False,
        }

        dic_agent_conf_extra = {
            "LEARNING_RATE": args.learning_rate,
            "LR_DECAY": args.lr_decay,
            "MIN_LR": args.min_lr,
            "EPOCHS": args.epochs,
            "SAMPLE_SIZE": args.sample_size,
            "MAX_MEMORY_LEN": 10000,
            "UPDATE_Q_BAR_EVERY_C_ROUND": args.update_q_bar_every_c_round,
            "UPDATE_Q_BAR_FREQ": 5,
            # network

            "N_LAYER": 2,
            "TRAFFIC_FILE": traffic_file,

            "ROTATION": True,
            "ROTATION_INPUT": args.rotation_input,
            "PRIORITY": args.priority,
            "CONFLICT_MATRIX": args.conflict_matrix,

            "EARLY_STOP_LOSS": args.early_stop_loss,
            "DROPOUT_RATE": args.dropout_rate,
            "MERGE": "multiply",  # concat, weight
            "PHASE_SELECTOR": True,
        }

        dic_traffic_env_conf_extra = {
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 10,

            "MIN_ACTION_TIME": args.min_action_time,
            "IF_GUI": args.sumo_gui,
            "DEBUG": False,
            "BINARY_PHASE_EXPANSION": True, # default, args.binary_phase,
            "DONE_ENABLE": args.done,

            "SIMULATOR_TYPE": [
                "sumo",
                "anon"
            ][1],

            "SAVEREPLAY": args.replay,
            "NUM_ROW": 1,
            "NUM_COL": 1,

            "TRAFFIC_FILE": traffic_file,
            #"ROADNET_FILE": "roadnet_1_1.json",
            "ROADNET_FILE": roadnet_file,

            "LIST_STATE_FEATURE": [
                "cur_phase",
                # "time_this_phase",
                # "vehicle_position_img",
                # "vehicle_speed_img",
                # "vehicle_acceleration_img",
                # "vehicle_waiting_time_img",
                "lane_num_vehicle",
                # "lane_num_vehicle_been_stopped_thres01",
                # "lane_num_vehicle_been_stopped_thres1",
                # "lane_queue_length",
                # "lane_num_vehicle_left",
                # "lane_sum_duration_vehicle_left",
                # "lane_sum_waiting_time",
                # "terminal"
            ],

            "DIC_REWARD_INFO": {
                "flickering": 0,
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25
            },

            "LOG_DEBUG": args.debug,

            "N_LEG": number_of_legs,
        }

        if ".json" in traffic_file:
            dic_traffic_env_conf_extra.update({"SIMULATOR_TYPE": "anon"})
        else:
            dic_traffic_env_conf_extra.update({"SIMULATOR_TYPE": "sumo"})

        # if "Lit" == model_name:
        #     dic_traffic_env_conf_extra["BINARY_PHASE_EXPANSION"] = False

        dic_traffic_env_conf_extra_update = _configure_intersection(number_of_legs, args.num_phase)
        dic_traffic_env_conf_extra.update(dic_traffic_env_conf_extra_update)

        dic_phase_expansion = {}
        for i, phase in enumerate(dic_traffic_env_conf_extra["PHASE"]):
            movements = phase.split("_")
            zeros = [0]*len(dic_traffic_env_conf_extra['list_lane_order'])

            for movement in movements:
                zeros[dic_traffic_env_conf_extra["list_lane_order"].index(movement)] = 1

            dic_phase_expansion[i + 1] = zeros
        dic_traffic_env_conf_extra.update(
            {
                "phase_expansion": dic_phase_expansion,
            }
        )

        postfix = "_" + str(args.min_action_time)

        template = "template_ls"

        #if dic_traffic_env_conf_extra["N_LEG"] == 5 or dic_traffic_env_conf_extra["N_LEG"] == 6:
        #    template = "template_{0}_leg".format(dic_traffic_env_conf_extra["N_LEG"])
        #else:
        #    ## ==================== multi_phase ====================
        #    if dic_traffic_env_conf_extra["LANE_NUM"] == config._LS:
        #        template = "template_ls"
        #    elif dic_traffic_env_conf_extra["LANE_NUM"] == config._S:
        #        template = "template_s"
        #    else:
        #        raise ValueError

        print(traffic_file)
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())) + postfix),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())) + postfix),
            "PATH_TO_DATA": os.path.join("data", template),
            "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", traffic_file),
            "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", traffic_file),
            "PATH_TO_ERROR": os.path.join("errors", memo)

        }

        deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
        deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(model_name.upper())),
                                      dic_agent_conf_extra)
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

        if multi_process:
            ppl = Process(target=pipeline_wrapper,
                          args=(deploy_dic_exp_conf,
                                deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                deploy_dic_path,
                                external_configurations))
            process_list.append(ppl)
        else:
            pipeline_wrapper(dic_exp_conf=deploy_dic_exp_conf,
                             dic_agent_conf=deploy_dic_agent_conf,
                             dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                             dic_path=deploy_dic_path,
                             external_configurations=external_configurations)

    if multi_process:
        i = 0
        list_cur_p = []
        for p in process_list:
            if len(list_cur_p) < n_workers:
                print(i)
                p.start()
                list_cur_p.append(p)
                i += 1
            if len(list_cur_p) < n_workers:
                continue

            idle = check_all_workers_working(list_cur_p)

            while idle == -1:
                time.sleep(1)
                idle = check_all_workers_working(
                    list_cur_p)
            del list_cur_p[idle]

        for i in range(len(list_cur_p)):
            p = list_cur_p[i]
            p.join()

    return memo


def _configure_intersection(number_of_legs, number_of_phases):

    if number_of_legs == 5:

        dict_update = \
            {
                "LANE_NUM": {
                    "LEFT": 1,
                    "RIGHT": 0,
                    "STRAIGHT": 1
                },

                # "PHASE": [
                #    '0L_0T',
                #    '1T_3T',
                #    '2T_4T',
                #    '1L_3L',
                #    '2L_4L',
                #    '1L_1T',
                #    '2L_2T',
                #    '3L_3T',
                #    '4L_4T',
                # ],

                # "list_lane_order": ["0L", "0T", "1L", "1T", "2L", "2T", "3L", "3T", "4L", "4T"],
                "list_lane_order": ["0L1", "0L2", "0T", "1L1", "1L2", "1T", "2L1", "2L2", "2T", "3L1", "3L2", "3T",
                                    "4L1", "4L2", "4T"],
            }

        conflicts = {
            '0L1': ['1L1', '1L2', '1T', '2T', '3L2', '4L1'],
            '0L2': ['1L2', '1T', '2L1', '2L2', '3L2', '3T', '4L1', '4L2'],
            '0T': ['1L2', '1T', '2L1', '2L2', '3L1', '4L1', '4L2', '4T'],
            '1L1': ['2L1', '2L2', '2T', '3T', '4L2', '0L1'],
            '1L2': ['2L2', '2T', '3L1', '3L2', '4L2', '4T', '0L1', '0L2'],
            '1T': ['2L2', '2T', '3L1', '3L2', '4L1', '0L1', '0L2', '0T'],
            '2L1': ['3L1', '3L2', '3T', '4T', '0L2', '1L1'],
            '2L2': ['3L2', '3T', '4L1', '4L2', '0L2', '0T', '1L1', '1L2'],
            '2T': ['3L2', '3T', '4L1', '4L2', '0L1', '1L1', '1L2', '1T'],
            '3L1': ['4L1', '4L2', '4T', '0T', '1L2', '2L1'],
            '3L2': ['4L2', '4T', '0L1', '0L2', '1L2', '1T', '2L1', '2L2'],
            '3T': ['4L2', '4T', '0L1', '0L2', '1L1', '2L1', '2L2', '2T'],
            '4L1': ['0L1', '0L2', '0T', '1T', '2L2', '3L1'],
            '4L2': ['0L2', '0T', '1L1', '1L2', '2L2', '2T', '3L1', '3L2'],
            '4T': ['0L2', '0T', '1L1', '1L2', '2L1', '3L1', '3L2', '3T']
        }

        movements = dict_update['list_lane_order']
        phases = []
        for i1 in range(0, len(movements)):
            a1 = movements[i1]
            a1_movements_left = [movement for movement in movements if
                                 movement not in conflicts[a1] and movement != a1]
            for i2 in range(i1 + 1, len(movements)):
                a2 = movements[i2]
                if a2 not in a1_movements_left:
                    continue
                a2_movements_left = [movement for movement in a1_movements_left if
                                     movement not in conflicts[a2] and movement != a2]
                for i3 in range(i2 + 1, len(movements)):
                    a3 = movements[i3]
                    if a3 not in a2_movements_left:
                        continue
                    phases.append(a1 + '_' + a2 + '_' + a3)

        dict_update['PHASE'] = phases

    elif number_of_legs == 4:

        dict_update = \
            {
                "list_lane_order": ["3L", "3T", "1L", "1T", "0L", "0T", "2L", "2T"]
            }

        if number_of_phases == 2:
            dict_update.update(
                {
                    "LANE_NUM": {
                        "LEFT": 0,
                        "RIGHT": 0,
                        "STRAIGHT": 1
                    },

                    "PHASE": [
                        '3T_1T',
                        '0T_2T',
                        # '3L_1L',
                        # '0L_2L',

                    ]
                }
            )
        elif number_of_phases == 4:
            dict_update.update(
                {
                    "LANE_NUM": {
                        "LEFT": 1,
                        "RIGHT": 0,
                        "STRAIGHT": 1
                    },

                    "PHASE": [
                        '3T_1T',
                        '0T_2T',
                        '3L_1L',
                        '0L_2L',

                    ]
                }
            )
        elif number_of_phases == 8:
            dict_update.update(
                {
                    "LANE_NUM": {
                        "LEFT": 1,
                        "RIGHT": 0,
                        "STRAIGHT": 1
                    },

                    "PHASE": [
                        '3T_1T',
                        '0T_2T',
                        '3L_1L',
                        '0L_2L',
                        '3L_3T',
                        '1L_1T',
                        '2L_2T',
                        '0L_0T',
                    ],
                }
            )
    elif number_of_legs == 3:

        dict_update = \
            {
                "LANE_NUM": {
                    "LEFT": 1,
                    "RIGHT": 0,
                    "STRAIGHT": 1
                },

                "PHASE": [
                    '0L_0T',
                    '0T_1L',
                    '0L_2T'
                ],

                "list_lane_order": ["0L", "0T", "1L", "2T"],
            }

    elif number_of_legs == 2:

        dict_update = \
            {
                "LANE_NUM": {
                    "LEFT": 1,
                    "RIGHT": 0,
                    "STRAIGHT": 1
                },

                "PHASE": [
                    '0T_1L',
                    '1L_1T'
                ],

                "list_lane_order": ["0T", "1L", "1T"]
            }
    else:
        print(number_of_legs, " leg error")
        sys.exit()

    return dict_update


if __name__ == "__main__":

    main()

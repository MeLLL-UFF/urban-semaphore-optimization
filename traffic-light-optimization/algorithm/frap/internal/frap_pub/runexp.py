import os
import time
import copy
import json
from multiprocessing import Process

from lxml import etree

from utils import sumo_util

import algorithm.frap.internal.frap_pub.config as config
from algorithm.frap.internal.frap_pub.pipeline import Pipeline
from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR


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


def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path, 
                     external_configurations={}, existing_experiment=None):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf,
                   dic_agent_conf=dic_agent_conf,
                   dic_traffic_env_conf=dic_traffic_env_conf,
                   dic_path=dic_path,
                   external_configurations=external_configurations,
                   existing_experiment=existing_experiment
                   )
    ppl.run(multi_process=True)

    print("pipeline_wrapper end")
    return


def main(args=None, memo=None, external_configurations={}):

    traffic_file_list = external_configurations['TRAFFIC_FILE_LIST']
    roadnet_file = external_configurations['ROADNET_FILE']
    traffic_level_configuration = external_configurations['TRAFFIC_LEVEL_CONFIGURATION']
    number_of_legs = external_configurations['N_LEG']
    number_of_legs_network_compatibility = external_configurations.get('NUMBER_OF_LEGS_NETWORK_COMPATIBILITY', 'same')
    use_sumo_directions_in_movement_detection = external_configurations.get('USE_SUMO_DIRECTIONS_IN_MOVEMENT_DETECTION',
                                                                          False)
    unique_id = external_configurations['UNIQUE_ID']

    process_list = []
    n_workers = args.workers #len(traffic_file_list)

    multi_process = True


    if not memo:
        memo = "headway_test"

    for traffic_file in traffic_file_list:

        postfix = "_" + str(args.min_action_time)

        template = "template_ls"

        suffix = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())) + postfix + '__' + unique_id

        roadnet_file_name = roadnet_file.rsplit('.', 2)[0]
        experiment_name_base = roadnet_file_name + '__' + '_'.join(traffic_level_configuration)

        print(traffic_file)
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", memo, experiment_name_base + "___" + suffix),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, experiment_name_base + "___" + suffix),
            "PATH_TO_DATA": os.path.join("data", template),
            "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", experiment_name_base),
            "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", experiment_name_base),
            "PATH_TO_ERROR": os.path.join("errors", memo)

        }

        output_file = external_configurations['SUMOCFG_PARAMETERS']['--log']

        split_output_filename = output_file.rsplit('.', 2)
        split_output_filename[0] += '___' + suffix
        output_file = '.'.join(split_output_filename)

        external_configurations['SUMOCFG_PARAMETERS']['--log'] = output_file

        execution_base = split_output_filename[0].rsplit('/', 1)[1]
        dic_path_extra["EXECUTION_BASE"] = execution_base

        model_name = args.algorithm
        ratio = 1
        dic_exp_conf_extra = {
            "RUN_COUNTS": args.run_counts,
            "TEST_RUN_COUNTS": args.test_run_counts,
            "MODEL_NAME": model_name,
            "TRAFFIC_FILE": [traffic_file], # here: change to multi_traffic
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

        '''
        if number_of_legs_network_compatibility == 'same':
            dic_traffic_env_conf_extra.update(
                {
                    "list_lane_order_compatibility": dic_traffic_env_conf_extra['list_lane_order']
                }
            )
        else:

            compatibility_dict = _configure_intersection(number_of_legs_network_compatibility, args.num_phase)

            dic_traffic_env_conf_extra.update(
                {
                    "list_lane_order_compatibility": compatibility_dict['list_lane_order']
                }
            )
        '''

        net_file = os.path.join(ROOT_DIR, dic_path_extra["PATH_TO_DATA"], roadnet_file)
        parser = etree.XMLParser(remove_blank_text=True)
        net_xml = etree.parse(net_file, parser)

        movements, movement_to_connection = \
            sumo_util.detect_movements(net_xml, use_sumo_directions_in_movement_detection)
        dic_traffic_env_conf_extra['list_lane_order'] = movements

        serializable_movement_to_connection = dict(copy.deepcopy(movement_to_connection))
        for movement in serializable_movement_to_connection.keys():
            serializable_movement_to_connection[movement] = dict(serializable_movement_to_connection[movement].attrib)
        dic_traffic_env_conf_extra['movement_to_connection'] = serializable_movement_to_connection

        conflicts = sumo_util.detect_movement_conflicts(net_xml, movement_to_connection)
        phases = sumo_util.detect_phases(movements, conflicts)
        dic_traffic_env_conf_extra['PHASE'] = phases
        dic_traffic_env_conf_extra['CONFLICTS'] = conflicts

        phase_expansion = sumo_util.build_phase_expansions(movements, phases)
        dic_traffic_env_conf_extra["phase_expansion"] = phase_expansion

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

    return memo, deploy_dic_path

def continue_(existing_experiment, args=None, memo=None, external_configurations={}):

    process_list = []
    n_workers = args.workers #len(traffic_file_list)

    multi_process = True

    dir_ = os.path.join('TransferDQN', existing_experiment)

    model_dir = "model/" + dir_
    records_dir = "records/" + dir_
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = model_dir
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    with open(os.path.join(ROOT_DIR, records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    with open(os.path.join(ROOT_DIR, records_dir, "exp.conf"), "r") as f:
        dic_exp_conf = json.load(f)
    with open(os.path.join(ROOT_DIR, records_dir, "traffic_env.conf"), "r") as f:
        dic_traffic_env_conf = json.load(f)

    dic_traffic_env_conf['phase_expansion'] = {int(key): value for key, value in dic_traffic_env_conf['phase_expansion'].items()}

    if multi_process:
        ppl = Process(target=pipeline_wrapper,
                        args=(dic_exp_conf,
                            dic_agent_conf,
                            dic_traffic_env_conf,
                            dic_path,
                            external_configurations,
                            existing_experiment))
        process_list.append(ppl)
    else:
        pipeline_wrapper(dic_exp_conf=dic_exp_conf,
                            dic_agent_conf=dic_agent_conf,
                            dic_traffic_env_conf=dic_traffic_env_conf,
                            dic_path=dic_path,
                            external_configurations=external_configurations,
                            existing_experiment=existing_experiment)

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

    return memo, dic_path


if __name__ == "__main__":

    main()

import algorithm.frap.internal.frap_pub.config as config
import copy
from algorithm.frap.internal.frap_pub.pipeline import Pipeline
import os
import time
from multiprocessing import Process

from sympy import Point2D, Segment2D
from lxml import etree

from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR
from algorithm.frap.internal.utils.bidict import bidict
from algorithm.frap.internal.frap_pub.sumo_env import get_intersection_edge_ids, get_connections, get_intersections_ids


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
    number_of_legs_network_compatibility = external_configurations.get('NUMBER_OF_LEGS_NETWORK_COMPATIBILITY', 'same')
    use_sumo_directions_in_movement_detection = external_configurations.get('USE_SUMO_DIRECTIONS_IN_MOVEMENT_DETECTION',
                                                                          False)
    unique_id = external_configurations['UNIQUE_ID']

    process_list = []
    n_workers = args.workers #len(traffic_file_list)

    multi_process = True


    # ind_arg = int(sys.argv[1])

    if not memo:
        memo = "headway_test"

    for traffic_file in traffic_file_list:

        postfix = "_" + str(args.min_action_time)

        template = "template_ls"

        # if dic_traffic_env_conf_extra["N_LEG"] == 5 or dic_traffic_env_conf_extra["N_LEG"] == 6:
        #    template = "template_{0}_leg".format(dic_traffic_env_conf_extra["N_LEG"])
        # else:
        #    ## ==================== multi_phase ====================
        #    if dic_traffic_env_conf_extra["LANE_NUM"] == config._LS:
        #        template = "template_ls"
        #    elif dic_traffic_env_conf_extra["LANE_NUM"] == config._S:
        #        template = "template_s"
        #    else:
        #        raise ValueError

        suffix = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())) + postfix + '_' + unique_id

        print(traffic_file)
        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join("model", memo, traffic_file + "_" + suffix),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, traffic_file + "_" + suffix),
            "PATH_TO_DATA": os.path.join("data", template),
            "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", traffic_file),
            "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", traffic_file),
            "PATH_TO_ERROR": os.path.join("errors", memo)

        }

        output_file = external_configurations['SUMOCFG_PARAMETERS']['--log']

        split_output_filename = output_file.rsplit('.', 2)
        split_output_filename[0] += '_' + suffix
        output_file = '.'.join(split_output_filename)

        external_configurations['SUMOCFG_PARAMETERS']['--log'] = output_file

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
            _detect_movements(net_xml, use_sumo_directions_in_movement_detection)
        dic_traffic_env_conf_extra['list_lane_order'] = movements

        conflicts = _detect_movement_conflicts(net_xml, movement_to_connection)
        phases = _detect_phases(movements, conflicts)
        dic_traffic_env_conf_extra['PHASE'] = phases

        phase_expansion = _build_phase_expansions(movements, phases)
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

    return memo


def _detect_movements(net_xml, use_sumo_directions=False, is_right_on_red=True):

    incoming_edges, _ = get_intersection_edge_ids(net_xml)

    movement_to_connection = bidict()

    movements = []
    for edge_index, edge in enumerate(incoming_edges):

        connections = get_connections(net_xml, from_edge=edge)

        sorted_connections = list(reversed(connections))

        if use_sumo_directions:
            dir_to_from_lane = {}
            for connection in sorted_connections:

                from_lane = connection.get('fromLane')
                dir = connection.get('dir').lower()
                if dir in dir_to_from_lane:
                    dir_to_from_lane[dir].append(from_lane)
                else:
                    dir_to_from_lane[dir] = [from_lane]

            for connection in sorted_connections:

                dir = connection.get('dir').lower()
                from_lane = connection.get('fromLane')

                dir_from_lane = dir_to_from_lane[dir]
                if len(dir_from_lane) == 1:
                    dir_label = dir.upper()
                else:
                    dir_label = dir.upper() + str(dir_to_from_lane[dir].index(from_lane) + 1)

                movement = str(edge_index) + dir_label

                movements.append(movement)

                movement_to_connection[movement] = connection

        else:
            dir_labels = [None]*len(sorted_connections)
            if sorted_connections[0].get('dir').lower() == 'l':
                dir_labels[0] = 'L'
            if sorted_connections[len(sorted_connections) - 1].get('dir').lower() == 'r':
                dir_labels[len(sorted_connections) - 1] = 'R'
            count = 0
            for index, dir_label in enumerate(dir_labels):
                if dir_label is None:
                    if count == 0:
                        dir_labels[index] = 'S'
                    else:
                        dir_labels[index] = 'S' + str(count)
                    count += 1

            for index, connection in enumerate(sorted_connections):

                movement = str(edge_index) + dir_labels[index]

                if is_right_on_red and dir_labels[index] != 'R':
                    movements.append(movement)

                movement_to_connection[movement] = connection

    return movements, movement_to_connection

def _detect_movement_conflicts(net_xml, movement_to_connection):

    conflicts = {}

    incoming_edges, outgoing_edges = get_intersection_edge_ids(net_xml)
    connections = get_connections(net_xml)

    all_edges = incoming_edges + outgoing_edges

    intersection_id = get_intersections_ids(net_xml)[0]
    intersection = net_xml.find(".//junction[@id='" + intersection_id + "']")
    intersection_point = Point2D(intersection.get('x'), intersection.get('y'))

    lane_to_movement_point = {}

    for edge in all_edges:

        lanes = net_xml.findall(".//edge[@id='" + edge + "']/lane")

        for lane in lanes:

            lane_id = lane.get('id')

            lane_points = lane.get('shape').split()
            first_lane_point = Point2D(lane_points[0].split(','))
            last_lane_point = Point2D(lane_points[-1].split(','))

            if intersection_point.distance(first_lane_point) < intersection_point.distance(last_lane_point):
                movement_lane_point = first_lane_point
            else:
                movement_lane_point = last_lane_point

            lane_to_movement_point[lane_id] = movement_lane_point

    same_lane_origin_movements = {}
    for index_1 in range(0, len(connections)):
        for index_2 in range(index_1 + 1, len(connections)):

            connection_1 = connections[index_1]
            connection_2 = connections[index_2]

            connection_1_from_lane = connection_1.get('from') + '_' + connection_1.get('fromLane')
            connection_1_to_lane = connection_1.get('to') + '_' + connection_1.get('toLane')

            connection_2_from_lane = connection_2.get('from') + '_' + connection_2.get('fromLane')
            connection_2_to_lane = connection_2.get('to') + '_' + connection_2.get('toLane')

            connection_1_line = \
                Segment2D(lane_to_movement_point[connection_1_from_lane], lane_to_movement_point[connection_1_to_lane])

            connection_2_line = \
                Segment2D(lane_to_movement_point[connection_2_from_lane], lane_to_movement_point[connection_2_to_lane])

            line_intersections = connection_1_line.intersection(connection_2_line)

            movement_1 = movement_to_connection.inverse[connection_1][0]
            movement_2 = movement_to_connection.inverse[connection_2][0]
            if connection_1_line.p1 == connection_2_line.p1:
                if movement_1 in same_lane_origin_movements:
                    same_lane_origin_movements[movement_1].append(movement_2)
                else:
                    same_lane_origin_movements[movement_1] = [movement_2]

                if movement_2 in same_lane_origin_movements:
                    same_lane_origin_movements[movement_2].append(movement_1)
                else:
                    same_lane_origin_movements[movement_2] = [movement_1]

            elif len(line_intersections) > 0:
                if movement_1 in conflicts:
                    conflicts[movement_1].append(movement_2)
                else:
                    conflicts[movement_1] = [movement_2]

                if movement_2 in conflicts:
                    conflicts[movement_2].append(movement_1)
                else:
                    conflicts[movement_2] = [movement_1]
            else:
                if movement_1 not in conflicts:
                    conflicts[movement_1] = []

                if movement_2 not in conflicts:
                    conflicts[movement_2] = []

    for key, values in same_lane_origin_movements.items():
        original_conflicts = set(conflicts[key])

        for value in values:

            inherited_conflicts = conflicts[value]

            original_conflicts.update(set(inherited_conflicts))

            for inherited_conflict in inherited_conflicts:
                original_conflicts.update(set(same_lane_origin_movements[inherited_conflict]))

        conflicts[key] = list(original_conflicts)

    return conflicts

def _detect_phases(movements, conflicts, is_right_on_red=True):

    if is_right_on_red:
        movements = [movement for movement in movements if 'R' not in movement]

    phases = []

    depth_first_search_tracking = [copy.deepcopy(movements)]
    movements_left_list = [movements]
    elements_tracking = []

    i = [-1]
    while len(depth_first_search_tracking) != 0:

        while len(depth_first_search_tracking[0]) != 0:
            i[0] += 1

            element = depth_first_search_tracking[0].pop(0)
            elements_tracking.append(element)

            movements_left = [movement for movement in movements[i[0]+1:]
                              if movement not in conflicts[element] + [element] and
                              movement in movements_left_list[-1]]

            movements_left_list.append(movements_left)

            if movements_left:
                depth_first_search_tracking = [movements_left] + depth_first_search_tracking
                i = [i[0]] + i
            else:
                phases.append('_'.join(elements_tracking))
                elements_tracking.pop()
                movements_left_list.pop()

        depth_first_search_tracking.pop(0)
        if elements_tracking:
            elements_tracking.pop()
            i.pop(0)
        movements_left_list.pop()

    phase_sets = [set(phase.split('_')) for phase in phases]

    indices_to_remove = set()
    for i in range(0, len(phase_sets)):
        for j in range(i+1, len(phase_sets)):

            phase_i = phase_sets[i]
            phase_j = phase_sets[j]

            if phase_i.issubset(phase_j):
                indices_to_remove.add(i)
            elif phase_j.issubset(phase_i):
                indices_to_remove.add(j)

    indices_to_remove = sorted(indices_to_remove, reverse=True)
    for index_to_remove in indices_to_remove:
        phases.pop(index_to_remove)
        phase_sets.pop(index_to_remove)

    return phases

def _build_phase_expansions(movements, phases):

    phase_expansion = {}
    for i, phase in enumerate(phases):
        phase_movements = phase.split("_")
        zeros = [0] * len(movements)

        for phase_movement in phase_movements:
            zeros[movements.index(phase_movement)] = 1

        phase_expansion[i + 1] = zeros

    return phase_expansion


if __name__ == "__main__":

    main()

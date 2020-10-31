import os
import time
import json
import pickle
from multiprocessing import Process

import numpy as np

from algorithm.frap.internal.frap_pub.config import DIC_AGENTS, DIC_ENVS

from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--memo", type=str, default="default")
    parser.add_argument("--round", type=int, default=0)

    return parser.parse_args()


def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i

    return -1


def downsample(path_to_log):
    path_to_pkl = os.path.join(path_to_log, "inter_0.pkl")
    with open(ROOT_DIR + '/' + path_to_pkl, "rb") as f_logging_data:
        logging_data = pickle.load(f_logging_data)
    subset_data = logging_data[::10]
    os.remove(ROOT_DIR + '/' + path_to_pkl)
    with open(ROOT_DIR + '/' + path_to_pkl, "wb") as f_subset:
        pickle.dump(subset_data, f_subset)


def run(dir, round_number, run_cnt, execution_name, if_gui, rewrite_mode=False, external_configurations={}):
    model_dir = "model/" + dir
    records_dir = "records/" + dir
    model_round = 'round' + '_' + str(round_number)
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

    dic_exp_conf["RUN_COUNTS"] = run_cnt
    dic_traffic_env_conf["IF_GUI"] = if_gui
    dic_traffic_env_conf["SAVEREPLAY"] = True

    # dump dic_exp_conf
    with open(os.path.join(ROOT_DIR, records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)

    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
        dic_agent_conf["MIN_EPSILON"] = 0

    agent_name = dic_exp_conf["MODEL_NAME"]
    agent = DIC_AGENTS[agent_name](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
        dic_exp_conf=dic_exp_conf,
        cnt_round=round_number + 1,  # useless
        mode='replay'
    )
    agent.load_network(model_round)

    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
    if not os.path.exists(ROOT_DIR + '/' + path_to_log):
        os.makedirs(ROOT_DIR + '/' + path_to_log)
    
    env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](path_to_log=path_to_log,
                        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
                        dic_traffic_env_conf=dic_traffic_env_conf,
                        dic_path=dic_path,
                        external_configurations=external_configurations,
                        mode='replay', write_mode=rewrite_mode)

    if agent_name == 'PlanningOnly' or agent_name == 'TransferDQNwithPlanning':
        agent.set_simulation_environment(env)

    done = False
    state, next_action = env.reset(execution_name)
    step = 0

    while not done and step < dic_exp_conf["RUN_COUNTS"]:
        action_list = [None]*len(next_action)
        
        new_actions_needed = np.where(np.array(next_action) == None)[0]
        for index in new_actions_needed:
            
            one_state = state[index]

            action = agent.choose_action(step, one_state, intersection_index=index)

            action_list[index] = action

        next_state, reward, done, steps_iterated, next_action, _ = env.step(action_list)

        state = next_state
        step += steps_iterated
    env.bulk_log()
    env.end_sumo()
    
    if rewrite_mode:
        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        downsample(path_to_log)



def run_wrapper(dir, one_round, run_cnt, if_gui, external_configurations={}):
    model_dir = "model/" + dir
    records_dir = "records/" + dir
    model_round = one_round
    dic_path = {}
    dic_path["PATH_TO_MODEL"] = model_dir
    dic_path["PATH_TO_WORK_DIRECTORY"] = records_dir

    with open(os.path.join(ROOT_DIR, records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    with open(os.path.join(ROOT_DIR, records_dir, "exp.conf"), "r") as f:
        dic_exp_conf = json.load(f)
    with open(os.path.join(ROOT_DIR, records_dir, "traffic_env.conf"), "r") as f:
        dic_traffic_env_conf = json.load(f)

    dic_exp_conf["RUN_COUNTS"] = run_cnt
    dic_traffic_env_conf["IF_GUI"] = if_gui
    dic_traffic_env_conf["SAVEREPLAY"] = True

    # dump dic_exp_conf
    with open(os.path.join(ROOT_DIR, records_dir, "test_exp.conf"), "w") as f:
        json.dump(dic_exp_conf, f)

    if dic_exp_conf["MODEL_NAME"] in dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
        dic_agent_conf["EPSILON"] = 0  # dic_agent_conf["EPSILON"]  # + 0.1*cnt_gen
        dic_agent_conf["MIN_EPSILON"] = 0

    agent_name = dic_exp_conf["MODEL_NAME"]
    agent = DIC_AGENTS[agent_name](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path,
        dic_exp_conf=dic_exp_conf,
        cnt_round=0,  # useless
        mode='replay'
    )
    if 1:
        agent.load_network(model_round)

        path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
        if not os.path.exists(ROOT_DIR + '/' + path_to_log):
            os.makedirs(ROOT_DIR + '/' + path_to_log)
        env = DIC_ENVS[dic_traffic_env_conf["SIMULATOR_TYPE"]](path_to_log=path_to_log,
                         path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
                         dic_traffic_env_conf=dic_traffic_env_conf,
                         dic_path=dic_path,
                         external_configurations=external_configurations,
                         mode='replay')

        if agent_name == 'PlanningOnly' or agent_name == 'TransferDQNwithPlanning':
            agent.set_simulation_environment(env)

        done = False
        state, next_action = env.reset()
        step = 0

        while not done and step < dic_exp_conf["RUN_COUNTS"]:
            action_list = [None]*len(next_action)
            
            new_actions_needed = np.where(np.array(next_action) == None)[0]
            for index in new_actions_needed:
                
                one_state = state[index]

                action = agent.choose_action(step, one_state, intersection_index=index)

                action_list[index] = action

            next_state, reward, done,  steps_iterated, next_action, _ = env.step(action_list)

            state = next_state
            step += steps_iterated
        env.bulk_log()
        env.end_sumo()
        if not __debug__:
            path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
            # print("downsample", path_to_log)
            downsample(path_to_log)
            # print("end down")

    #except:
    #    pass
        # import sys
        # sys.stderr.write("fail to test model_%"%model_round)
        # raise SystemExit(1)

    return


def main(memo=None, external_configurations={}):
    # run name
    if not memo:
        memo = "learning_rate/anon_2_phase_done"

    #args = parse_args()

    # test run_count
    run_cnt = 3600

    # add the specific rounds in the given_round_list, like [150, 160]
    # if none, test all the round
    given_round_list = [7]

    given_traffic_list = [
        # "cross.2phases_rou01_equal_650.xml",
        # "cross.2phases_rou01_equal_600.xml",
        # "cross.2phases_rou01_equal_550.xml",
        # "cross.2phases_rou01_equal_500.xml",
        # "cross.2phases_rou01_equal_450.xml",
        # "cross.2phases_rou01_equal_400.xml",
        # "cross.2phases_rou01_equal_350.xml",
        # "cross.2phases_rou01_equal_300.xml",
    ]

    if_gui = True

    multi_process = True
    n_workers = 100
    process_list = []
    for traffic in os.listdir(ROOT_DIR + '/' + "records/" + memo):
        print(traffic)
        if not ".xml" in traffic and not ".json" in traffic:
            continue

        if traffic != "flow_1_1_700.json_01_06_02_45_01_10":
            continue
        test_round_dir = os.path.join("records", memo, traffic, "test_round")
        if os.path.exists(ROOT_DIR + '/' + test_round_dir):
            print("exist")
            #continue
        # if traffic[0:-15] not in given_traffic_list:
        #    continue

        work_dir = os.path.join(memo, traffic)

        if given_round_list:
            for one_round in given_round_list:
                _round = "round_" + str(one_round)
                if multi_process:
                    p = Process(target=run_wrapper, args=(work_dir, _round, run_cnt, if_gui, external_configurations))
                    process_list.append(p)
                else:
                    run_wrapper(work_dir, _round, run_cnt, if_gui, external_configurations=external_configurations)
        else:
            train_round_dir = os.path.join("records", memo, traffic, "train_round")
            for one_round in os.listdir(ROOT_DIR + '/' + train_round_dir):
                if "round" not in one_round:
                    continue

                if multi_process:
                    p = Process(target=run_wrapper, args=(work_dir, one_round, run_cnt, if_gui, external_configurations))
                    process_list.append(p)
                else:
                    run_wrapper(work_dir, one_round, run_cnt, if_gui, external_configurations=external_configurations)

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

        for p in list_cur_p:
            p.join()


if __name__ == "__main__":
    main()

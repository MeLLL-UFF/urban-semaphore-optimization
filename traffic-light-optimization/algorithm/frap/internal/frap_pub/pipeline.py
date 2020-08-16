import json
import os
import shutil
import xml.etree.ElementTree as ET
from algorithm.frap.internal.frap_pub.generator import Generator
from algorithm.frap.internal.frap_pub.construct_sample import ConstructSample
from algorithm.frap.internal.frap_pub.updater import Updater
from multiprocessing import Process
from algorithm.frap.internal.frap_pub.model_pool import ModelPool
import random
import pickle
import algorithm.frap.internal.frap_pub.model_test as model_test
import pandas as pd
import numpy as np
from math import isnan
import sys
import time

from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR

class Pipeline:
    #_LIST_SUMO_FILES = [
    #    "0_regular-intersection__right_on_red.sumocfg",
    #    "0_regular-intersection__right_on_red.net.xml"
    #    #"cross.car.type.xml",
    #    #"cross.con.xml",
    #    #"cross.edg.xml",
    #    #"cross.net.xml",
    #    #"cross.netccfg",
    #    #"cross.nod.xml",
    #    #"cross.sumocfg",
    #    #"cross.tll.xml",
    #    #"cross.typ.xml"
    #]

    @staticmethod
    def _set_traffic_file(sumo_config_file_tmp_name, sumo_config_file_output_name, list_traffic_file_name):

        # update sumocfg
        sumo_cfg = ET.parse(ROOT_DIR + '/' + sumo_config_file_tmp_name)
        config_node = sumo_cfg.getroot()
        input_node = config_node.find("input")
        for route_files in input_node.findall("route-files"):
            input_node.remove(route_files)
        input_node.append(
            ET.Element("route-files", attrib={"value": ",".join(list_traffic_file_name)}))
        sumo_cfg.write(ROOT_DIR + '/' + sumo_config_file_output_name)

    def _path_check(self):
        # check path
        if os.path.exists(ROOT_DIR + '/' + self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            if ROOT_DIR + '/' + self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(ROOT_DIR + '/' + self.dic_path["PATH_TO_WORK_DIRECTORY"])

        if os.path.exists(ROOT_DIR + '/' + self.dic_path["PATH_TO_MODEL"]):
            if ROOT_DIR + '/' + self.dic_path["PATH_TO_MODEL"] != "model/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(ROOT_DIR + '/' + self.dic_path["PATH_TO_MODEL"])

        if self.dic_exp_conf["PRETRAIN"]:
            if os.path.exists(ROOT_DIR + '/' + self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
                pass
            else:
                os.makedirs(ROOT_DIR + '/' + self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

            if os.path.exists(ROOT_DIR + '/' + self.dic_path["PATH_TO_PRETRAIN_MODEL"]):
                pass
            else:
                os.makedirs(ROOT_DIR + '/' + self.dic_path["PATH_TO_PRETRAIN_MODEL"])

    def _copy_conf_file(self, path=None):
        # write conf files
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(ROOT_DIR, path, "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(ROOT_DIR, path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(ROOT_DIR, path, "traffic_env.conf"), "w"), indent=4)

    def _copy_sumo_file(self, path=None, _list_sumo_files=[]):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files
        for file_name in _list_sumo_files:
            shutil.copy(os.path.join(ROOT_DIR, self.dic_path["PATH_TO_DATA"], file_name),
                        os.path.join(ROOT_DIR, path, file_name))
        for file_name in self.dic_exp_conf["TRAFFIC_FILE"]:
            shutil.copy(os.path.join(ROOT_DIR, self.dic_path["PATH_TO_DATA"], file_name),
                        os.path.join(ROOT_DIR, path, file_name))

    def _copy_anon_file(self, path=None):
        # hard code !!!
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # copy sumo files

        shutil.copy(os.path.join(ROOT_DIR, self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["TRAFFIC_FILE"][0]),
                        os.path.join(ROOT_DIR, path, self.dic_exp_conf["TRAFFIC_FILE"][0]))
        shutil.copy(os.path.join(ROOT_DIR, self.dic_path["PATH_TO_DATA"], self.dic_exp_conf["ROADNET_FILE"]),
                    os.path.join(ROOT_DIR, path, self.dic_exp_conf["ROADNET_FILE"]))

    def _modify_sumo_file(self, path=None, sumocfg_file=''):
        if path == None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        # modify sumo files
        self._set_traffic_file(os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"],
                                            sumocfg_file),
                               os.path.join(path, sumocfg_file),
                               self.dic_exp_conf["TRAFFIC_FILE"])

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path, external_configurations={}):

        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.external_configurations = external_configurations

        # do file operations
        self._path_check()
        self._copy_conf_file()
        if self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'sumo':
            _list_sumo_files = self.external_configurations['_LIST_SUMO_FILES']
            sumocfg_file = self.external_configurations['SUMOCFG_FILE']
            self._copy_sumo_file(_list_sumo_files=_list_sumo_files)
            self._modify_sumo_file(sumocfg_file=sumocfg_file)
        elif self.dic_traffic_env_conf["SIMULATOR_TYPE"] == 'anon':
            self._copy_anon_file()
        # test_duration
        self.test_duration = []

    def early_stopping(self, dic_path, cnt_round):
        print("decide whether to stop")
        record_dir = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", "round_"+str(cnt_round))

        # compute duration
        df_vehicle_inter_0 = pd.read_csv(os.path.join(ROOT_DIR, record_dir, "vehicle_inter_0.csv"),
                                         sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                         names=["vehicle_id", "enter_time", "leave_time"])
        duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
        ave_duration = np.mean([time for time in duration if not isnan(time)])
        self.test_duration.append(ave_duration)
        if len(self.test_duration) < 30:
            return 0
        else:
            duration_under_exam = np.array(self.test_duration[-15:])
            mean_duration = np.mean(duration_under_exam)
            std_duration = np.std(duration_under_exam)
            max_duration = np.max(duration_under_exam)
            if std_duration/mean_duration < 0.1 and max_duration < 1.5 * mean_duration:
                return 1
            else:
                return 0



    def generator_wrapper(self, cnt_round, cnt_gen, dic_path, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                          best_round=None):
        generator = Generator(cnt_round=cnt_round,
                              cnt_gen=cnt_gen,
                              dic_path=dic_path,
                              dic_exp_conf=dic_exp_conf,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              best_round=best_round,
                              external_configurations=self.external_configurations
                              )
        print("make generator")
        generator.generate()
        print("generator_wrapper end")
        return

    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, best_round=None, bar_round=None):

        updater = Updater(
            cnt_round=cnt_round,
            dic_agent_conf=dic_agent_conf,
            dic_exp_conf=dic_exp_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            best_round=best_round,
            bar_round=bar_round
        )

        updater.load_sample()
        updater.update_network()
        print("updater_wrapper end")
        return

    def model_pool_wrapper(self, dic_path, dic_exp_conf, cnt_round):
        model_pool = ModelPool(dic_path, dic_exp_conf)
        model_pool.model_compare(cnt_round)
        model_pool.dump_model_pool()

        return
        #self.best_round = model_pool.get()
        #print("self.best_round", self.best_round)

    def downsample(self, path_to_log):

        path_to_pkl = os.path.join(path_to_log, "inter_0.pkl")
        f_logging_data = open(ROOT_DIR + '/' + path_to_pkl, "rb")
        logging_data = pickle.load(f_logging_data)
        subset_data = logging_data[::10]
        f_logging_data.close()
        os.remove(ROOT_DIR + '/' + path_to_pkl)
        f_subset = open(ROOT_DIR + '/' + path_to_pkl, "wb")
        pickle.dump(subset_data, f_subset)
        f_subset.close()

    def run(self, multi_process=True):

        best_round, bar_round = None, None
        # pretrain for acceleration
        if self.dic_exp_conf["PRETRAIN"]:
            if os.listdir(ROOT_DIR + '/' + self.dic_path["PATH_TO_PRETRAIN_MODEL"]):
                shutil.copy(os.path.join(ROOT_DIR, self.dic_path["PATH_TO_PRETRAIN_MODEL"],
                                         "%s.h5" % self.dic_exp_conf["TRAFFIC_FILE"][0]),
                            os.path.join(ROOT_DIR, self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            else:
                if not os.listdir(ROOT_DIR + '/' + self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
                    for cnt_round in range(self.dic_exp_conf["PRETRAIN_NUM_ROUNDS"]):
                        print("round %d starts" % cnt_round)

                        process_list = []

                        # ==============  generator =============
                        if multi_process:
                            for cnt_gen in range(self.dic_exp_conf["PRETRAIN_NUM_GENERATORS"]):
                                p = Process(target=self.generator_wrapper,
                                            args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                                  self.dic_agent_conf, self.dic_sumo_env_conf, best_round)
                                            )
                                print("before")
                                p.start()
                                print("end")
                                process_list.append(p)
                            print("before join")
                            for p in process_list:
                                p.join()
                            print("end join")
                        else:
                            for cnt_gen in range(self.dic_exp_conf["PRETRAIN_NUM_GENERATORS"]):
                                self.generator_wrapper(cnt_round=cnt_round,
                                                       cnt_gen=cnt_gen,
                                                       dic_path=self.dic_path,
                                                       dic_exp_conf=self.dic_exp_conf,
                                                       dic_agent_conf=self.dic_agent_conf,
                                                       dic_sumo_env_conf=self.dic_sumo_env_conf,
                                                       best_round=best_round)

                        # ==============  make samples =============
                        # make samples and determine which samples are good

                        train_round = os.path.join(self.dic_path["PATH_TO_PRETRAIN_WORK_DIRECTORY"],
                                                   "train_round")
                        if not os.path.exists(ROOT_DIR + '/' + train_round):
                            os.makedirs(ROOT_DIR + '/' + train_round)
                        cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                             dic_sumo_env_conf=self.dic_sumo_env_conf)
                        cs.make_reward()

                if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                    if multi_process:
                        p = Process(target=self.updater_wrapper,
                                    args=(0,
                                          self.dic_agent_conf,
                                          self.dic_exp_conf,
                                          self.dic_sumo_env_conf,
                                          self.dic_path,
                                          best_round))
                        p.start()
                        p.join()
                    else:
                        self.updater_wrapper(cnt_round=0,
                                             dic_agent_conf=self.dic_agent_conf,
                                             dic_exp_conf=self.dic_exp_conf,
                                             dic_sumo_env_conf=self.dic_sumo_env_conf,
                                             dic_path=self.dic_path,
                                             best_round=best_round)
        # train with aggregate samples
        if self.dic_exp_conf["AGGREGATE"]:
            if "aggregate.h5" in os.listdir(ROOT_DIR + '/' + "model/initial"):
                shutil.copy(ROOT_DIR + '/' + "model/initial/aggregate.h5",
                            os.path.join(ROOT_DIR, self.dic_path["PATH_TO_MODEL"], "round_0.h5"))
            else:
                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(0,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_sumo_env_conf,
                                      self.dic_path,
                                      best_round))
                    p.start()
                    p.join()
                else:
                    self.updater_wrapper(cnt_round=0,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_sumo_env_conf=self.dic_sumo_env_conf,
                                         dic_path=self.dic_path,
                                         best_round=best_round)

        self.dic_exp_conf["PRETRAIN"] = False
        self.dic_exp_conf["AGGREGATE"] = False

        # train
        for cnt_round in range(self.dic_exp_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)

            round_start_t = time.time()

            process_list = []

            # ==============  generator =============
            if multi_process:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    p = Process(target=self.generator_wrapper,
                                args=(cnt_round, cnt_gen, self.dic_path, self.dic_exp_conf,
                                      self.dic_agent_conf, self.dic_traffic_env_conf, best_round)
                                )
                    p.start()
                    process_list.append(p)
                for i in range(len(process_list)):
                    p = process_list[i]
                    p.join()
            else:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    self.generator_wrapper(cnt_round=cnt_round,
                                           cnt_gen=cnt_gen,
                                           dic_path=self.dic_path,
                                           dic_exp_conf=self.dic_exp_conf,
                                           dic_agent_conf=self.dic_agent_conf,
                                           dic_traffic_env_conf=self.dic_traffic_env_conf,
                                           best_round=best_round)

            # ==============  make samples =============
            # make samples and determine which samples are good

            train_round = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
            if not os.path.exists(ROOT_DIR + '/' + train_round):
                os.makedirs(ROOT_DIR + '/' + train_round)
            cs = ConstructSample(path_to_samples=train_round, cnt_round=cnt_round,
                                 dic_traffic_env_conf=self.dic_traffic_env_conf)
            cs.make_reward()

            # EvaluateSample()

            # ==============  update network =============
            if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                if multi_process:
                    p = Process(target=self.updater_wrapper,
                                args=(cnt_round,
                                      self.dic_agent_conf,
                                      self.dic_exp_conf,
                                      self.dic_traffic_env_conf,
                                      self.dic_path,
                                      best_round,
                                      bar_round))
                    p.start()
                    p.join()
                else:
                    self.updater_wrapper(cnt_round=cnt_round,
                                         dic_agent_conf=self.dic_agent_conf,
                                         dic_exp_conf=self.dic_exp_conf,
                                         dic_traffic_env_conf=self.dic_traffic_env_conf,
                                         dic_path=self.dic_path,
                                         best_round=best_round,
                                         bar_round=bar_round)

            if not self.dic_exp_conf["DEBUG"]:
                for cnt_gen in range(self.dic_exp_conf["NUM_GENERATORS"]):
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "train_round",
                                               "round_" + str(cnt_round), "generator_" + str(cnt_gen))
                    self.downsample(path_to_log)

            # ==============  test evaluation =============
            if multi_process:
                p = Process(target=model_test.test,
                            args=(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["TEST_RUN_COUNTS"],
                                  self.dic_traffic_env_conf, False,
                                  self.external_configurations))
                p.start()
                if self.dic_exp_conf["EARLY_STOP"] or self.dic_exp_conf["MODEL_POOL"]:
                    p.join()
            else:
                model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round, self.dic_exp_conf["RUN_COUNTS"],
                                self.dic_traffic_env_conf, if_gui=False,
                                external_configurations=self.external_configurations)

            # ==============  early stopping =============
            if self.dic_exp_conf["EARLY_STOP"]:
                flag = self.early_stopping(self.dic_path, cnt_round)
                if flag == 1:
                    break

            # ==============  model pool evaluation =============
            if self.dic_exp_conf["MODEL_POOL"]:
                if multi_process:
                    p = Process(target=self.model_pool_wrapper,
                                args=(self.dic_path,
                                      self.dic_exp_conf,
                                      cnt_round),
                                )
                    p.start()
                    p.join()
                else:
                    self.model_pool_wrapper(dic_path=self.dic_path,
                                            dic_exp_conf=self.dic_exp_conf,
                                            cnt_round=cnt_round)
                model_pool_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "best_model.pkl")
                if os.path.exists(ROOT_DIR + '/' + model_pool_dir):
                    model_pool = pickle.load(open(ROOT_DIR + '/' + model_pool_dir, "rb"))
                    ind = random.randint(0, len(model_pool) - 1)
                    best_round = model_pool[ind][0]
                    ind_bar = random.randint(0, len(model_pool) - 1)
                    flag = 0
                    while ind_bar == ind and flag < 10:
                        ind_bar = random.randint(0, len(model_pool) - 1)
                        flag += 1
                    # bar_round = model_pool[ind_bar][0]
                    bar_round = None
                else:
                    best_round = None
                    bar_round = None

                # downsample
                if not self.dic_exp_conf["DEBUG"]:
                    path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round",
                                               "round_" + str(cnt_round))
                    self.downsample(path_to_log)
            else:
                best_round = None

            print("best_round: ", best_round)

            print("round %s ends" % cnt_round)

            round_end_t = time.time()
            f_timing = open(os.path.join(ROOT_DIR, self.dic_path["PATH_TO_WORK_DIRECTORY"], "timing.txt"), "a+")
            f_timing.write("round_{0}: {1}\n".format(cnt_round, round_end_t-round_start_t))
            f_timing.close()

            return

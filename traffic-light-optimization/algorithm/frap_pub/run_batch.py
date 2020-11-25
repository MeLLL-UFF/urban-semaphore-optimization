import os
import argparse
import time

from algorithm.frap_pub import config
import algorithm.frap_pub.runexp as runexp
import algorithm.frap_pub.summary as summary
from algorithm.frap_pub.definitions import ROOT_DIR


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--memo", type=str, default=config.DIC_EXP_CONF['MODEL_NAME'])
    parser.add_argument("--workers", type=int, default=12)

    #parser.add_argument("--visible_gpu", type=str, default="")

    return parser.parse_args()


def run(external_configurations=None):

    if external_configurations is None:
        external_configurations = {}

    args = parse_args()
    memo = args.memo

    t1 = time.time()
    _, dic_path = runexp.main(args, memo, external_configurations)
    print("****************************** runexp ends (generate, train, test)!! ******************************")
    t2 = time.time()
    f_timing = open(os.path.join(ROOT_DIR, "records", memo, "timing.txt"), "a+")
    f_timing.write(str(t2 - t1) + '\n')
    f_timing.close()
    records_dir = dic_path["PATH_TO_WORK_DIRECTORY"]
    summary.single_experiment_summary(memo, records_dir, plots='summary_only')
    print("****************************** summary_detail ends ******************************")
    experiment_name = dic_path["EXECUTION_BASE"]

    return experiment_name

def continue_(experiment, external_configurations=None):

    if external_configurations is None:
        external_configurations = {}

    args = parse_args()
    memo = args.memo

    _, dic_path = runexp.continue_(experiment, 'FROM_THE_LAST', args, memo, external_configurations)
    print("****************************** runexp ends (generate, train, test)!! ******************************")
    records_dir = dic_path["PATH_TO_WORK_DIRECTORY"]
    summary.single_experiment_summary(memo, records_dir, plots='summary_only')
    print("****************************** summary_detail ends ******************************")

    return experiment

def re_run(experiment, round_, external_configurations=None):

    if external_configurations is None:
        external_configurations = {}

    args = parse_args()
    memo = args.memo

    _, dic_path = runexp.continue_(experiment, round_, args, memo, external_configurations)
    print("****************************** runexp ends (generate, train, test)!! ******************************")

    return experiment

if __name__ == "__main__":
    run()

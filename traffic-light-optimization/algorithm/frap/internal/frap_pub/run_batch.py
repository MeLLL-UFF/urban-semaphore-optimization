import algorithm.frap.internal.frap_pub.runexp as runexp
import algorithm.frap.internal.frap_pub.testexp as testexp
import algorithm.frap.internal.frap_pub.summary as summary
import argparse
import os
import time

from algorithm.frap.internal.frap_pub.definitions import ROOT_DIR

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--memo", type=str, default="TransferDQN")
    parser.add_argument("--algorithm", type=str, default="TransferDQN")
    parser.add_argument("--num_phase", type=int, default=8)
    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--run_round", type=int, default=200)

    parser.add_argument("--done", action="store_true")
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--rotation_input", action="store_true")
    parser.add_argument("--conflict_matrix", action="store_true")

    parser.add_argument("--run_counts", type=int, default=3600)
    parser.add_argument("--test_run_counts", type=int, default=3600)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.98)
    parser.add_argument("--min_lr", type=float, default=0.001)
    parser.add_argument("--update_q_bar_every_c_round", type=bool, default=False)
    parser.add_argument("--early_stop_loss", type=str, default="val_loss")
    parser.add_argument("--dropout_rate", type=float, default=0)

    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sumo_gui", action="store_true")
    parser.add_argument("--min_action_time", type=int, default=10)
    parser.add_argument("--workers", type=int, default=12)


    parser.add_argument("--visible_gpu", type=str, default="")

    return parser.parse_args()


def run(external_configurations={}):
    # python run_batch.py - -num_phase = 8 - -algorithm = TransferDQN - -workers = 12 - -memo = TransferDQN
    args = parse_args()
    # memo = "multi_phase/optimal_search_new/new_headway_anon"
    memo = args.memo
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

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

def continue_(experiment, external_configurations={}):
    # python run_batch.py - -num_phase = 8 - -algorithm = TransferDQN - -workers = 12 - -memo = TransferDQN
    args = parse_args()
    memo = args.memo

    _, dic_path = runexp.continue_(experiment, args, memo, external_configurations)
    print("****************************** runexp ends (generate, train, test)!! ******************************")
    records_dir = dic_path["PATH_TO_WORK_DIRECTORY"]
    summary.single_experiment_summary(memo, records_dir, plots='summary_only')
    print("****************************** summary_detail ends ******************************")
    experiment_name = dic_path["EXECUTION_BASE"]

    return experiment_name

if __name__ == "__main__":
    run()

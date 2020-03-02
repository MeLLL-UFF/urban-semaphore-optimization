import os

import matplotlib.pyplot as plt
import skopt
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from tqdm import tqdm

import sumo_main
import hyperparameter_optimization.skopt_objective as skopt_objective
from hyperparameter_optimization.configuration.skopt_configuration_factory import get_skopt_configuration, \
    skopt_objective_function
from experiment.experiment import Experiment
from experiment.objective import TIME_LOSS
from experiment.strategy import OFF, STATIC, TIME_GAP_ACTUATED, TIME_LOSS_ACTUATED, SOTL_MARCHING, SOTL_PHASE, \
    SOTL_PLATOON, SOTL_REQUEST, SOTL_WAVE, REINFORCEMENT_LEARNING
from city.scenario.scenario_factory import get_scenario
from city.scenario.inga_small.inga_small_scenario import INGA_SMALL
from definitions import ROOT_DIR, OUTPUT_DIR

class TqdmSkopt:
    def __init__(self, **kwargs):
        self._bar = tqdm(**kwargs)

    def __call__(self, res):
        self._bar.update()

n_random_starts = 10
n_calls = 1

HPO_PARAMS = {
    'n_random_starts': n_random_starts if n_random_starts <= n_calls else n_calls,
    'n_calls': n_calls,
    'callback': [TqdmSkopt(total=n_calls)],
    'random_state': 34
}

def run(scenario=None, objective=None, strategy=None, experiment=None):

    # run a single experiment
    if experiment:
        _run(experiment)
        return

    # run entire scenario
    if scenario and objective and strategy:
        for traffic_configuration in scenario:
            _run(Experiment(scenario, traffic_configuration, objective, strategy))
            return

def _run(experiment):

    tls_objective = skopt_objective_function(experiment.scenario, experiment.strategy, experiment.objective)
    TLS_SPACE = get_skopt_configuration(experiment.scenario, experiment.strategy)

    if experiment.strategy == 'OFF':
        sumo_main.main(experiment, 0)
        return

    skopt_objective.FIXED_PARAMS['experiment'] = experiment
    skopt_objective.control_space['execution'] = 0

    tls_results = skopt.gp_minimize(tls_objective, TLS_SPACE, **HPO_PARAMS)

    # clear callback and objective functions out as they are not serializable
    tls_results.specs['args']['callback'] = []
    tls_results.specs['args']['func'] = []

    output_dir_path = ROOT_DIR + '/output'

    skopt.dump(tls_results, output_dir_path + '/' + experiment.path + '/results' + '_' + experiment.name + '.pkl')

    tls_best_metric = tls_results.fun
    tls_best_params = tls_results.x

    print(experiment.name)

    print(tls_best_metric)
    print(tls_best_params)
    print(tls_results.x_iters)

    '''
    try:
        converge_file_path = output_dir_path + '/' + experiment.path + '/convergence' + '_' + experiment.name + '.png'
        if not os.path.isfile(converge_file_path):
            plot_convergence(tls_results)
            plt.savefig(converge_file_path)
            plt.close()

        evaluation_file_path = output_dir_path + '/' + experiment.path + '/evaluations' + '_' + experiment.name + '.png'
        if not os.path.isfile(evaluation_file_path):
            plot_evaluations(tls_results)
            plt.savefig(evaluation_file_path)
            plt.close()

        objective_file_path = output_dir_path + '/' + experiment.path + '/objective' + '_' + experiment.name + '.png'
        if not os.path.isfile(objective_file_path):
            plot_objective(tls_results)
            plt.savefig(objective_file_path)
            plt.close()
    except:
        pass
    '''


def main():

    scenario = get_scenario(INGA_SMALL)

    #run(scenario, TIME_LOSS, OFF)

    #run(scenario, TIME_LOSS, STATIC)

    #run(scenario, TIME_LOSS, TIME_GAP_ACTUATED)
    #run(scenario, TIME_LOSS, TIME_LOSS_ACTUATED)

    #run(scenario, TIME_LOSS, SOTL_MARCHING)
    #run(scenario, TIME_LOSS, SOTL_PHASE)
    #run(scenario, TIME_LOSS, SOTL_PLATOON)
    #run(scenario, TIME_LOSS, SOTL_REQUEST)
    #run(scenario, TIME_LOSS, SOTL_WAVE)

    run(scenario, TIME_LOSS, REINFORCEMENT_LEARNING)


def read_results():

    for experiment_path in tqdm(os.listdir(OUTPUT_DIR)):

        experiment_name = experiment_path.replace('/', '__')

        print(experiment_name)

        results = skopt.load(OUTPUT_DIR + '/' + experiment_path + '/results' + '_' + experiment_name + '.pkl')

        print(results.x)
        print(results.fun)
        try:
            converge_file_path = OUTPUT_DIR + '/' + experiment_path + '/convergence' + '_' + experiment_name + '.png'
            if not os.path.isfile(converge_file_path):
                plot_convergence(results)
                plt.savefig(converge_file_path)
                plt.close()

            evaluation_file_path = OUTPUT_DIR + '/' + experiment_path + '/evaluations' + '_' + experiment_name + '.png'
            if not os.path.isfile(evaluation_file_path):
                plot_evaluations(results)
                plt.savefig(evaluation_file_path)
                plt.close()

            objective_file_path = OUTPUT_DIR + '/' + experiment_path + '/objective' + '_' + experiment_name + '.png'
            if not os.path.isfile(objective_file_path):
                plot_objective(results)
                plt.savefig(objective_file_path)
                plt.close()
        except:
            continue

        print('\n')



if __name__ == "__main__":
    main()
    print('breakpoint')
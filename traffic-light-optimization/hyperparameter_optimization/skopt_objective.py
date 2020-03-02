import os

import sumolib

import sumo_main
from definitions import get_tripinfo_file_path

FIXED_PARAMS = {}

FILL = 'FILL'
REPLACE = 'REPLACE'

control_space = {
    'execution': 0,
    'execution_mode': FILL
}

def time_loss_objective(parameters):

    experiment = FIXED_PARAMS['experiment']

    control_space['execution'] += 1
    execution = control_space['execution']
    execution_mode = control_space['execution_mode']

    if not (execution_mode == FILL and os.path.isfile(get_tripinfo_file_path(experiment, execution))):
        sumo_main.main(experiment, parameters, execution)

    tripinfo = sumolib.output.parse(get_tripinfo_file_path(experiment, execution), ['tripinfo'])

    trip_time_losses = 0
    total_trips = 0
    for trip in tripinfo:
        trip_time_losses += float(trip.timeLoss)
        total_trips += 1

    if total_trips == 0:
        return 0

    average_time_loss = trip_time_losses / total_trips

    return average_time_loss

def depart_delay_objective(parameters):

    experiment = FIXED_PARAMS['experiment']

    control_space['execution'] += 1
    execution = control_space['execution']
    execution_mode = control_space['execution_mode']

    if not (execution_mode == FILL and os.path.isfile(get_tripinfo_file_path(experiment, execution))):
        sumo_main.main(experiment, parameters, execution)

    tripinfo = sumolib.output.parse(get_tripinfo_file_path(experiment, execution), ['tripinfo'])

    trip_depart_delays = 0
    total_trips = 0
    for trip in tripinfo:
        trip_depart_delays += float(trip.departDelay)
        total_trips += 1

    average_depart_delay = trip_depart_delays / total_trips

    return average_depart_delay
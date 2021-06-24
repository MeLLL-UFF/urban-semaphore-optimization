# parameters and paths

from algorithm.frap_pub.frap_agent import FrapAgent
from algorithm.frap_pub.frap_plus_plus_agent import FrapPlusPlusAgent
from algorithm.frap_pub.sumo_agent import SumoAgent
from algorithm.frap_pub.planning_only_agent import PlanningOnlyAgent
from algorithm.frap_pub.frap_with_planning_agent import FrapWithPlanningAgent
from algorithm.frap_pub.sumo_env import SumoEnv
from algorithm.frap_pub.anon_env import AnonEnv

DIC_EXP_CONF = {
    "RUN_COUNTS": 3600,
    "TEST_RUN_COUNTS": 3600,
    "TRAFFIC_FILE": [],  # Added in runexp
    "MODEL_NAME": "Frap",
    "NUM_ROUNDS": 400,
    "NUM_GENERATORS": 3,
    "LIST_MODEL_NEED_TO_UPDATE_BETWEEN_ROUNDS":
        ["Frap", "FrapWithPlanning"],
    "LIST_MODEL_NEED_TO_UPDATE_BETWEEN_STEPS":
        ["FrapPlusPlus"],
    "LIST_MODEL_NEED_TO_UPDATE":
        ["Frap", "FrapPlusPlus", "FrapWithPlanning"],
    "MODEL_POOL": False,
    "NUM_BEST_MODEL": 3,
    "PRETRAIN": True,
    "PRETRAIN_MODEL_NAME": "Random",
    "PRETRAIN_NUM_ROUNDS": 10,
    "PRETRAIN_NUM_GENERATORS": 10,
    "AGGREGATE": False,
    "DEBUG": False,
    "EARLY_STOP": False,
}
DIC_TRAFFIC_ENV_CONF = {
    "SIMULATOR_TYPE": 'sumo',
    "ACTION_PATTERN": "set",
    "IS_RIGHT_ON_RED": True,
    "MAJOR_CONFLICTS_ONLY": True,
    "DEDICATED_MINOR_LINKS_PHASES": True,
    "DETECT_EXISTING_PHASES": False,
    "SIMPLIFY_PHASE_REPRESENTATION": False,
    "PER_SECOND_DECISION": False,
    "MIN_ACTION_TIME": 10,
    "MEASURE_TIME": 10,
    "DETECTOR_EXTENSION": 300,
    "DEFAULT_YELLOW_TIME": 3,
    "ALL_RED_TIME": 0,
    "WAITING_TIME_RESTRICTION": 120,
    "IF_GUI": False,
    "DEBUG": False,
    "BINARY_PHASE_EXPANSION": True,
    "DONE_ENABLE": False,

    "INTERVAL": 1,
    "THREADNUM": 1,
    "SAVEREPLAY": False,
    "RLTRAFFICLIGHT": True,

    "DIC_FEATURE_DIM": dict(  # review
        D_MOVEMENT_QUEUE_LENGTH=(4,),
        #D_LANE_NUM_VEHICLE=(4,),
        D_MOVEMENT_NUMBER_OF_VEHICLES=(1,),
        D_MOVEMENT_NUMBER_OF_VEHICLES_BEEN_STOPPED_THRESHOLD_01=(4,),
        D_MOVEMENT_NUMBER_OF_VEHICLES_BEEN_STOPPED_THRESHOLD_1=(4,),
        D_CURRENT_PHASE=(1,),
        D_NEXT_PHASE=(1,),
        D_TIME_THIS_PHASE=(1,),
        D_TERMINAL=(1,),
        D_MOVEMENT_SUM_WAITING_TIME=(4,),
        D_VEHICLE_POSITION_IMG=(4, 60,),
        D_VEHICLE_SPEED_IMG=(4, 60,),
        D_VEHICLE_WAITING_TIME_IMG=(4, 60,),
        D_MOVEMENT_PRESSURE_PRESSLIGHT=(1,),
        D_MOVEMENT_PRESSURE_MPLIGHT=(1,),
        D_MOVEMENT_PRESSURE_TIME_LOSS=(1,),
        D_MOVEMENT_SUM_TIME_LOSS=(1,)
    ),

    "STATE_FEATURE_LIST": [
        "current_phase",
        "time_this_phase",
        "vehicle_position_img",
        "vehicle_speed_img",
        "vehicle_acceleration_img",
        "vehicle_waiting_time_img",
        "movement_number_of_vehicles",
        "movement_number_of_vehicles_been_stopped_threshold_01",
        "movement_number_of_vehicles_been_stopped_threshold_1",
        "movement_queue_length",
        "movement_number_of_vehicles_left",
        "movement_sum_duration_vehicles_left",
        "movement_sum_waiting_time",
        "terminal",
        "movement_pressure_presslight",
        "movement_pressure_mplight",
        "movement_pressure_time_loss",
        "movement_sum_time_loss"
    ],

    "REWARD_INFO_DICT": {
        "flickering": 0,
        "sum_queue_length": 0,
        "avg_movement_queue_length": 0,
        "sum_waiting_time": 0,
        "sum_number_of_vehicles_left": 0,
        "sum_duration_vehicles_left": 0,
        "sum_number_of_vehicles_been_stopped_threshold_01": 0,
        "sum_number_of_vehicles_been_stopped_threshold_1": 1,
        "pressure_presslight": 0,
        "pressure_mplight": 0,
        "pressure_time_loss": 0,
        "time_loss": 0
    },

    "LOG_DEGUB": False,
}


DIC_FRAP_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "LR_DECAY": 0.98,  # review default=0.98 or 1
    "MIN_LR": 0.001,  # review default=0.001 or 0.0001
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "NORMAL_FACTOR": 20,
    "EARLY_STOP_LOSS": "val_loss",
    "DROPOUT_RATE": 0,
    "MERGE": "multiply"
}

DIC_FRAPPLUSPLUS_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "LR_DECAY": 0.98,  # review default=0.98 or 1
    "MIN_LR": 0.001,  # review default=0.001 or 0.0001
    "SAMPLE_SIZE": 30,
    'UPDATE_START': 100,
    'UPDATE_PERIOD': 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 2000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "NORMAL_FACTOR": 20,
    "EARLY_STOP_LOSS": "val_loss",
    "DROPOUT_RATE": 0,
    "MERGE": "multiply"
}

DIC_SUMO_AGENT_CONF = {
}

DIC_PLANNINGONLY_AGENT_CONF = {
    "PICK_ACTION_AND_KEEP_WITH_IT": False,
    "ACTION_SAMPLING_SIZE": 2,
    "PLANNING_ITERATIONS": 3,
    "TIEBREAK_POLICY": 'random'  # 'random', 'maintain', 'change'
}

DIC_FRAPWITHPLANNING_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "LR_DECAY": 0.98,  # review default=0.98 or 1
    "MIN_LR": 0.001,  # review default=0.001 or 0.0001
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": 20,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "NORMAL_FACTOR": 20,
    "EARLY_STOP_LOSS": "val_loss",
    "DROPOUT_RATE": 0,
    "MERGE": "multiply",

    "PLANNING_ITERATIONS": 2,
    "TIEBREAK_POLICY": 'random',  # 'random', 'maintain', 'change'
    "ACTION_SAMPLING_SIZE": 2,
    "ACTION_SAMPLING_POLICY": 'random',  # 'best', 'random', 'exploration_exploitation'
    "PLANNING_SAMPLE_ONLY": True
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_PRETRAIN_WORK_DIRECTORY": "records/default",
    "PATH_TO_PRETRAIN_DATA": "data/template",
    "PATH_TO_AGGREGATE_SAMPLES": "records/initial",
    "PATH_TO_ERROR": "errors/default"
}

DIC_AGENTS = {
    "Frap": FrapAgent,
    "FrapPlusPlus": FrapPlusPlusAgent,
    "Sumo": SumoAgent,
    "PlanningOnly": PlanningOnlyAgent,
    "FrapWithPlanning": FrapWithPlanningAgent,
}

DIC_ENVS = {
    "sumo": SumoEnv,
    "anon": AnonEnv
}

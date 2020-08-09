from os import getenv, path

import load_envs
load_envs.load()


class Config:

    VERSION = '0.1.0-dev'

    REGIONS_PATH = path.abspath(getenv('REGIONS_PATH'))
    SCENARIO_PATH = path.abspath(getenv('SCENARIO_PATH'))
    OUTPUT_PATH = path.abspath(getenv('OUTPUT_PATH'))

    FRAP_DATA_PATH = path.abspath(getenv('FRAP_DATA_PATH'))
    FRAP_MODEL_PATH = path.abspath(getenv('FRAP_MODEL_PATH'))
    FRAP_RECORDS_PATH = path.abspath(getenv('FRAP_RECORDS_PATH'))
    FRAP_SUMMARY_PATH = path.abspath(getenv('FRAP_SUMMARY_PATH'))

from city.scenario.scenario import Scenario
from utils.bidict import bidict
from definitions import ROOT_DIR


INGA_SMALL = 'inga_small'

ICARAI_CENTRO_DIRECTION = 'icarai-centro'
CENTRO_ICARAI_DIRECTION = 'centro-icarai'


PRAIA_ICARAI__CENTRO_FLOW = 'flow_praia_icarai__centro'
PRAIA_ICARAI__UFF_FLOW = 'flow_praia_icarai__UFF'
ICARAI_MEIO__CENTRO_FLOW = 'flow_icarai_meio__centro'
ICARAI_MEIO__ICARAI_PRAIA_FLOW = 'flow_icarai_meio__icarai_praia'

CENTRO__PRAIA_ICARAI_FLOW = 'flow_centro__praia_icarai'
UFF__PRAIA_ICARAI_FLOW = 'flow_UFF__praia_icarai'
UFF__ICARAI_MEIO_FLOW = 'flow_UFF__icarai_meio'
UFF__UFF__RETORNO_FLOW = 'flow_UFF__UFF__retorno'

SCENARIO_FILES = ROOT_DIR + '/../.regions/' + INGA_SMALL


class IngaSmallScenario(Scenario):

    def __init__(self):

        name = INGA_SMALL
        flows_to_direction = bidict({
            PRAIA_ICARAI__CENTRO_FLOW: ICARAI_CENTRO_DIRECTION,
            PRAIA_ICARAI__UFF_FLOW: ICARAI_CENTRO_DIRECTION,
            ICARAI_MEIO__CENTRO_FLOW: ICARAI_CENTRO_DIRECTION,
            ICARAI_MEIO__ICARAI_PRAIA_FLOW: ICARAI_CENTRO_DIRECTION,

            CENTRO__PRAIA_ICARAI_FLOW: CENTRO_ICARAI_DIRECTION,
            UFF__PRAIA_ICARAI_FLOW: CENTRO_ICARAI_DIRECTION,
            UFF__ICARAI_MEIO_FLOW: CENTRO_ICARAI_DIRECTION,
            UFF__UFF__RETORNO_FLOW: CENTRO_ICARAI_DIRECTION,

        })

        _direction_traffic_parametrization = {
            ICARAI_CENTRO_DIRECTION: {
                'cars_total_range': {
                    'start': 10,
                    'step': 10,
                    'stop': 10
                }
            },
            CENTRO_ICARAI_DIRECTION: {
                'cars_total_range': {
                    'start': 10,
                    'step': 10,
                    'stop': 10
                }
            },
        }

        _traffic_distribution = {
            # ICARAI_CENTRO_DIRECTION
            PRAIA_ICARAI__CENTRO_FLOW: 0.5,
            PRAIA_ICARAI__UFF_FLOW: 0.3,
            ICARAI_MEIO__CENTRO_FLOW: 0.15,
            ICARAI_MEIO__ICARAI_PRAIA_FLOW: 0.05,

            # CENTRO_ICARAI_DIRECTION
            CENTRO__PRAIA_ICARAI_FLOW: 0.5,
            UFF__PRAIA_ICARAI_FLOW: 0.3,
            UFF__ICARAI_MEIO_FLOW: 0.15,
            UFF__UFF__RETORNO_FLOW: 0.05
        }

        super().__init__(name, flows_to_direction, _direction_traffic_parametrization, _traffic_distribution)

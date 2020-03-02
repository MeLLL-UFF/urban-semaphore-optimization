import ast

from city.traffic_light_system.traffic_light.configurer.traffic_light_configurer import TrafficLightConfigurer

TYPE = 'type'

PHASE_TYPE = 'type'

MINIMUM_DURATION = 'minDur'
MAXIMUM_DURATION = 'maxDur'


TRANSIENT_VALUE = 'transient'
TARGET__DECISIONAL_VALUE = 'target;decisional'

TARGET_LANES = 'targetLanes'

class SotlTrafficLightConfigurer(TrafficLightConfigurer):

    def configure_traffic_light(self, tlLogic):
        raise NotImplemented

    def configure_optimizable_phase(self, tlLogic, phase):

        key = tlLogic.get('id') + '.' + phase.get('name') + '.' + MINIMUM_DURATION
        minimum_duration = str(self.traffic_light_parameters.get(key))

        phase.set(MINIMUM_DURATION, minimum_duration)

        key = tlLogic.get('id') + '.' + phase.get('name') + '.' + MAXIMUM_DURATION
        maximum_duration = str(self.traffic_light_parameters.get(key))

        phase.set(MAXIMUM_DURATION, maximum_duration)

        phase_type = TARGET__DECISIONAL_VALUE
        phase.set(PHASE_TYPE, phase_type)

        lanes = self._get_lanes(tlLogic.get('id'), phase.get('name'))

        phase.set(TARGET_LANES, ' '.join(lanes))

    def configure_non_optimizable_phase(self, tlLogic, phase):

        params = tlLogic.findall('.//param')
        optimizable_phases_param = params[0]
        optimizable_phases = ast.literal_eval(optimizable_phases_param.get('value'))

        if len(optimizable_phases) == 1:
            if 'g' in phase.get('state') or 'G' in phase.get('state'):
                phase_type = TARGET__DECISIONAL_VALUE
                phase.set(PHASE_TYPE, phase_type)

                phase_name = tlLogic.get('id')
                lanes = self._get_lanes(tlLogic.get('id'), phase_name)
                phase.set(TARGET_LANES, ' '.join(lanes))
                return

        phase_type = TRANSIENT_VALUE
        phase.set(PHASE_TYPE, phase_type)


    def _get_lanes(self, traffic_light_id, phase_name):

        target_lanes = []

        phase_names = []

        if '__' in phase_name:
            phase_names = phase_name.split('__')
        else:
            phase_names.append(phase_name)


        edges = self.network_definition.findall(".//edge")

        for edge in edges:

            if edge.get('to') and traffic_light_id in edge.get('to'):
                if edge.get('id').split('#')[0] in phase_names:

                    lanes = edge.findall('.//lane')

                    for lane in lanes:
                        target_lanes.append(lane.get('id'))

        return target_lanes

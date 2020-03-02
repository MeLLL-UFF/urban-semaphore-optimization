
import lxml.etree as etree

from city.traffic_light_system.traffic_light.configurer.traffic_light_configurer import TrafficLightConfigurer

TYPE = 'type'
MIN_TIME_LOSS = 'minTimeLoss'
DETECTOR_RANGE = 'detectorRange'

MINIMUM_DURATION = 'minDur'
MAXIMUM_DURATION = 'maxDur'

class TimeLossActuatedTrafficLightConfigurer(TrafficLightConfigurer):

    def configure_traffic_light(self, tlLogic):

        params = []

        param = etree.Element('param')
        param.set('key', MIN_TIME_LOSS)
        param.set('value', str(self.traffic_light_parameters[tlLogic.get('id') + '.' + MIN_TIME_LOSS]))
        params.append(param)

        param = etree.Element('param')
        param.set('key', DETECTOR_RANGE)
        param.set('value', str(self.traffic_light_parameters[tlLogic.get('id') + '.' + DETECTOR_RANGE]))
        params.append(param)

        for param in params:
            tlLogic.append(param)

        tlLogic.set(TYPE, 'delay_based')


    def configure_optimizable_phase(self, tlLogic, phase):

        key = tlLogic.get('id') + '.' + phase.get('name') + '.' + MINIMUM_DURATION
        minimum_duration = str(self.traffic_light_parameters.get(key))

        phase.set(MINIMUM_DURATION, minimum_duration)

        key = tlLogic.get('id') + '.' + phase.get('name') + '.' + MAXIMUM_DURATION
        maximum_duration = str(self.traffic_light_parameters.get(key))

        phase.set(MAXIMUM_DURATION, maximum_duration)


instance = TimeLossActuatedTrafficLightConfigurer()
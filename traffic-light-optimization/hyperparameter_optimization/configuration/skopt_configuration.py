import os
import ast

import skopt
import lxml.etree as etree

from definitions import get_network_file_path


class SkoptConfiguration:

    def build_tls_space(self, scenario):

        TLS_SPACE = []

        TLS_SPACE.extend(self._add_general_parameters())

        net_filename = os.path.abspath(get_network_file_path(scenario))

        parser = etree.XMLParser(remove_blank_text=True)
        net_xml = etree.parse(net_filename, parser)

        tlLogics = net_xml.findall(".//tlLogic")

        for tlLogic in tlLogics:

            traffic_light_id = tlLogic.attrib['id']

            params = tlLogic.findall('.//param')
            optimizable_phases_param = params[0]
            phases = tlLogic.findall('.//phase')

            optimizable_phases = ast.literal_eval(optimizable_phases_param.get('value'))

            TLS_SPACE.extend(self._add_traffic_lights_parameters(traffic_light_id))

            for i, phase in enumerate(phases):

                if i in optimizable_phases:
                    phase_name = phase.attrib['name']

                    TLS_SPACE.extend(self._add_optimizable_phases_parameters(traffic_light_id, phase_name))

        return TLS_SPACE

    def _add_general_parameters(self):
        return []

    def _add_traffic_lights_parameters(self, traffic_light_id):
        return []

    def _add_optimizable_phases_parameters(self, traffic_light_id, phase_name):
        return []
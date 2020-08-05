import os

from algorithm.frap.internal.frap_pub import run_batch
from algorithm.frap.internal.frap_pub import frap_definitions

class Frap:

    def run(self, net_file, route_file, output_file):
        input_data = os.path.join(frap_definitions.ROOT_DIR, 'data', 'template_ls')

        run_batch.run()
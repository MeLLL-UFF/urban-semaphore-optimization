import os

from algorithm.frap.internal.frap_pub import run_batch
from algorithm.frap.internal.frap_pub import definitions

class Frap:

    def run(self, net_file, route_file, output_file):
        input_data = os.path.join(definitions.ROOT_DIR, 'data', 'template_ls')

        run_batch.run()
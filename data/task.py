import os
import sys
import shutil
from apache_beam.options.pipeline_options import PipelineOptions
from data.process import TFTJob

if __name__ == '__main__':
    if os.path.exists("model"):
        shutil.rmtree('model')

    options = PipelineOptions(flag=sys.argv)

    tft_job = TFTJob()
    tft_job.run(options)
import os
import shutil

from data.process import TFTJob

if __name__ == '__main__':
    if os.path.exists("model"):
        shutil.rmtree('model')
    tft_job = TFTJob()
    tft_job.run()
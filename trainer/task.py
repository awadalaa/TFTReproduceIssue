import os
import shutil
from trainer.model import CensusModel

if __name__ == '__main__':
    census_model_wrapper = CensusModel()

    if os.path.exists(census_model_wrapper.model_dir):
        shutil.rmtree(census_model_wrapper.model_dir)

    census_model_wrapper.train_and_evaluate()
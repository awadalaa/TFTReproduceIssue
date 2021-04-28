from trainer.model import CensusModel

if __name__ == '__main__':
    census_model_wrapper = CensusModel()
    census_model_wrapper.train_and_evaluate()
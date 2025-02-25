import json
from src.utils.utils import get_abs_class_file_path
import os

# at the moment this seems to be unused, idea use this in place of the whole set of command line arguments for the trainign procedure
# this is NOT the same as the EncodersExecutorConfig though

class ModelTrainConfig():
    """abstraction to hold the configuration of a model for the training,
    as for now it is only a mere dictionary holder, ideally it would become a bit more structured
    """

    CONFIG_PATH = os.path.join('data', 'training_configs')

    def __init__(self, config_data):
        self.config_data = config_data

    def __get_abs_config_path(self):
        return os.path.join(get_abs_class_file_path(self), __class__.CONFIG_PATH)


    def load(cls, fn):
        with open(os.path.join(fn)) as f:
            data = json.load(f)
        return ModelTrainConfig(data)

    def store(self, fn):
        with open(os.path.join(self.__get_abs_config_path(), fn), 'x') as f:
            json.dump(self.config_data, f)

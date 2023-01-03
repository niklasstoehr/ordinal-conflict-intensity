import os
import typing as th
from pathlib import Path
import toml

import logging
LOGGER = logging.getLogger("__main__")

CONFIG_FILE = 'g0configs/configs.toml'
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


TRAIN_ARGS = {
}


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



class ConfigBase():


    def __init__(self):
        super().__init__()

        self.paths = {}
        self.train_args = AttrDict(TRAIN_ARGS)
        self.attach_config(self.load_configs(CONFIG_FILE))

    def attach_config(self, configs: th.Dict):

        for key, value in configs.items():
            self.__setattr__(key, value)

    def load_configs(self, config_files: str):

        CONFIG_PATH = Path(ROOT_DIR) / config_files
        CONFIG_DICT = toml.load(CONFIG_PATH, _dict=dict)

        return CONFIG_DICT


    def get_path(self, pathkey: str):

        resource_path = Path(DATA_ROOT_DIR) / self.paths[pathkey]
        return resource_path


import os
from configparser import ConfigParser

from misc import LOGGER

CONFIG_SAMPLE_URL = 'https://github.com/skela5528/players_count/blob/master/config.ini'


def get_config(config_path: str) -> ConfigParser:
    if not os.path.exists(config_path):
        full_config_path = os.path.abspath(config_path)
        LOGGER.error(f'[config] Config not found at {full_config_path}! Please download config from {CONFIG_SAMPLE_URL}.')
        exit()
    config = ConfigParser()
    config.read(config_path)
    return config


def parse_config_section(config_section) -> dict:
    params_dict = {}
    for param, value in config_section.items():
        if value.isnumeric():
            value = int(value)
        params_dict[param] = value
    return params_dict

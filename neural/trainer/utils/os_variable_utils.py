import os

from utils.consts import UNET_6_LAYERS, UNET_DENSE_4_LAYERS


def get_profile_name():
    return os.environ.get('PROFILE') if os.environ.get('PROFILE') else 'local'


def get_model_name():
    return os.environ.get('MODEL_NAME') if os.environ.get('MODEL_NAME') else UNET_DENSE_4_LAYERS

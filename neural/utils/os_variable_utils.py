import os


def get_profile_name():
    return os.environ.get('PROFILE') if os.environ.get('PROFILE') else 'local'


def get_model_name():
    return os.environ.get('MODEL_NAME') if os.environ.get('MODEL_NAME') else 'unet_4_layers'
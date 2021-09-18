from models import unet_4_layers, dense_unet_4_layers, unet_6_layers, plus_plus_unet_4_layers

from utils.consts import UNET_4_LAYERS, UNET_6_LAYERS, UNET_PLUS_PLUS_4_LAYERS, UNET_DENSE_4_LAYERS


def get_model_builder(model_name):
    if model_name == UNET_4_LAYERS:
        return unet_4_layers.build_model
    elif model_name == UNET_6_LAYERS:
        return unet_6_layers.build_model
    elif model_name == UNET_DENSE_4_LAYERS:
        return dense_unet_4_layers.build_model
    elif model_name == UNET_PLUS_PLUS_4_LAYERS:
        return plus_plus_unet_4_layers.build_model_plus



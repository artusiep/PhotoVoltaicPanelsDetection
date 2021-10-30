from models import unet_4_layers, dense_unet_4_layers, unet_6_layers, plus_plus_unet_4_layers, res_net_152, res_net_34, \
    linknet, vgg19, fpn

from trainer.utils.consts import UNET_4_LAYERS, UNET_6_LAYERS, UNET_PLUS_PLUS_4_LAYERS, UNET_DENSE_4_LAYERS, \
    RES_NET_152, VGG19, RES_NET_34, LINKNET, FPN


def get_model_builder(model_name):
    if model_name == UNET_4_LAYERS:
        return unet_4_layers.build_model
    elif model_name == UNET_6_LAYERS:
        return unet_6_layers.build_model
    elif model_name == UNET_DENSE_4_LAYERS:
        return dense_unet_4_layers.build_model
    elif model_name == UNET_PLUS_PLUS_4_LAYERS:
        return plus_plus_unet_4_layers.build_model_plus
    elif model_name == RES_NET_152:
        return res_net_152.build_model
    elif model_name == RES_NET_34:
        return vgg19.build_model
    elif model_name == LINKNET:
        return linknet.build_model
    elif model_name == FPN:
        return fpn.build_model

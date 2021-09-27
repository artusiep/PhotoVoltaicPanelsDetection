from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models_builders import get_model_builder


def build(model_name, img_size, channel_numbers, starts_neuron):
    return get_model_builder(model_name)(img_size, img_size, channel_numbers, starts_neuron)

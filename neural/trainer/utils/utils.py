import os
from datetime import datetime
from typing import Tuple

from trainer.utils.os_variable_utils import get_model_name
from trainer.utils.paths_definition import get_model_save_path


def get_save_model_path(run_id, model_name, grayscale):
    return get_model_save_path().format(run_id=run_id, model=model_name,
                                        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                                        f1_score='', is_final='', grayscale='gray' if grayscale else 'rgb')


def get_final_save_model_path(run_id, model_name, f1_score, grayscale):
    return get_model_save_path().format(run_id=run_id, model=model_name,
                                        timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                                        f1_score=f'_{round(f1_score, 5)}', is_final='final_',
                                        grayscale='gray' if grayscale else 'rgb')


def load_variables() -> Tuple[str, int, int, int, int, int, str]:
    # profile = get_profile_name()
    model_name = get_model_name()
    channel_numbers = 1
    img_size = int(os.environ.get('IMG_SIZE')) if os.environ.get('IMG_SIZE') else 128
    epochs = int(os.environ.get('EPOCHS')) if os.environ.get('EPOCHS') else 100
    batch_size = int(os.environ.get('BATCH_SIZE')) if os.environ.get('BATCH_SIZE') else 32
    starts_neuron = int(os.environ.get('STARTS_NEURON')) if os.environ.get('STARTS_NEURON') else 16
    trained_model_weights_path = os.environ.get('TRAINED_MODEL_WEIGHTS_PATH') if os.environ.get(
        'TRAINED_MODEL_WEIGHTS_PATH') else ''

    print(
        'Executing model with parameters: \n'
        f'MODEL_NAME = {model_name}\n',
        f'IMG_SIZE = {img_size}\n',
        f'EPOCHS = {epochs}\n',
        f'BATCH_SIZE = {batch_size}\n',
        f'STARTS_NEURON = {starts_neuron}\n',
        f'TRAINED_MODEL_WEIGHTS_PATH = {trained_model_weights_path}\n',
    )

    return model_name, channel_numbers, img_size, epochs, batch_size, starts_neuron, trained_model_weights_path

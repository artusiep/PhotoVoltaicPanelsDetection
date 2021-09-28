import os
from datetime import datetime

from trainer.utils.os_variable_utils import get_model_name
from trainer.utils.paths_definition import get_model_save_path


def get_save_model_path(run_id, model_name, grayscale):
    return get_model_save_path().format(run_id=run_id, model=model_name, timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                                        f1_score='', is_final='', grayscale='gray' if grayscale else 'rgb')


def get_final_save_model_path(run_id, model_name, f1_score, grayscale):
    return get_model_save_path().format(run_id=run_id, model=model_name, timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                                        f1_score=f'_{round(f1_score,5)}', is_final='final_',
                                        grayscale='gray' if grayscale else 'rgb')


def load_variables():
    # profile = get_profile_name()
    model_name = get_model_name()
    channel_numbers = 1
    img_size = int(os.environ.get('IMG_SIZE')) if os.environ.get('IMG_SIZE') else 128
    epochs = int(os.environ.get('EPOCHS')) if os.environ.get('EPOCHS') else 100
    batch_size = int(os.environ.get('BATCH_SIZE')) if os.environ.get('BATCH_SIZE') else 32
    starts_neuron = int(os.environ.get('STARTS_NEURON')) if os.environ.get('STARTS_NEURON') else 16

    # start_case_index_train = int(os.environ.get('START_CASE_INDEX_TRAIN')) if os.environ.get('START_CASE_INDEX_TRAIN') else 0
    # end_case_index_train = int(os.environ.get('END_CASE_INDEX_TRAIN')) if os.environ.get('END_CASE_INDEX_TRAIN') else 180
    #
    # start_case_index_test = int(os.environ.get('START_CASE_INDEX_TEST')) if os.environ.get('START_CASE_INDEX_TEST') else 181
    # end_case_index_test = int(os.environ.get('END_CASE_INDEX_TEST')) if os.environ.get('END_CASE_INDEX_TEST') else 209

    trained_model_weights_path = os.environ.get('TRAINED_MODEL_WEIGHTS_PATH') if os.environ.get(
        'TRAINED_MODEL_WEIGHTS_PATH') else ''
    # threshold = float(os.environ.get('THRESHOLD')) if os.environ.get('THRESHOLD') else 0.6

    print(
        'Executing model with parameters: \n'
        # 'PROFILE = %s\n' % profile,
        'MODEL_NAME = %s\n' % model_name,
        'IMG_SIZE = %s\n' % img_size,
        'EPOCHS = %s\n' % epochs,
        'BATCH_SIZE = %s\n' % batch_size,
        'STARTS_NEURON = %s\n' % starts_neuron,
        # 'START_CASE_INDEX_TRAIN = %s\n' % start_case_index_train,
        # 'END_CASE_INDEX_TRAIN = %s\n' % end_case_index_train,
        # 'START_CASE_INDEX_TEST = %s\n' % start_case_index_test,
        # 'END_CASE_INDEX_TEST = %s\n' % end_case_index_test,
        'TRAINED_MODEL_WEIGHTS_PATH = %s\n' % trained_model_weights_path,
        # 'THRESHOLD = %s\n' % threshold
    )

    # return (model_name, channel_numbers, img_size, epochs, batch_size, starts_neuron, start_case_index_train,
    #         end_case_index_train,
    #         start_case_index_test, end_case_index_test,
    #         trained_model_weights_path, threshold)
    return (
        model_name,
        channel_numbers,
        img_size,
        epochs,
        batch_size,
        starts_neuron,
        # start_case_index_train,
        # end_case_index_train,
        # start_case_index_test,
        # end_case_index_test,
        trained_model_weights_path
    )

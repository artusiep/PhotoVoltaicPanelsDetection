import tensorflow as tf

from utils.paths_definition import get_logs_dir


def get_callbacks(model_save_path):
    checkpoint_path = model_save_path

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     validation_split=0.1,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)

    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        tf.keras.callbacks.TensorBoard(log_dir=get_logs_dir()),
        cp_callback
    ]

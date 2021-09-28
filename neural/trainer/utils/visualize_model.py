import tensorflow as tf

def to_file(filepath, model):
    tf.keras.utils.plot_model(
        model, to_file=filepath, show_shapes=False, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )
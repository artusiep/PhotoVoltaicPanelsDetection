import sys

import numpy as np
from sklearn.metrics import f1_score

from models_builders import get_model_builder
from utils.read_data import get_images_and_masks
from utils.utils import load_variables


def test_model(model_save_dir_path):
    print('Started')
    (model_name, channel_numbers, img_size, epochs, batch_size, starts_neuron, start_case_index_train,
     end_case_index_train, start_case_index_test, end_case_index_test) = load_variables()
    print('Variables loaded')

    model = get_model_builder(model_name)(img_size, img_size, channel_numbers, starts_neuron)
    print("Model built")

    model.summary()
    print("Model summary")

    test_images, test_labels = get_images_and_masks(img_size, img_size, start_case_index_test, end_case_index_test, True)
    print("Test data loaded")

    loss, acc = model.evaluate(test_images, test_labels, verbose=1)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    model.load_weights(model_save_dir_path)
    print("Model weights loaded")

    print("Prediction started")
    preds_test = model.predict(test_images, verbose=1)
    preds_test_t = (preds_test > 0.6).astype(np.uint8)

    f1_score_result = f1_score(test_labels.flatten().flatten(), preds_test_t.flatten().flatten())
    print('F1 score: %f' % f1_score_result)


if __name__ == "__main__":
    test_model(str(sys.argv[1]))

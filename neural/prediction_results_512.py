import cv2
import numpy as np
from sklearn.metrics import f1_score

from models_builders import get_model_builder
from utils.read_data import get_images_and_masks
from utils.utils import load_variables

print("[LOG] Starting application")

print('[LOG] Loading variables')
(
    model_name,
    channel_numbers,
    img_size,
    epochs,
    batch_size,
    starts_neuron,
    start_case_index_train,
    end_case_index_train,
    start_case_index_test,
    end_case_index_test,
    trained_model_weights_path,
    threshold
) = load_variables()
print('[LOG] Did load variables')

print('[LOG] Building model')
model = get_model_builder(model_name)(
    img_size,
    img_size,
    channel_numbers,
    starts_neuron
)

print('[LOG] Load trained model weights')
model.load_weights(trained_model_weights_path)

print('[LOG] Loading test data for cases from: ', start_case_index_test, " to: ", end_case_index_test)
test_images, test_labels = get_images_and_masks(
    img_size,
    img_size,
    start_case_index_test,
    end_case_index_test,
    True
)
test_images_512, test_labels_512 = get_images_and_masks(
    512,
    512,
    start_case_index_test,
    end_case_index_test,
    True
)
print("[LOG] Did load test data")

print("[LOG] Start model evaluation with test data")
loss, acc = model.evaluate(
    test_images,
    test_labels,
    verbose=1
)
print("[LOG] Did evaluate model")
print("[LOG] Current model accuracy: {:5.2f}%".format(100 * acc))

print("[LOG] Start model predictions with test data")
pred_test = model.predict(test_images, verbose=1)
pred_test_t = (pred_test > threshold).astype(np.uint8)
print("[LOG] Did finish predictions")

pred_test_t_512 = np.empty([len(pred_test_t), 512, 512])
print("[LOG] Resize predictions to 512x512")
for pred_index in range(0, len(pred_test_t)):
    pred_test_t_512[pred_index] = cv2.resize(pred_test_t[pred_index], (512, 512))

print("[LOG] Start F1 calculation with predicted data")
f1_result = f1_score(test_labels_512.flatten().flatten(), pred_test_t_512.flatten().flatten())
print("[LOG] Did calculate F1 score")
print('[LOG] F1 score for case: %f' % f1_result)

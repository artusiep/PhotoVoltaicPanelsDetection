import numpy as np
import pandas as pd
from numpy import arange
from sklearn.metrics import f1_score, precision_score, recall_score

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
    trained_model_weights_path
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
print("[LOG] Did load test data")

print("[LOG] Start model evaluation with test data")
loss, acc = model.evaluate(
    test_images,
    test_labels,
    verbose=1
)
print("[LOG] Did evaluate model")
print("[LOG] Current model accuracy: {:5.2f}%".format(100 * acc))

print("[LOG] Calculate f1, precision, recall - for thresholds")
thresholds = arange(0.0, 1.0, 0.04)
result = np.zeros((len(thresholds), 4))

test_labels.flatten().flatten()

index = 0
pred_test = model.predict(test_images, verbose=1)
for threshold in thresholds:
    pred_test_t = (pred_test > threshold).astype(np.uint8)

    f1_result = f1_score(test_labels.flatten().flatten(), pred_test_t.flatten().flatten())
    precision_result = precision_score(test_labels.flatten().flatten(), pred_test_t.flatten().flatten())
    recall_result = recall_score(test_labels.flatten().flatten(), pred_test_t.flatten().flatten())
    print("[LOG] Threshold: {:5.4f}, f1 score: {:5.4f}, precision score: {:5.4f}, recall score: {:5.4f}".format(threshold, f1_result, precision_result, recall_result))
    result = np.insert(result, index, (threshold, f1_result, precision_result, recall_result), axis=0)
    index = index + 1

data_frame = pd.DataFrame(result)
data_frame.to_csv("{}_thresholds_and_scores.csv".format(model_name))





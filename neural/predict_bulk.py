import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

from models.models_builders import get_model_builder
from trainer.utils.consts import UNET_4_LAYERS, UNET_6_LAYERS, UNET_DENSE_4_LAYERS, UNET_PLUS_PLUS_4_LAYERS
from trainer.utils.read_data import get_images_and_masks
from trainer.utils.visualize_model import to_file

gray_models = {
    UNET_4_LAYERS: [
        'trainer/data/training_result/1_training_unet_4_layers_2021-09-27T23:56:34_gray/cp.ckpt',
        'trainer/data/training_result/1_training_unet_4_layers_2021-09-27T23:58:51_gray/cp.ckpt',
        'trainer/data/training_result/1_training_unet_4_layers_2021-09-28T00:07:42_gray/cp.ckpt',
        'trainer/data/training_result/2_training_unet_4_layers_2021-09-28T01:38:35_gray/cp.ckpt',
        'trainer/data/training_result/3_training_unet_4_layers_2021-09-28T02:46:21_gray/cp.ckpt',
        'trainer/data/training_result/4_training_unet_4_layers_2021-09-28T04:14:58_gray/cp.ckpt',
        'trainer/data/training_result/5_training_unet_4_layers_2021-09-28T05:44:58_gray/cp.ckpt',
    ],
    UNET_6_LAYERS: [
        'trainer/data/training_result/1_training_unet_6_layers_2021-09-28T00:00:11_gray/cp.ckpt',
        'trainer/data/training_result/1_training_unet_6_layers_2021-09-28T00:23:27_gray/cp.ckpt',
        'trainer/data/training_result/2_training_unet_6_layers_2021-09-28T01:50:47_gray/cp.ckpt',
        'trainer/data/training_result/3_training_unet_6_layers_2021-09-28T03:05:06_gray/cp.ckpt',
        'trainer/data/training_result/4_training_unet_6_layers_2021-09-28T04:24:26_gray/cp.ckpt',
        'trainer/data/training_result/5_training_unet_6_layers_2021-09-28T05:54:05_gray/cp.ckpt',
    ],
    UNET_DENSE_4_LAYERS: [
        'trainer/data/training_result/1_training_unet_dense_4_layers_2021-09-28T01:01:23_gray/cp.ckpt',
        'trainer/data/training_result/2_training_unet_dense_4_layers_2021-09-28T02:07:30_gray/cp.ckpt',
        'trainer/data/training_result/3_training_unet_dense_4_layers_2021-09-28T03:36:08_gray/cp.ckpt',
        'trainer/data/training_result/4_training_unet_dense_4_layers_2021-09-28T04:37:15_gray/cp.ckpt',
        'trainer/data/training_result/5_training_unet_dense_4_layers_2021-09-28T06:26:33_gray/cp.ckpt',

    ],
    UNET_PLUS_PLUS_4_LAYERS: [
        'trainer/data/training_result/1_training_unet_plus_plus_4_layers_2021-09-28T01:14:54_gray/cp.ckpt',
        'trainer/data/training_result/2_training_unet_plus_plus_4_layers_2021-09-28T02:26:39_gray/cp.ckpt',
        'trainer/data/training_result/3_training_unet_plus_plus_4_layers_2021-09-28T04:02:37_gray/cp.ckpt',
        'trainer/data/training_result/4_training_unet_plus_plus_4_layers_2021-09-28T05:22:23_gray/cp.ckpt',
        'trainer/data/training_result/5_training_unet_plus_plus_4_layers_2021-09-28T06:45:33_gray/cp.ckpt',
    ]
}

rgb_models = {
    UNET_4_LAYERS: [
        'trainer/data/training_result/1_training_unet_4_layers_2021-09-28T08:50:51_rgb/cp.ckpt',
        'trainer/data/training_result/2_training_unet_4_layers_2021-09-28T10:25:41_rgb/cp.ckpt',
        'trainer/data/training_result/3_training_unet_4_layers_2021-09-28T12:46:25_rgb/cp.ckpt',
        'trainer/data/training_result/4_training_unet_4_layers_2021-09-28T14:32:02_rgb/cp.ckpt',
        'trainer/data/training_result/5_training_unet_4_layers_2021-09-28T16:11:21_rgb/cp.ckpt',
    ],
    UNET_6_LAYERS: [
        'trainer/data/training_result/1_training_unet_6_layers_2021-09-28T09:07:46_rgb/cp.ckpt',
        'trainer/data/training_result/2_training_unet_6_layers_2021-09-28T10:44:59_rgb/cp.ckpt',
        'trainer/data/training_result/3_training_unet_6_layers_2021-09-28T13:10:27_rgb/cp.ckpt',
        'trainer/data/training_result/4_training_unet_6_layers_2021-09-28T14:43:06_rgb/cp.ckpt',
        'trainer/data/training_result/5_training_unet_6_layers_2021-09-28T16:24:14_rgb/cp.ckpt',
    ],
    UNET_DENSE_4_LAYERS: [
        'trainer/data/training_result/1_training_unet_dense_4_layers_2021-09-28T09:27:13_rgb/cp.ckpt',
        'trainer/data/training_result/2_training_unet_dense_4_layers_2021-09-28T11:23:08_rgb/cp.ckpt',
        'trainer/data/training_result/3_training_unet_dense_4_layers_2021-09-28T13:45:38_rgb/cp.ckpt',
        'trainer/data/training_result/4_training_unet_dense_4_layers_2021-09-28T15:06:02_rgb/cp.ckpt',
        'trainer/data/training_result/5_training_unet_dense_4_layers_2021-09-28T16:43:34_rgb/cp.ckpt',
    ],
    UNET_PLUS_PLUS_4_LAYERS: [
        'trainer/data/training_result/1_training_unet_plus_plus_4_layers_2021-09-28T09:53:06_rgb/cp.ckpt',
        'trainer/data/training_result/2_training_unet_plus_plus_4_layers_2021-09-28T12:06:02_rgb/cp.ckpt',
        'trainer/data/training_result/3_training_unet_plus_plus_4_layers_2021-09-28T14:06:28_rgb/cp.ckpt',
        'trainer/data/training_result/4_training_unet_plus_plus_4_layers_2021-09-28T15:58:00_rgb/cp.ckpt',
        'trainer/data/training_result/5_training_unet_plus_plus_4_layers_2021-09-28T16:57:17_rgb/cp.ckpt',
    ]
}

img_size = 128
starts_neuron = 16


def predict(model, weights_path, test_images, test_labels, output_path=None):
    model.load_weights(weights_path)

    print('[LOG] Pre loading weights')

    pred_test = model.predict(test_images, verbose=0, use_multiprocessing=True)

    pred_test = (pred_test > 0.6).astype(np.uint8)
    f1score = f1_score(test_labels.flatten().flatten(), pred_test.flatten().flatten())
    if output_path is not None:
        for id, pred_mask in enumerate(zip(pred_test, test_images)):
            Image.fromarray(np.squeeze((pred_mask[0] * 255).astype(np.uint8), axis=2)).save(
                f'data/{output_path}/{id}-result.png')
            Image.fromarray(np.squeeze((pred_mask[1]).astype(np.uint8), axis=2)).save(
                f'data/{output_path}/{id}-source.png')
    return (f1score, weights_path)


def start(models, grayscale, test_images, test_labels):
    result_file_name = 'results.txt'
    for model_name, weights_paths in models.items():
        model = get_model_builder(model_name)(img_size, img_size, 1 if grayscale else 3, starts_neuron)
        to_file("test", model)
        for weights_path in weights_paths:
            try:
                result = (predict(model, weights_path, test_images, test_labels), model_name, grayscale)
                with open(result_file_name, 'a') as f:
                    f.write(str(result) + '\n')
            except Exception as e:
                print(f"Failed to predict of {model_name} and {weights_path} due to {e}")


if __name__ == '__main__':
    # _, _, test_img, test_lab = get_images_and_masks(img_size, img_size, True, grayscale=True)
    test_img, test_lab = None, None
    start(gray_models, True, test_img, test_lab)
    _, _, test_img, test_lab = get_images_and_masks(img_size, img_size, True, grayscale=False)
    start(rgb_models, False, test_img, test_lab)


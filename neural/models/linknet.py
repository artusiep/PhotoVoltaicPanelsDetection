def build_model(img_width, img_height, channels_number, start_neurons=16):
    import segmentation_models as sm
    model = sm.Linknet('resnet34', input_shape=(img_height, img_width, channels_number), classes=1, encoder_weights=None,
                    activation='sigmoid')
    model.summary()
    model.compile(optimizer='adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

    print("[LOG] Linknet model with resnet34 backbone  built.")
    return model

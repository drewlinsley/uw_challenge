import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'BSDS500',
    ]
    exp['val_dataset'] = [
        'BSDS500',
    ]
    exp['model'] = [
        # 'hgru_bn_bsds',
        'fgru_bsds'
    ]

    exp['validation_period'] = [2000]
    exp['validation_steps'] = [50]
    exp['shuffle_val'] = [True]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [1e-3]
    exp['loss_function'] = ['bce']
    exp['score_function'] = ['accuracy']
    exp['optimizer'] = ['nadam', 'adam']
    exp['train_batch_size'] = [2]
    exp['val_batch_size'] = [2]
    exp['epochs'] = [2000]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'random_crop_image_label',
        'left_right',
        'uint8_rescale',
        'zero_one'
    ]]
    exp['val_augmentations'] = [[
        'center_crop_image_label',
        'uint8_rescale',
        'zero_one'
    ]]
    return exp

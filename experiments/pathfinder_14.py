import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 5  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'curv_contour_length_14_full',
    ]
    exp['val_dataset'] = [
        'curv_contour_length_14_full',
    ]
    exp['model'] = [
        # 'resnet_18',
        # 'resnet_50',
        # 'resnet_152',
        # 'unet',
        # 'seung_unet',
        # 'bu_fgru',
        # 'htd_fgru',
        # 'h_fgru',
        # 'td_fgru',
        # 'htd_fgru_t1',
        # 'h_fgru_t1',
        # 'td_fgru_t1',
        'htd_fgru_linear'
    ]

    exp['validation_period'] = [2000]
    exp['validation_steps'] = [625]
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
    exp['optimizer'] = ['nadam']
    exp['train_batch_size'] = [32]
    exp['val_batch_size'] = [32]
    exp['epochs'] = [2]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'grayscale',
        # 'left_right',
        # 'up_down',
        'uint8_rescale',
        'singleton',
        'resize',
        'zero_one'
    ]]
    exp['val_augmentations'] = [[
        'grayscale',
        # 'left_right',
        # 'up_down',
        'uint8_rescale',
        'singleton',
        'resize',
        'zero_one'
    ]]
    return exp

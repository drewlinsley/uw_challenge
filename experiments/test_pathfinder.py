import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 50  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'curv_contour_length_14_full',
    ]
    exp['val_dataset'] = [
        'curv_contour_length_14_full',
    ]
    exp['model'] = [
        'h_td_fgru',
        # 'hgru_bn'
    ]

    exp['validation_period'] = [10]
    exp['validation_steps'] = [15]
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
    exp['train_batch_size'] = [8]
    exp['val_batch_size'] = [8]
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

import os


def experiment_params():
    """Specifification of experiment params and model hps."""
    exp = {}
    exp['repeat'] = 1  # Repeat each derived experiment this many times

    # Experiment params. All below this line need to be in lists.
    exp['experiment'] = [__file__.split(os.path.sep)[-1].split('.')[0]]
    exp['train_dataset'] = [
        'uw_challenge',
    ]
    exp['val_dataset'] = [
        'uw_challenge',
    ]
    exp['model'] = [
        'hgru_bn',
    ]

    exp['validation_period'] = [50]
    exp['validation_steps'] = [1]
    exp['shuffle_val'] = [True]  # Shuffle val data.
    exp['shuffle_train'] = [True]
    exp['save_checkpoints'] = [1]
    exp['save_activities'] = [False]
    exp['save_weights'] = [False]
    exp['save_gradients'] = [False]

    # Model hyperparameters
    exp['lr'] = [1e-2]
    # exp['freeze_lr'] = [1e-5]

    exp['loss_function'] = [['mse_nn', 'pearson']]
    exp['loss_weights'] = [[1, 0.5]]
    # exp['loss_function'] = ['mse_nn']
    # exp['loss_weights'] = [1]  # [[1, 0.2]]
    exp['val_loss_function'] = ['mse_nn']
    exp['score_function'] = ['mse_nn']
    exp['optimizer'] = ['nadam']
    exp['train_batch_size'] = [50]
    exp['val_batch_size'] = [50]
    exp['epochs'] = [2000]

    # Augmentations specified in lists of lists
    exp['train_augmentations'] = [[
        'pad',
        # 'left_right',
    ]]
    exp['val_augmentations'] = [[
        'pad'
    ]]
    return exp


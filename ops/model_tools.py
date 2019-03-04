# import numpy as np
import tensorflow as tf
from utils import py_utils
from ops import training
# from ops import metrics
# from ops import data_structure
from ops import data_loader
from ops import optimizers
from ops import losses
from ops import gradients
from ops import tf_fun


def initialize_tf(config, directories, placeholders):
    """Initialize tensorflow model variables."""
    # Initialize tf variables
    saver = tf.train.Saver(
        var_list=tf.global_variables(),
        max_to_keep=config.save_checkpoints)
    summary_op = tf.summary.merge_all()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()))
    summary_writer = tf.summary.FileWriter(
        directories['summaries'],
        sess.graph)
    if not placeholders:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    else:
        coord, threads = None, None
    return sess, saver, summary_op, summary_writer, coord, threads


def get_placeholders(train_dataset, val_dataset, config):
    """Create placeholders and apply augmentations."""
    train_images = tf.placeholder(
        dtype=train_dataset.tf_reader['image']['dtype'],
        shape=[config.train_batch_size] + train_dataset.im_size,
        name='train_images')
    train_labels = tf.placeholder(
        dtype=train_dataset.tf_reader['label']['dtype'],
        shape=[config.train_batch_size] + train_dataset.label_size,
        name='train_labels')
    val_images = tf.placeholder(
        dtype=val_dataset.tf_reader['image']['dtype'],
        shape=[config.val_batch_size] + val_dataset.im_size,
        name='val_images')
    val_labels = tf.placeholder(
        dtype=val_dataset.tf_reader['label']['dtype'],
        shape=[config.val_batch_size] + val_dataset.label_size,
        name='val_labels')
    aug_train_ims, aug_train_labels = [], []
    aug_val_ims, aug_val_labels = [], []
    split_train_ims = tf.split(train_images, config.train_batch_size, axis=0)
    split_train_labels = tf.split(train_labels, config.train_batch_size, axis=0)
    split_val_ims = tf.split(val_images, config.val_batch_size, axis=0)
    split_val_labels = tf.split(val_labels, config.val_batch_size, axis=0)
    for tr_im, tr_la, va_im, va_la in zip(
            split_train_ims,
            split_train_labels,
            split_val_ims,
            split_val_labels):
        tr_im, tr_la = data_loader.image_augmentations(
            image=tf.squeeze(tr_im),
            label=tf.squeeze(tr_la),
            model_input_image_size=train_dataset.model_input_image_size,
            data_augmentations=config.train_augmentations)
        va_im, va_la = data_loader.image_augmentations(
            image=tf.squeeze(va_im),
            label=tf.squeeze(va_la),
            model_input_image_size=val_dataset.model_input_image_size,
            data_augmentations=config.val_augmentations)
        aug_train_ims += [tr_im]
        aug_train_labels += [tr_la]
        aug_val_ims += [va_im]
        aug_val_labels += [va_la]
    aug_train_ims = tf.stack(aug_train_ims, axis=0)
    aug_train_labels = tf.stack(aug_train_labels, axis=0)
    aug_val_ims = tf.stack(aug_val_ims, axis=0)
    aug_val_labels = tf.stack(aug_val_labels, axis=0)
    # return aug_train_ims, aug_train_labels, aug_val_ims, aug_val_labels
    return train_images, train_labels, val_images, val_labels, aug_train_ims, aug_train_labels, aug_val_ims, aug_val_labels


def get_placeholders_test(test_dataset, config):
    """Create test placeholders and apply augmentations."""
    test_images = tf.placeholder(
        dtype=test_dataset.tf_reader['image']['dtype'],
        shape=[config.test_batch_size] + test_dataset.im_size,
        name='test_images')
    test_labels = tf.placeholder(
        dtype=test_dataset.tf_reader['label']['dtype'],
        shape=[config.test_batch_size] + test_dataset.label_size,
        name='test_labels')
    aug_test_ims, aug_test_labels = [], []
    split_test_ims = tf.split(test_images, config.test_batch_size, axis=0)
    split_test_labels = tf.split(test_labels, config.test_batch_size, axis=0)
    for te_im, te_la in zip(
            split_test_ims,
            split_test_labels):
        te_im, te_la = data_loader.image_augmentations(
            image=tf.squeeze(te_im),
            label=tf.squeeze(te_la),
            model_input_image_size=test_dataset.model_input_image_size,
            data_augmentations=config.test_augmentations)
        aug_test_ims += [te_im]
        aug_test_labels += [te_la]
    aug_test_ims = tf.stack(aug_test_ims, axis=0)
    aug_test_labels = tf.stack(aug_test_labels, axis=0)
    return test_images, test_labels, aug_test_ims, aug_test_labels


def build_model(
        exp_params,
        config,
        log,
        dt_string,
        gpu_device,
        cpu_device,
        use_db=True,
        placeholders=False,
        checkpoint=None,
        test=False,
        tensorboard_images=False):
    """Standard model building routines."""
    config = py_utils.add_to_config(
        d=exp_params,
        config=config)
    exp_label = '%s_%s' % (exp_params['experiment'], py_utils.get_dt_stamp())
    directories = py_utils.prepare_directories(config, exp_label)
    dataset_module = py_utils.import_module(
        pre_path=config.dataset_classes,
        module=config.train_dataset)
    train_dataset_module = dataset_module.data_processing()
    (
        train_data,
        train_means_image,
        train_means_label) = py_utils.get_data_pointers(
        dataset=train_dataset_module.output_name,
        base_dir=config.tf_records,
        cv='train')
    dataset_module = py_utils.import_module(
        pre_path=config.dataset_classes,
        module=config.val_dataset)
    val_dataset_module = dataset_module.data_processing()
    val_data, val_means_image, val_means_label = py_utils.get_data_pointers(
        dataset=val_dataset_module.output_name,
        base_dir=config.tf_records,
        cv='val')

    # Create data tensors
    if hasattr(train_dataset_module, 'aux_loss'):
        train_aux_loss = train_dataset_module.aux_loss
    else:
        train_aux_loss = None
    with tf.device(cpu_device):
        if placeholders and not test:
            # Train with placeholders
	    (
                pl_train_images,
                pl_train_labels,
                pl_val_images, 
                pl_val_labels,
                train_images,
                train_labels,
                val_images,
                val_labels) = get_placeholders(
                    train_dataset=train_dataset_module,
                    val_dataset=val_dataset_module,
                    config=config)
            train_module_data = train_dataset_module.get_data()
            val_module_data = val_dataset_module.get_data()
            placeholders = {
                'train': {'images': train_module_data[0]['train'], 'labels': train_module_data[1]['train']},
                'val': {'images': val_module_data[0]['val'], 'labels': val_module_data[1]['val']},
            }
            train_aux, val_aux = None, None
        elif placeholders and test:
            test_dataset_module = train_dataset_module
            # Test with placeholders
            (
                pl_test_images,
                pl_test_labels,
                test_images,
                test_labels) = get_placeholders_test(
                    test_dataset=test_dataset_module,
                    config=config)
            test_module_data = test_dataset_module.get_data()
            placeholders = {
                'test': {'images': test_module_data[0]['test'], 'labels': test_module_data[1]['test']},
            }
            train_aux, val_aux = None, None
        else:
            train_images, train_labels, train_aux = data_loader.inputs(
                dataset=train_data,
                batch_size=config.train_batch_size,
                model_input_image_size=train_dataset_module.model_input_image_size,
                tf_dict=train_dataset_module.tf_dict,
                data_augmentations=config.train_augmentations,
                num_epochs=config.epochs,
                aux=train_aux_loss,
                tf_reader_settings=train_dataset_module.tf_reader,
                shuffle=config.shuffle_train)
            val_images, val_labels, val_aux = data_loader.inputs(
                dataset=val_data,
                batch_size=config.val_batch_size,
                model_input_image_size=val_dataset_module.model_input_image_size,
                tf_dict=val_dataset_module.tf_dict,
                data_augmentations=config.val_augmentations,
                num_epochs=None,
                tf_reader_settings=val_dataset_module.tf_reader,
                shuffle=config.shuffle_val)

    # Build training and val models
    model_spec = py_utils.import_module(
        module=config.model,
        pre_path=config.model_classes)
    if hasattr(train_dataset_module, 'force_output_size'):
        train_dataset_module.output_size = train_dataset_module.force_output_size
    if hasattr(val_dataset_module, 'force_output_size'):
        val_dataset_module.output_size = val_dataset_module.force_output_size 

    # Route test vs train/val
    h_check = [x
        for x in tf.trainable_variables()
        if 'homunculus' in x.name or 'humonculus' in x.name]
    if test:
        with tf.device(gpu_device):
            test_logits, test_vars = model_spec.build_model(
                data_tensor=test_images,
                reuse=None,
                training=False,
                output_shape=test_dataset_module.output_size)

        # Derive loss
        if isinstance(config.loss_function, list):
            test_loss = config.val_loss_function
        else:
            test_loss = config.loss_function
        test_loss, _ = losses.derive_loss(
            labels=test_labels,
            logits=test_logits,
            loss_type=test_loss,
            loss_weights=config.loss_weights)

        # Derive score
        test_score = losses.derive_score(
            labels=test_labels,
            logits=test_logits,
            loss_type=config.loss_function,
            dataset=train_dataset_module.output_name,
            score_type=config.score_function)

        # Initialize model
        sess, saver, summary_op, summary_writer, coord, threads = initialize_tf(
            config=config,
            placeholders=placeholders,
            directories=directories)

        if placeholders:
            test_images = pl_test_images
            test_labels = pl_test_labels

        test_dict = {
            'test_loss': test_loss,
            'test_score': test_score,
            'test_images': test_images,
            'test_labels': test_labels,
            'test_logits': test_logits
        }
        if len(h_check):
            test_dict['homunculus'] = h_check[0]
    else:
        with tf.device(gpu_device):
            train_logits, train_vars = model_spec.build_model(
                data_tensor=train_images,
                reuse=None,
                training=True,
                output_shape=train_dataset_module.output_size)
            val_logits, val_vars = model_spec.build_model(
                data_tensor=val_images,
                reuse=tf.AUTO_REUSE,
                training=False,
                output_shape=val_dataset_module.output_size)

        # Derive loss
        train_loss, train_loss_list = losses.derive_loss(
            labels=train_labels,
            logits=train_logits,
            loss_type=config.loss_function,
            loss_weights=config.loss_weights)

        # Add regularization
        # wd = (1e-4 * tf.add_n(
        #     [tf.nn.l2_loss(v) for v in tf.trainable_variables()
        #     if 'batch_normalization' not in v.name and
        #     'block' not in v.name and
        #     'training' not in v.name]))
        wd_vars = [v for v in tf.trainable_variables() if 'regularize' in v.name and 'batch_normalization' not in v.name and 'training' not in v.name and 'instance' not in v.name]
        if len(wd_vars):
            if 0:
                wd = 1e-2 * tf.add_n([tf.reduce_mean(tf.abs(v)) for v in wd_vars])
            else:
                wd = 1e-4 * tf.add_n([tf.nn.l2_loss(v) for v in wd_vars])
            train_loss += wd
        if isinstance(config.loss_function, list):
            val_loss = config.val_loss_function
        else:
            val_loss = config.loss_function
        val_loss, val_loss_list = losses.derive_loss(
            labels=val_labels,
            logits=val_logits,
            loss_type=val_loss,
            loss_weights=config.loss_weights)
        tf.summary.scalar('train_loss', train_loss)
        tf.summary.scalar('val_loss', val_loss)

        # Derive auxilary losses
        if hasattr(train_dataset_module, 'aux_loss'):
            for k, v in train_vars.iteritems():
                if k in train_dataset_module.aux_loss.keys():
                    aux_loss_type, scale = train_dataset_module.aux_loss[k]
                    train_loss += (losses.derive_loss(
                        labels=train_aux,
                        logits=v,
                        loss_type=aux_loss_type) * scale)

        # Derive score
        train_score = losses.derive_score(
            labels=train_labels,
            logits=train_logits,
            loss_type=config.loss_function,
            dataset=train_dataset_module.output_name,
            score_type=config.score_function)
        val_score = losses.derive_score(
            labels=val_labels,
            logits=val_logits,
            loss_type=config.loss_function,
            dataset=train_dataset_module.output_name,
            score_type=config.score_function)
        tf.summary.scalar('train_score', train_score)
        tf.summary.scalar('val_score', val_score)
        if tensorboard_images and not placeholders:
            tf.summary.image('train_images', train_images)
            tf.summary.image('val_images', val_images)

        # Build optimizer
        freeze_lr = None
        config.lr_placeholder = tf.placeholder(tf.float32, name='main_lr', shape=[])
        if hasattr(config, 'freeze_lr'):
            freeze_lr = config.freeze_lr
        train_op = optimizers.get_optimizer(
            train_loss,
            config.lr_placeholder,
            config.optimizer,
            freeze_lr=freeze_lr,
            dtype=train_images.dtype)

        # Initialize model
        sess, saver, summary_op, summary_writer, coord, threads = initialize_tf(
            config=config,
            placeholders=placeholders,
            directories=directories)

        # Create dictionaries of important training and validation information
        if placeholders:
            train_images = pl_train_images
            train_labels = pl_train_labels
            val_images = pl_val_images
            val_labels = pl_val_labels

        train_dict = {
            'train_loss': train_loss,
            'train_score': train_score,
            'train_images': train_images,
            'train_labels': train_labels,
            'train_logits': train_logits,
            'train_op': train_op
        }
        if train_loss_list is not None:
            for k, v in train_loss_list.iteritems():
                train_dict[k] = v
        if train_aux is not None:
            train_dict['train_aux']  = train_aux    

        if isinstance(train_vars, dict):
            for k, v in train_vars.iteritems():
                train_dict[k] = v
        else:
            train_dict['activity'] = train_vars
        if hasattr(config, 'save_gradients'):
            grad = tf.gradients(train_logits, train_images)[0]
            if grad is not None:
                train_dict['gradients'] = grad
            else:
                log.warning('Could not calculate val gradients.')

        val_dict = {
            'val_loss': val_loss,
            'val_score': val_score,
            'val_images': val_images,
            'val_logits': val_logits,
            'val_labels': val_labels
        }
        if val_loss_list is not None:
            for k, v in val_loss_list.iteritems():
                val_dict[k] = v
        if val_aux is not None:
            val_dict['aux']  = val_aux

        if isinstance(val_vars, dict):
            for k, v in val_vars.iteritems():
                val_dict[k] = v
        else:
            val_dict['activity'] = val_vars
        if hasattr(config, 'save_gradients'):
            grad = tf.gradients(val_logits, val_images)[0]
            if grad is not None:
                val_dict['gradients'] = grad
            else:
                log.warning('Could not calculate val gradients.')
        if len(h_check):
            val_dict['homunculus'] = h_check[0]

    # Count parameters
    num_params = tf_fun.count_parameters(var_list=tf.trainable_variables())
    print 'Model has approximately %s trainable params.' % num_params
    if test:
        return training.test_loop(
            log=log,
            config=config,
            sess=sess,
            summary_op=summary_op,
            summary_writer=summary_writer,
            saver=saver,
            directories=directories,
            test_dict=test_dict,
            exp_label=exp_label,
            num_params=num_params,
            checkpoint=checkpoint,
            save_weights=config.save_weights,
            save_checkpoints=config.save_checkpoints,
            save_activities=config.save_activities,
            save_gradients=config.save_gradients,
            placeholders=placeholders)
    else:
        # Start training loop
        training.training_loop(
            log=log,
            config=config,
            coord=coord,
            sess=sess,
            summary_op=summary_op,
            summary_writer=summary_writer,
            saver=saver,
            threads=threads,
            directories=directories,
            train_dict=train_dict,
            val_dict=val_dict,
            exp_label=exp_label,
            num_params=num_params,
            checkpoint=checkpoint,
            use_db=use_db,
            save_weights=config.save_weights,
            save_checkpoints=config.save_checkpoints,
            save_activities=config.save_activities,
            save_gradients=config.save_gradients,
            placeholders=placeholders)


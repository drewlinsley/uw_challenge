"""Model training with tfrecord queues or placeholders."""
import os
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import logger
from utils import py_utils
from ops import data_to_tfrecords
from tqdm import tqdm
from db import db


def val_status(
        log,
        dt,
        step,
        train_loss,
        rate,
        timer,
        score_function,
        train_score,
        val_score,
        summary_dir):
    """Print training status."""
    format_str = (
        '%s: step %d, loss = %.2f (%.1f examples/sec; '
        '%.3f sec/batch) | Training %s = %s | '
        'Validation %s = %s | logdir = %s')
    log.info(
        format_str % (
            dt,
            step,
            train_loss,
            rate,
            timer,
            score_function,
            train_score,
            score_function,
            val_score,
            summary_dir))


def train_status(
        log,
        dt,
        step,
        train_loss,
        rate,
        timer,
        score_function,
        train_score):
    """Print training status."""
    format_str = (
        '%s: step %d, loss = %.5f (%.1f examples/sec; '
        '%.3f sec/batch) | Training %s = %s')
    log.info(
        format_str % (
            dt,
            step,
            train_loss,
            rate,
            timer,
            score_function,
            train_score))


def training_step(
        sess,
        train_dict,
        feed_dict=False):
    """Run a step of training."""
    start_time = time.time()
    if feed_dict:
        it_train_dict = sess.run(train_dict, feed_dict=feed_dict)
    else:
        it_train_dict = sess.run(train_dict)
    train_score = it_train_dict['train_score']
    train_loss = it_train_dict['train_loss']
    duration = time.time() - start_time
    timesteps = duration
    return train_score, train_loss, it_train_dict, timesteps


def validation_step(
        sess,
        val_dict,
        config,
        log,
        sequential=True,
        dict_image_key='val_images',
        dict_label_key='val_labels',
        eval_score_key='val_score',
        eval_loss_key='val_loss',
        val_images=False,
        val_labels=False,
        val_batch_idx=None,
        val_batches=False):
    it_val_score = np.asarray([])
    it_val_loss = np.asarray([])
    start_time = time.time()
    if np.unique(val_batch_idx) == 0:
        sequential = True
    for idx in range(config.validation_steps):
        # Validation accuracy as the average of n batches
        if val_batch_idx is not None:
            if not sequential:
                it_val_batch_idx = val_batch_idx[
                    np.random.permutation(len(val_batch_idx))]
                val_step = np.random.randint(low=0, high=val_batch_idx.max())
            else:
                it_val_batch_idx = val_batch_idx
                val_step = idx
            it_idx = it_val_batch_idx == val_step
            it_ims = val_images[it_idx]
            it_labs = val_labels[it_idx]
            if isinstance(it_ims[0], basestring):
                it_ims = np.asarray(
                    [
                        data_to_tfrecords.load_image(im)
                        for im in it_ims])
            if isinstance(it_labs[0], basestring):
                it_labs = np.asarray(
                    [
                        data_to_tfrecords.load_image(im)
                        for im in it_labs])
            feed_dict = {
                val_dict[dict_image_key]: it_ims,
                val_dict[dict_label_key]: it_labs
            }
            it_val_dict = sess.run(val_dict, feed_dict=feed_dict)
        else:
            it_val_dict = sess.run(val_dict)
        it_val_score = np.append(
            it_val_score,
            it_val_dict[eval_score_key])
        it_val_loss = np.append(
            it_val_loss,
            it_val_dict[eval_loss_key])
    val_score = it_val_score.mean()
    val_lo = it_val_loss.mean()
    duration = time.time() - start_time
    return val_score, val_lo, it_val_dict, duration


def save_progress(
        config,
        weight_dict,
        it_val_dict,
        exp_label,
        step,
        directories,
        sess,
        saver,
        val_check,
        val_score,
        val_loss,
        val_perf,
        train_score,
        train_loss,
        timer,
        num_params,
        log,
        use_db,
        summary_op,
        summary_writer,
        save_activities,
        save_gradients,
        save_checkpoints):
    """Save progress and important data."""
    if config.save_weights and val_check:
        it_weights = {
            k: it_val_dict[k] for k in weight_dict.keys()}
        py_utils.save_npys(
            data=it_weights,
            model_name='%s_%s' % (
                exp_label,
                step),
            output_string=directories['weights'])

    if save_activities and val_check:
        py_utils.save_npys(
            data=it_val_dict,
            model_name='%s_%s' % (
                exp_label,
                step),
            output_string=directories['weights'])

    ckpt_path = os.path.join(
        directories['checkpoints'],
        'model_%s.ckpt' % step)
    val_check = np.where(val_loss < val_perf)[0]
    if save_checkpoints and len(val_check):
        log.info('Saving checkpoint to: %s' % ckpt_path)
        saver.save(
            sess,
            ckpt_path,
            global_step=step)
        val_check = val_check[0]
        val_perf[val_check] = val_loss
              
    if save_gradients and val_check:
        # np.savez(
        #     os.path.join(
        #         config.results,
        #         '%s_train_gradients' % exp_label),
        #     **it_train_dict)
        np.savez(
            os.path.join(
                config.results,
                '%s_val_gradients' % exp_label),
            **it_val_dict)

    if use_db:
        db.update_performance(
            experiment_id=config._id,
            experiment=config.experiment,
            train_score=float(train_score),
            train_loss=float(train_loss),
            val_score=float(val_score),
            val_loss=float(val_loss),
            step=step,
            num_params=int(num_params),
            ckpt_path=ckpt_path,
            results_path=config.results,
            summary_path=directories['summaries'])

    # Summaries
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, step)
    return val_perf


def test_loop(
        config,
        sess,
        summary_op,
        summary_writer,
        saver,
        directories,
        test_dict,
        exp_label,
        num_params,
        log,
        placeholders=False,
        checkpoint=None,
        save_weights=False,
        save_checkpoints=False,
        save_activities=False,
        save_gradients=False):
    """Run the model test loop."""
    if placeholders:
        test_images = placeholders['test']['images']
        test_labels = placeholders['test']['labels']
        test_batches = len(test_images) / config.test_batch_size
        test_batch_idx = np.arange(
            test_batches).reshape(-1, 1).repeat(
                config.test_batch_size)
        test_images = test_images[:len(test_batch_idx)]
        test_labels = test_labels[:len(test_batch_idx)]

        # Check that labels are appropriate shape
        tf_label_shape = test_dict['test_labels'].get_shape().as_list()
        np_label_shape = test_labels.shape
        if len(tf_label_shape) == 2 and len(np_label_shape) == 1:
            test_labels = test_labels[..., None]
        elif len(tf_label_shape) == len(np_label_shape):
            pass
        else:
            raise RuntimeError(
                'Mismatch label shape np: %s vs. tf: %s' % (
                    np_label_shape,
                    tf_label_shape))

        # Loop through all the images
        config.validation_steps = test_batches
        test_score, test_lo, it_test_dict, duration = validation_step(
            sequential=True,
            sess=sess,
            val_dict=test_dict,
            config=config,
            log=log,
            dict_image_key='test_images',
            dict_label_key='test_labels',
            eval_score_key='test_score',
            eval_loss_key='test_loss',
            val_images=test_images,
            val_labels=test_labels,
            val_batch_idx=test_batch_idx,
            val_batches=test_batches)
        return {
            'scores': test_score,
            'losses': test_lo,
            'test_dict': it_test_dict,
            'duration': duration}
    else:
        raise NotImplementedError('Testing with tf records is not yet implemented.')


def training_loop(
        config,
        coord,
        sess,
        summary_op,
        summary_writer,
        saver,
        threads,
        directories,
        train_dict,
        val_dict,
        exp_label,
        num_params,
        use_db,
        log,
        placeholders=False,
        checkpoint=None,
        save_weights=False,
        save_checkpoints=False,
        save_activities=False,
        save_gradients=False):
    """Run the model training loop."""
    if checkpoint is not None:
        saver.restore(sess, checkpoint)
        print 'Restored checkpoint %s' % checkpoint
    val_perf = np.asarray([np.inf])
    step = 0
    if save_weights:
        try:
            weight_dict = {v.name: v for v in tf.trainable_variables()}
            val_dict = dict(
                val_dict,
                **weight_dict)
        except Exception:
            raise RuntimeError('Failed to find weights to save.')
    else:
        weight_dict = None
    if placeholders:
        train_images = placeholders['train']['images']
        val_images = placeholders['val']['images']
        train_labels = placeholders['train']['labels']
        val_labels = placeholders['val']['labels']
        train_batches = len(train_images) / config.train_batch_size
        train_batch_idx = np.arange(
            train_batches).reshape(-1, 1).repeat(
                config.train_batch_size)
        train_images = train_images[:len(train_batch_idx)]
        train_labels = train_labels[:len(train_batch_idx)]
        val_batches = len(val_images) / config.val_batch_size
        val_batch_idx = np.arange(
            val_batches).reshape(-1, 1).repeat(
                config.val_batch_size)
        val_images = val_images[:len(val_batch_idx)]
        val_labels = val_labels[:len(val_batch_idx)]

        # Check that labels are appropriate shape
        tf_label_shape = train_dict['train_labels'].get_shape().as_list()
        np_label_shape = train_labels.shape
        if len(tf_label_shape) == 2 and len(np_label_shape) == 1:
            train_labels = train_labels[..., None]
            val_labels = val_labels[..., None]
        elif len(tf_label_shape) == len(np_label_shape):
            pass
        else:
            raise RuntimeError(
                'Mismatch label shape np: %s vs. tf: %s' % (
                    np_label_shape,
                    tf_label_shape))

        # Start training
        for epoch in tqdm(
                range(config.epochs),
                desc='Epoch',
                total=config.epochs):
            for train_batch in range(train_batches):
                data_idx = train_batch_idx == train_batch
                it_train_images = train_images[data_idx]
                it_train_labels = train_labels[data_idx]
                if isinstance(it_train_images[0], basestring):
                    it_train_images = np.asarray(
                        [
                            data_to_tfrecords.load_image(im)
                            for im in it_train_images])
                feed_dict = {
                    train_dict['train_images']: it_train_images,
                    train_dict['train_labels']: it_train_labels
                }
                (
                    train_score,
                    train_loss,
                    it_train_dict,
                    timer) = training_step(
                    sess=sess,
                    train_dict=train_dict,
                    feed_dict=feed_dict)
                if step % config.validation_period == 0:
                    val_score, val_lo, it_val_dict, duration = validation_step(
                        sess=sess,
                        val_dict=val_dict,
                        config=config,
                        log=log,
                        val_images=val_images,
                        val_labels=val_labels,
                        val_batch_idx=val_batch_idx,
                        val_batches=val_batches)

                    # Save progress and important data
                    try:
                        val_check = np.where(val_lo < val_perf)[0]
                        val_perf = save_progress(
                            config=config,
                            val_check=val_check,
                            weight_dict=weight_dict,
                            it_val_dict=it_val_dict,
                            exp_label=exp_label,
                            step=step,
                            directories=directories,
                            sess=sess,
                            saver=saver,
                            val_score=val_score,
                            val_loss=val_lo,
                            val_perf=val_perf,
                            train_score=train_score,
                            train_loss=train_loss,
                            timer=duration,
                            num_params=num_params,
                            log=log,
                            use_db=use_db,
                            summary_op=summary_op,
                            summary_writer=summary_writer,
                            save_activities=save_activities,
                            save_gradients=save_gradients,
                            save_checkpoints=save_checkpoints)
                    except Exception as e:
                        log.info('Failed to save checkpoint: %s' % e)

                    # Training status and validation accuracy
                    val_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        score_function=config.score_function,
                        train_score=train_score,
                        val_score=val_score,
                        summary_dir=directories['summaries'])
                else:
                    # Training status
                    train_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        score_function=config.score_function,
                        train_score=train_score)

                # End iteration
                val_perf = np.concatenate([val_perf, [val_lo]])
                step += 1

    else:
        try:
            while not coord.should_stop():
                (
                    train_score,
                    train_loss,
                    it_train_dict,
                    duration) = training_step(
                    sess=sess,
                    train_dict=train_dict)
                if step % config.validation_period == 0:
                    val_score, val_lo, it_val_dict, duration = validation_step(
                        sess=sess,
                        val_dict=val_dict,
                        config=config,
                        log=log)

                    # Save progress and important data
                    try:
                        val_check = np.where(val_lo < val_perf)[0]
                        val_perf = save_progress(
                            config=config,
                            val_check=val_check,
                            weight_dict=weight_dict,
                            it_val_dict=it_val_dict,
                            exp_label=exp_label,
                            step=step,
                            directories=directories,
                            sess=sess,
                            saver=saver,
                            val_score=val_score,
                            val_loss=val_lo,
                            val_perf=val_perf,
                            train_score=train_score,
                            train_loss=train_loss,
                            timer=duration,
                            num_params=num_params,
                            log=log,
                            use_db=use_db,
                            summary_op=summary_op,
                            summary_writer=summary_writer,
                            save_activities=save_activities,
                            save_gradients=save_gradients,
                            save_checkpoints=save_checkpoints)
                    except Exception as e:
                        log.info('Failed to save checkpoint: %s' % e)

                    # Training status and validation accuracy
                    val_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        score_function=config.score_function,
                        train_score=train_score,
                        val_score=val_score,
                        summary_dir=directories['summaries'])
                else:
                    # Training status
                    train_status(
                        log=log,
                        dt=datetime.now(),
                        step=step,
                        train_loss=train_loss,
                        rate=config.val_batch_size / duration,
                        timer=float(duration),
                        score_function=config.score_function,
                        train_score=train_score)

                # End iteration
                step += 1
        except tf.errors.OutOfRangeError:
            log.info(
                'Done training for %d epochs, %d steps.' % (
                    config.epochs, step))
            log.info('Saved to: %s' % directories['checkpoints'])
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
    return


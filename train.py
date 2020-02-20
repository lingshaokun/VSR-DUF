import tensorflow as tf
from tensorflow.python.platform import flags
import os
import json

from nets import FR_16L
from utils import DynFilter3D, depth_to_space_3D, Huber

tf.logging.set_verbosity(tf.logging.INFO)

flags.DEFINE_integer('num_epochs', 22, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 8, 'Batch size')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_string('train_dir', 'data/train/',
                    'Filename of training dataset')
flags.DEFINE_string('eval_dir', 'data/val/',
                    'Filename of evaluation dataset')
flags.DEFINE_string('model_dir', 'modelsa/',
                    'Directory to save models')
flags.DEFINE_integer('log_step_count_steps', 100,
                     'log_step_count_steps')
flags.DEFINE_integer('save_summary_steps', 100,
                     'save_summary_steps')
flags.DEFINE_integer('save_checkpoints_steps', 100,
                     'save_checkpoints_steps')
flags.DEFINE_integer('keep_checkpoint_max', 20,
                     'keep_checkpoint_max')

FLAGS = flags.FLAGS


def duf_model_fn(features, labels, mode):
    """Model function for cifar10"""
    r = 4
    T_in = 7
    is_train = tf.constant(mode == tf.estimator.ModeKeys.TRAIN, dtype=tf.bool)
    # Input layer
    Fx, Rx = FR_16L(features, is_train, r)

    x_c = []
    for c in range(3):
        t = DynFilter3D(features[:, T_in // 2:T_in // 2 + 1, :, :, c], Fx[:, 0, :, :, :, :], [1, 5, 5])  # [B,H,W,R*R]
        t = tf.depth_to_space(t, r)  # [B,H*R,W*R,1]
        x_c += [t]
    x = tf.concat(x_c, axis=3)  # [B,H*R,W*R,3]
    x = tf.expand_dims(x, axis=1)

    Rx = depth_to_space_3D(Rx, r)  # [B,1,H*R,W*R,3]
    x += Rx

    # Predicition
    sr_img = tf.cast(tf.clip_by_value(x, 0, 1) * 255, dtype=tf.uint8)
    loss = get_cost(x, labels)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'sr_image': sr_img}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step=tf.train.get_global_step(), decay_steps=12500, decay_rate=0.1, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    hr_img = tf.cast(labels * 255, dtype=tf.uint8)
    PSNR, psnr_update_op = tf.metrics.mean(tf.image.psnr(sr_img, hr_img, max_val=255))

    SSIM, ssim_update_op = tf.metrics.mean(tf.image.ssim(sr_img, hr_img, max_val=255))
    tf.summary.scalar('PSNR', PSNR)
    tf.summary.scalar('SSIM', SSIM)
    eval_metric_ops = {
        'PSNR': (PSNR, psnr_update_op),
        'SSIM': (SSIM, ssim_update_op)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def get_cost(x, H_out_true):
    out_H = tf.clip_by_value(x, 0, 1, name='out_H')

    cost = Huber(y_true=H_out_true, y_pred=out_H, delta=0.01)
    return cost


def parse_hr_example(serialized_example):
    features = dict()
    features['image'] = tf.FixedLenFeature([], dtype=tf.string)
    parsed_features = tf.parse_single_example(serialized_example, features)
    im = tf.decode_raw(parsed_features['image'], dtype=tf.uint8)
    im = tf.cast(im, tf.float32)
    im = tf.expand_dims(im, axis=0)
    return im


def parse_lr_example(serialized_example, T_in=7):
    features = {}
    for t in range(T_in):
        features['img' + str(t)] = tf.FixedLenFeature([], dtype=tf.string)
    features['img' + str(T_in)] = tf.FixedLenFeature([], dtype=tf.string)
    parsed_features = tf.parse_single_example(serialized_example, features)
    # for t in range(T_in):
    #     image_raw = parsed_features['image' + str(t)]
    #     image = tf.image.decode_png(image_raw, dtype=tf.uint8)
    imgs = []
    for i in range(T_in):
        imgs.append(tf.reshape(tf.decode_raw(parsed_features['img' + str(i)], out_type=tf.uint8) / 255, [64, 112, 3]))
    lable = tf.reshape(tf.decode_raw(parsed_features['img' + str(T_in)], out_type=tf.uint8) / 255, [256, 448, 3])

    feature = tf.stack(imgs)
    lable = tf.expand_dims(lable, axis=0)
    return feature, lable


def save_hp_to_json():
    """
    Save hyperparameters to a json file
    :return:
    """

    filename = os.path.join(FLAGS.model_dir, 'hparams.json')
    hparams = FLAGS.flag_values_dict()
    with open(filename, 'w') as f:
        json.dump(hparams, f, indent=4, sort_keys=True)


def get_estimator(config):
    """
    Return the model as a Tensorflow Estimator object.
    :param config:
    :return:
    """

    return tf.estimator.Estimator(model_fn=duf_model_fn, config=config)


def main(unused_argv):
    def train_input_fn():
        train_dataset_list = [FLAGS.train_dir + i for i in os.listdir(FLAGS.train_dir)]
        train_dataset = tf.data.TFRecordDataset(train_dataset_list)
        train_dataset = train_dataset.map(parse_lr_example)
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.repeat(FLAGS.num_epochs)

        train_dataset = train_dataset.batch(FLAGS.batch_size)
        train_dataset = train_dataset.prefetch(tf.contrib.data.AUTOTUNE)
        train_iterator = train_dataset.make_one_shot_iterator()

        feature, label = train_iterator.get_next()

        return feature, label

    def eval_input_fn():
        eval_dataset_list = [FLAGS.eval_dir + i for i in os.listdir(FLAGS.eval_dir)]
        eval_dataset = tf.data.TFRecordDataset(eval_dataset_list)
        eval_dataset = eval_dataset.map(parse_lr_example)
        # eval_dataset = eval_dataset.repeat(FLAGS.num_epochs)
        eval_dataset = eval_dataset.batch(FLAGS.batch_size).prefetch(tf.contrib.data.AUTOTUNE)
        eval_iterator = eval_dataset.make_one_shot_iterator()
        feature, label = eval_iterator.get_next()

        return feature, label

    config = tf.estimator.RunConfig()
    config = config.replace(model_dir=FLAGS.model_dir, log_step_count_steps=FLAGS.log_step_count_steps, save_summary_steps=FLAGS.save_summary_steps,
                            save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    duf_estimator = get_estimator(config=config)

    # # Train
    # duf_estimator.train(input_fn=train_input_fn)
    # # Evaluation
    # eval_results = duf_estimator.evaluate(input_fn=eval_input_fn)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=30000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=30, start_delay_secs=60, throttle_secs=120)
    tf.estimator.train_and_evaluate(duf_estimator, train_spec, eval_spec)

    tf.logging.info('Saving hyperparameters ...')
    save_hp_to_json()


if __name__ == '__main__':
    tf.app.run()

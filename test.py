import numpy as np
import tensorflow as tf
import cv2 as cv
import os

from train import get_estimator

flags = tf.app.flags
flags.DEFINE_string('saved_model_dir', 'models/',
                    'Output directory for model and training stats.')
flags.DEFINE_string('test_dir', 'data/test/',
                    'Filename of testing dataset')
flags.DEFINE_string('result_dir', 'data/result/',
                    'Filename of testing dataset')
FLAGS = flags.FLAGS


def infer(argv=None):
    """
    Run the inference and return the result.
    :param argv:
    :return:
    """
    config = tf.estimator.RunConfig()
    config = config.replace(model_dir=FLAGS.saved_model_dir)
    estimator = get_estimator(config)
    result = estimator.predict(input_fn=test_input_fn)
    result_dir = FLAGS.result_dir
    for i, r in enumerate(result):
        # print(r['sr_image'].shape)
        result_path = result_dir + '{:03d}.png'.format(i)
        cv.imwrite(result_path, r['sr_image'][0])
    # return result


# def predict_input_fn():
#     '''Input function for prediction.'''
#     with tf.variable_scope('TEST_INPUT'):
#         image = tf.constant(load_image(), dtype=tf.float32)
#         dataset = tf.data.Dataset.from_tensor_slices((image,))
#         return dataset.batch(1).make_one_shot_iterator().get_next()


# def load_image():
#     """
#     Load image into numpy array.
#     :return:
#     """
#     T_in = 7
#     images = np.zeros((T_in, 144, 180, 3), dtype='float32')
#     for i, file in enumerate(os.listdir('./data/test/')):
#         file.replace('\\', '/')
#         image = cv.imread(file)
#         images[i, :] = image
#     images = np.asarray(images)
#     images = np.lib.pad(images, pad_width=((T_in // 2, T_in // 2), (0, 0), (0, 0), (0, 0)),
#                         mode='constant')
#     images_padded = images[np.newaxis, :, :, :, :]
#     return images_padded


def test_input_fn():
    test_dataset_list = [FLAGS.test_dir + i for i in os.listdir(FLAGS.test_dir)]
    test_dataset_list.sort()
    test_dataset = tf.data.TFRecordDataset(test_dataset_list)
    test_dataset = test_dataset.map(parse_test_example)
    # eval_dataset = eval_dataset.repeat(FLAGS.num_epochs)
    test_dataset = test_dataset.batch(FLAGS.batch_size)
    test_iterator = test_dataset.make_one_shot_iterator()
    lr = test_iterator.get_next()

    return lr


def parse_test_example(serialized_example, T_in=7):
    features = {}
    for t in range(T_in):
        features['img' + str(t)] = tf.FixedLenFeature([], dtype=tf.string)
    features['h'] = tf.FixedLenFeature([], tf.int64)
    features['w'] = tf.FixedLenFeature([], tf.int64)
    parsed_features = tf.parse_single_example(serialized_example, features)
    # for t in range(T_in):
    #     image_raw = parsed_features['image' + str(t)]
    #     image = tf.image.decode_png(image_raw, dtype=tf.uint8)
    imgs = []
    for i in range(T_in):
        imgs.append(tf.decode_raw(parsed_features['img' + str(i)], out_type=tf.uint8) / 255)
    h = tf.cast(parsed_features['h'], tf.int32)
    w = tf.cast(parsed_features['w'], tf.int32)
    imgs = [tf.reshape(i, [h, w, 3]) for i in imgs]
    feature = tf.stack(imgs)
    return feature


if __name__ == '__main__':
    tf.app.run(main=infer)

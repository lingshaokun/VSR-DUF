import tensorflow as tf
import cv2 as cv
from tqdm import trange
import os
import numpy as np

train_sample_size = 10000  # max 64612
val_sample_size = 200  # max 7824
vimeo_data_dir = 'D:/Jade/Downloads/Compressed/vimeo_septuplet/sequences/'
vid4_data_dir = 'D:/SR/Vid4/'


def train_img2tfrecord(data_dir=vimeo_data_dir, input_list_txt='./data/sep_trainlist.txt', tfr_dir='./data/train/', size=train_sample_size):
    T_in = 7
    r = 1 / 4
    count = 0
    tfr_num = 0
    with open(input_list_txt, 'r') as f_in:
        writer = tf.python_io.TFRecordWriter(tfr_dir + '{:03d}.tfrecord'.format(tfr_num))
        for i in trange(size):  # 每个样本
            if count >= 1000:
                count = 0
                tfr_num += 1
                writer = tf.python_io.TFRecordWriter(tfr_dir + '{:03d}.tfrecord'.format(tfr_num))
            f1, f2 = f_in.readline().strip().split('/')
            imgs_dir = data_dir + f1 + '/' + f2 + '/'
            img_paths = [imgs_dir + n for n in os.listdir(imgs_dir)]
            img_paths.sort()
            imgs = []
            for f in img_paths:
                imgs.append(cv.imread(f))
            lr_imgs = []
            for f in imgs:
                # lr_imgs.append(cv.resize(f, dsize=None, fx=r, fy=r, interpolation=cv.INTER_CUBIC).tobytes())
                lr_imgs.append(cv.pyrDown(cv.pyrDown(f)).tobytes())
            lr_imgs.append(imgs[T_in // 2].tobytes())
            feature = {}
            for f in range(T_in + 1):
                feature['img' + str(f)] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[lr_imgs[f]]))
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

            count += 1


def test_img2tfrecord(data_dir=vid4_data_dir, tfr_dir='./data/test_pydown/'):
    T_in = 7
    r = 1 / 4
    tfr_num = 0
    txt_lists = ['./data/calendar.txt', './data/city.txt', './data/foliage.txt', './data/walk.txt']
    shape_list = [(576, 720), (576, 704), (480, 720), (480, 720)]
    for t in range(len(txt_lists)):
        with open(txt_lists[t], 'r') as f_in:  # 每个文件夹
            lines = f_in.readlines()
            lines = [data_dir + i.strip() for i in lines]
            writer = tf.python_io.TFRecordWriter(tfr_dir + '{:03d}.tfrecord'.format(tfr_num))
            lr_imgs = []
            h = int(shape_list[t][0] * r)
            w = int(shape_list[t][1] * r)
            for i in trange(len(lines)):
                # lr_imgs.append(cv.resize(cv.imread(lines[i]), dsize=(w, h), interpolation=cv.INTER_CUBIC).tobytes())
                lr_imgs.append(cv.pyrDown(cv.pyrDown(cv.imread(lines[i]))).tobytes())
            zero_img = np.zeros([w, h, 3], dtype=np.uint8)
            for i in range(T_in // 2):
                lr_imgs.insert(0, zero_img.tobytes())
            for i in range(T_in // 2):
                lr_imgs.append(zero_img.tobytes())

            for i in trange(len(lines)):
                feature = {}
                for f in range(T_in):
                    feature['img' + str(f)] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[lr_imgs[i + f]]))
                feature['h'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[h]))
                feature['w'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[w]))
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

        tfr_num += 1


if __name__ == '__main__':
    #  train data
    train_img2tfrecord(tfr_dir='./data/train_c/')

    #  val data
    train_img2tfrecord(input_list_txt='./data/sep_testlist.txt', tfr_dir='./data/val_c/', size=val_sample_size)

    # test data
    # test_img2tfrecord()

    # val_test
    # train_img2tfrecord(input_list_txt='./data/sep_valtestlist.txt', tfr_dir='./data/valtest/', size=val_sample_size)

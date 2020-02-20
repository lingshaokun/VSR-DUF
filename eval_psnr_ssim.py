from skimage import measure

import cv2 as cv
import numpy as np
import os
from tqdm import tqdm


def PSNR(gt_path, pred_path):
    gt = cv.imread(gt_path).astype(np.float32)
    pr = cv.imread(pred_path).astype(np.float32)
    psnr = cv.PSNR(gt, pr)
    ssim = measure.compare_ssim(gt, pr, multichannel=True)
    return psnr, ssim


def get_all_files(root_dir):
    _files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            _files.append(os.path.join(root, f).replace('\\', '/'))
        for dirname in dirs:
            get_all_files(os.path.join(root, dirname).replace('\\', '/'))
    return _files


def bicubic_lr(img, rate, ininterpolation=cv.INTER_CUBIC):
    r = 1. / rate
    # w, h = img.shape[0:2]
    # img_lr = cv.resize(img, dsize=None, fx=r, fy=r, interpolation=ininterpolation)
    img_lr = cv.pyrDown(img)
    img_lr = cv.pyrDown(img_lr)
    return img_lr


def bicubic_hr(img, rate, ininterpolation=cv.INTER_CUBIC):
    r = 1. / rate
    img_lr = cv.resize(img, dsize=None, fx=r, fy=r, interpolation=ininterpolation)
    img_bicubic_hr = cv.resize(img_lr, (img.shape[1], img.shape[0]), interpolation=cv.INTER_CUBIC)
    return img_bicubic_hr


def hr2bicubic_lr(root_dir='./data/test_img_hr/', rate=4, ext='.png', ininterpolation=cv.INTER_CUBIC):
    assert os.path.isdir(root_dir), 'the path name is error!'

    root_dir = root_dir.replace('\\', '/').strip('/')
    out_root_dir = root_dir + '_bicubic_lr_x' + str(rate)

    if not os.path.exists(out_root_dir):
        os.mkdir(out_root_dir)

    files = get_all_files(root_dir)
    for f in tqdm(files):
        if f.endswith(ext):
            out_path = f.replace(root_dir, out_root_dir)
            out_dir, img_name = os.path.split(out_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            img = cv.imread(f)
            img_lr = bicubic_lr(img, rate, ininterpolation)
            cv.imwrite(out_path, img_lr)


def hr2bicubic_hr(root_dir='./data/test_img_hr/', rate=4, ext='.png', ininterpolation=cv.INTER_CUBIC):
    assert os.path.isdir(root_dir), 'the path name is error!'

    root_dir = root_dir.replace('\\', '/').strip('/')
    out_root_dir = root_dir + '_bicubic_hr_x' + str(rate)

    if not os.path.exists(out_root_dir):
        os.mkdir(out_root_dir)

    files = get_all_files(root_dir)
    for f in tqdm(files):
        if f.endswith(ext):
            out_path = f.replace(root_dir, out_root_dir)
            out_dir, img_name = os.path.split(out_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            img = cv.imread(f)
            img_lr = bicubic_hr(img, rate, ininterpolation)
            cv.imwrite(out_path, img_lr)


def main(gt_dir, pred_dir):
    gt_files = os.listdir(gt_dir)
    pred_files = os.listdir(pred_dir)
    assert len(gt_files) == len(pred_files), 'the gt number should be equal to pred number.'

    gt_files = [os.path.join(gt_dir, g).replace('\\', '/') for g in gt_files]
    pred_files = [os.path.join(pred_dir, p).replace('\\', '/') for p in pred_files]
    gt_files.sort()
    pred_files.sort()

    psnr_list = np.array([PSNR(g, p) for g, p in zip(gt_files, pred_files)])
    avg_psnr = sum(psnr_list[:, 0]) / len(psnr_list[:, 0])
    avg_ssim = sum(psnr_list[:, 1]) / len(psnr_list[:, 1])

    return avg_psnr, avg_ssim


val_sample_size = 200  # max 7824
vimeo_data_dir = 'D:/Jade/Downloads/Compressed/vimeo_septuplet/sequences/'


def get_head(data_dir=vimeo_data_dir, input_list_txt='./data/sep_valtestlist.txt', size=val_sample_size):
    T_in = 7
    with open(input_list_txt, 'r') as f_in:
        val_path = []
        for i in range(size):  # 每个样本
            f1, f2 = f_in.readline().strip().split('/')
            imgs_dir = data_dir + f1 + '/' + f2 + '/'
            img_paths = [imgs_dir + n for n in os.listdir(imgs_dir)]
            img_paths.sort()
            val_path.append(img_paths[T_in // 2])

    return val_path


if __name__ == '__main__':
    # hr_dir = 'D:/SR/Vid4/'
    # dirs = []
    # for file in os.listdir(hr_dir):
    #     path = hr_dir + file + '/'
    #     if os.path.isdir(path):
    #         dirs.append(path)
    # dirs.sort()
    # count = 0
    # for d in dirs:
    #     imgs = [d + i for i in os.listdir(d)]
    #     imgs.sort()
    #     for i in imgs:
    #         cv.imwrite('./data/test_img_hr/{:03d}.png'.format(count), cv.imread(i))
    #         count += 1

    hr_dir = './data/test_img_hr/'
    # sr_dir = './data/test_img_hr_bicubic_hr_x4/'  # avg_psnr and avg_ssim: (20.954291255509087, 0.48635316848392185)

    # sr_dir = './data/result_c/'  # avg_psnr and avg_ssim: (23.819510930140105, 0.5739594427201125)
    # sr_dir = './data/resulttestsame/'  # avg_psnr and avg_ssim: (24.166527827529976, 0.6137785953275403)
    sr_dir = './data/srcnn/'  # avg_psnr and avg_ssim: (28.264967406647745, 0.8463306356011534)
    print('avg_psnr and avg_ssim:', main(hr_dir, sr_dir), 'for {} and {}.'.format(hr_dir, sr_dir))
    #
    # # hr2bicubic_hr()

    # val_path = get_head()
    # gt_dir = './data/resultvaltest/'  # avg_psnr and avg_ssim: (31.43378400947379, 0.655059474848262)

    # val_path = get_head(input_list_txt='./data/sep_testlist.txt')
    # gt_dir = './data/resultval/'  # avg_psnr and avg_ssim: (30.905987445373793, 0.6794450295404783)
    #
    # gt_files = os.listdir(gt_dir)
    # gt_files = [os.path.join(gt_dir, g).replace('\\', '/') for g in gt_files]
    # gt_files.sort()
    # psnr_list = np.array([PSNR(g, p) for g, p in zip(gt_files, val_path)])
    # avg_psnr = sum(psnr_list[:, 0]) / len(psnr_list[:, 0])
    # avg_ssim = sum(psnr_list[:, 1]) / len(psnr_list[:, 1])
    # print(avg_ssim, avg_psnr)

    # hr2bicubic_lr()

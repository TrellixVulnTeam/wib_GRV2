#!/usr/bin/env python
# coding:utf-8

import cv2


def image_scale(input_image, input_scale_method='min_dim', input_std_dim=256):

    if input_scale_method == 'min_dim':
        [img_h, img_w, img_c] = input_image.shape
        cur_min_dim = min(img_h, img_w)
        cur_scale = input_std_dim / float(cur_min_dim)
    elif input_scale_method == 'max_dim':
        [img_h, img_w, img_c] = input_image.shape
        cur_min_dim = max(img_h, img_w)
        cur_scale = input_std_dim / float(cur_min_dim)

    cur_bgr_image = cv2.resize(input_image, (int(img_w * cur_scale + 0.5), int(img_h * cur_scale + 0.5)))
    return cur_bgr_image
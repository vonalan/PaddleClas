# coding: utf-8

import os, sys 
import cv2 
import numpy as np 
import mxnet as mx 
from collections import namedtuple
import argparse

sys.path.append('../tools/infer')
import utils 


def create_operators():
    size = 224
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    decode_op = utils.DecodeImage()
    resize_op = utils.ResizeImage(resize_short=256)
    crop_op = utils.CropImage(size=(size, size))
    normalize_op = utils.NormalizeImage(
        scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = utils.ToTensor()

    return [decode_op, resize_op, crop_op, normalize_op, totensor_op]


def preprocess(fname, ops):
    data = open(fname, 'rb').read()
    for op in ops:
        data = op(data)

    return data

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_file", type=str)
config = parser.parse_args()

ops = create_operators()
xs = preprocess(config.image_file, ops)
xs = np.expand_dims(xs, axis=0)
print(xs.shape)

sym, arg_params, aux_params = mx.model.load_checkpoint('symbol/ResNet50_vd_10w_pretrained', 0)
mod = mx.mod.Module(context=mx.cpu(), symbol=sym, label_names=[])
mod.bind(data_shapes=[('data', (1, 3, 224, 224))], for_training=False)
mod.set_params(arg_params=arg_params, aux_params=aux_params)
Batch = namedtuple('Batch', ['data'])
mod.forward(Batch([mx.nd.array(xs)]))
es = mod.get_outputs()[0]
es = es.asnumpy() 
print(es.shape, es.min(), es.max(), es.mean())
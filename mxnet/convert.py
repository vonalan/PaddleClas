# coding: utf-8

'''
convert pp model ResNet50_vd_10w_pretrained mxnet: 
    1. save parameters with numpy in ../tools/export_model.py 
    2. drop fc layer and fc op in gluoncv.model_zoo.resnetv1b.py 
    3. convert parameters to gluon and symbol 
    4. check the converted parameters 
    5. recover fc layer and fc op in gluoncv.model_zoo.resnetv1b.py 
'''

import numpy as np 
import mxnet as mx
import gluoncv
from collections import namedtuple

bn_map = {
    'gamma': 'scale', 
    'beta': 'offset', 
    'mean': 'mean', 
    'var': 'variance', 
}

use_pretraind = False
model_name = 'ResNet50_v1d'
net = gluoncv.model_zoo.get_model(model_name, pretrained=use_pretraind)
if not use_pretraind:
    net.initialize() 
    xs = mx.nd.random.uniform(shape=(2, 3, 224, 224))
    ys = net(xs)
    _ = ys.asnumpy()

for k, v in net.collect_params().items():
    if 'dense' in k:
        continue 
    __k = None 

    if 'layer' in k and 'conv' in k:
        _, layer_name, conv_name, _ = k.split('_')
        layer_id = int(layer_name.replace('layers', ''))
        conv_id = int(conv_name.replace('conv', ''))
        __k = 'res{}{}_branch2{}_weights'.format(
            layer_id + 1, 
            chr(ord('a') + (conv_id // 3)), 
            chr(ord('a') + (conv_id % 3)), 
        )

    if  'down' in k and 'conv' in k:
        _, layer_name, conv_name, _ = k.split('_')
        layer_id = int(layer_name.replace('down', ''))
        conv_id = int(conv_name.replace('conv', ''))
        __k = 'res{}{}_branch1_weights'.format(layer_id + 1, chr(ord('a') + (conv_id // 3)))

    if 'layer' in k and 'batchnorm' in k:
        temp = k.split('_')
        layer_name, conv_name = temp[1], temp[2]
        layer_id = int(layer_name.replace('layers', ''))
        conv_id = int(conv_name.replace('batchnorm', ''))
        __k = 'bn{}{}_branch2{}_{}'.format(
            layer_id + 1, 
            chr(ord('a') + (conv_id // 3)), 
            chr(ord('a') + (conv_id % 3)), 
            bn_map[temp[-1]], 
        )

    if 'down' in k and 'batchnorm' in k:
        temp = k.split('_')
        layer_name, conv_name = temp[1], temp[2]
        layer_id = int(layer_name.replace('down', ''))
        conv_id = int(conv_name.replace('batchnorm', ''))
        __k = 'bn{}{}_branch1_{}'.format(
            layer_id + 1, 
            chr(ord('a') + (conv_id // 3)), 
            bn_map[temp[-1]], 
        )

    if 'conv' in k and 'layer' not in k and 'down' not in k:
        temp = k.split('_')
        layer_id = 0
        conv_name = temp[1]
        conv_id = int(conv_name.replace('conv', ''))
        __k = 'conv{}_{}_weights'.format(layer_id + 1, conv_id + 1)

    if 'batchnorm' in k and 'layer' not in k and 'down' not in k:
        temp = k.split('_')
        layer_id = 0
        conv_name = temp[1]
        conv_id = int(conv_name.replace('batchnorm', ''))
        __k = 'bnv{}_{}_{}'.format(layer_id + 1, conv_id + 1, bn_map[temp[-1]])
    
    assert __k is not None 
    print(k)
    old = v.data().asnumpy()
    print('before {:6.4f} {:.4f} {:.4f} {} {}'.format(old.min(), old.max(), old.mean(), old.shape, old.dtype))
    new = np.load('npz/{}.npy'.format(__k))
    print(' after {:6.4f} {:.4f} {:.4f} {} {}'.format(new.min(), new.max(), new.mean(), new.shape, new.dtype))
    v.set_data(mx.nd.array(new))
    new = v.data().asnumpy()
    print(' after {:6.4f} {:.4f} {:.4f} {} {}'.format(new.min(), new.max(), new.mean(), new.shape, new.dtype))

print(len(net.collect_params('^(?!.*?dense).*$').keys()))
net.collect_params('^(?!.*?dense).*$').save('gluon/ResNet50_vd_10w_pretrained.params')

xs = mx.nd.random.uniform(shape=(2, 3, 224, 224))
net.hybridize()
ys = net(xs)
ys = ys.asnumpy()
print(ys.shape, ys.min(), ys.max(), ys.mean())
net.export('symbol/ResNet50_vd_10w_pretrained')

sym, arg_params, aux_params = mx.model.load_checkpoint('symbol/ResNet50_vd_10w_pretrained', 0)
mod = mx.mod.Module(context=mx.cpu(), symbol=sym, label_names=[])
mod.bind(data_shapes=[('data', (2, 3, 224, 224))], for_training=False)
mod.set_params(arg_params=arg_params, aux_params=aux_params)
Batch = namedtuple('Batch', ['data'])
mod.forward(Batch([xs]))
es = mod.get_outputs()[0]
es = es.asnumpy() 
print(es.shape, es.min(), es.max(), es.mean())
# coding: utf-8

import mxnet as mx
import gluoncv
from collections import OrderedDict

# filename = 'classification-demo.png'
# model_name = 'ResNet50_v1d'
# net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
# for k, v in net.collect_params().items():
#     print(k, v.shape)

params_dict_1 = OrderedDict()
with open('log.mx.txt', encoding='utf-16') as ifs:
    data = ifs.read()
    info = data.strip().split('\n')
    info = [item.split(' ', maxsplit=1) for item in info]
    for item in info:
        params_dict_1[item[0]] = item[1]
print(len(params_dict_1))

params_dict_2 = OrderedDict()
with open('log.pp.txt', encoding='utf-16') as ifs:
    data = ifs.read()
    info = data.strip().split('\n')
    info = [item.split(' ', maxsplit=1) for item in info]
    for item in info:
        params_dict_2[item[0]] = item[1]
print(len(params_dict_2))

bn_map = {
    'gamma': 'scale', 
    'beta': 'offset', 
    'mean': 'mean', 
    'var': 'variance', 
}

__params_dict1 = set()
__params_dict2 = set()
for k, v in params_dict_1.items():
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
    
    if __k is None:
        continue
    assert params_dict_1[k] == params_dict_2[__k]
    __params_dict1.add(k)
    __params_dict2.add(__k)

print('-------------------------->')
for k in set(params_dict_1.keys()).difference(__params_dict1):
    print(k, params_dict_1[k])
print('-------------------------->')
for k in set(params_dict_2.keys()).difference(__params_dict2):
    print(k)
print('-------------------------->')
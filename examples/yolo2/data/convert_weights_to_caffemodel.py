#!/usr/bin/env python

from __future__ import print_function, division
import argparse
import caffe
import numpy as np


def convert(deploy, weights, model):
    caffe.set_mode_cpu()
    net = caffe.Net(deploy, caffe.TEST)
    net_weights_float = np.fromfile(weights, dtype=np.float32)
    net_weights = net_weights_float[4:]
    # start from the 5th entry, the first 4 entries are major, minor, revision and net.seen
    print(net_weights.shape)
    count = 0
    batch_norm = None
    conv_bias = None
    for pr in net.params.keys():
        lidx = list(net._layer_names).index(pr)
        layer = net.layers[lidx]
        if count == net_weights.shape[0] and batch_norm is None:
            print("WARNING: no weights left for %s" % pr)
            break
        if layer.type == 'Convolution':
            print(pr + "(conv)")
            # bias
            if len(net.params[pr]) > 1:
                bias_dim = net.params[pr][1].data.shape
            else:
                bias_dim = (net.params[pr][0].data.shape[0],)
            bias_size = np.prod(bias_dim)
            conv_bias = np.reshape(net_weights[count:count + bias_size], bias_dim)
            if len(net.params[pr]) > 1:
                assert (bias_dim == net.params[pr][1].data.shape)
                net.params[pr][1].data[...] = conv_bias
                conv_bias = None
            count += bias_size
            # batch_norm
            next_layer = net.layers[lidx + 1]
            if next_layer.type == 'BatchNorm':
                bn_dims = (3, net.params[pr][0].data.shape[0])
                bn_size = np.prod(bn_dims)
                batch_norm = np.reshape(net_weights[count:count + bn_size], bn_dims)
                count += bn_size
            # weights
            dims = net.params[pr][0].data.shape
            weight_size = np.prod(dims)
            net.params[pr][0].data[...] = np.reshape(net_weights[count:count + weight_size], dims)
            count += weight_size
        elif layer.type == 'BatchNorm':
            print(pr + "(batchnorm)")
            net.params[pr][0].data[...] = batch_norm[1]  # mean
            net.params[pr][1].data[...] = batch_norm[2]  # variance
            net.params[pr][2].data[...] = 1.0  # scale factor
        elif layer.type == 'Scale':
            print(pr + "(scale)")
            net.params[pr][0].data[...] = batch_norm[0]  # scale
            batch_norm = None
            if len(net.params[pr]) > 1:
                net.params[pr][1].data[...] = conv_bias  # bias
                conv_bias = None
        else:
            print("WARNING: unsupported layer, " + pr)
    if np.prod(net_weights.shape) != count:
        print("ERROR: size mismatch: %d" % count)
        print("ERROR: size mismatch: %d" % net_weights.shape)
    else:
        net.save(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert')
    parser.add_argument('-d', '--deploy', action='store', dest='deploy',
                        required=True, help='deploy prototxt')
    parser.add_argument('-w', '--weights', action='store', dest='weights',
                        required=True, help='weights')
    parser.add_argument('-m', '--model', action='store', dest='model',
                        required=False, default='yolo.caffemodel', help='caffemodel')
    args = parser.parse_args()

    convert(args.deploy, args.weights, args.model)

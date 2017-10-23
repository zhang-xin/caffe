#!/usr/bin/env python

from __future__ import print_function, division
import argparse
import numpy as np
import cv2
import caffe
import math

voc_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'chair', 'cow', 'diningtable',
             'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def sigmoid(p):
    return 1.0 / (1 + math.exp(-p * 1.0))


def overlap(x1, w1, x2, w2):
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left


def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    return inter_area * 1.0 / union_area


def apply_nms(boxes, thres):
    sorted_boxes = sorted(boxes, key=lambda d: d[7])[::-1]
    p = dict()
    for i in range(len(sorted_boxes)):
        if i in p:
            continue

        truth = sorted_boxes[i]
        for j in range(i+1, len(sorted_boxes)):
            if j in p:
                continue
            box = sorted_boxes[j]
            iou = cal_iou(box, truth)
            if iou >= thres:
                p[j] = 1

    res = list()
    for i in range(len(sorted_boxes)):
        if i not in p:
            res.append(sorted_boxes[i])
    return res


def letterbox_image(im, w, h):
    if (w / im.shape[1]) < (h / im.shape[0]):
        new_w = w
        new_h = int((im.shape[0] * w) / im.shape[1])
    else:
        new_h = h
        new_w = int((im.shape[1] * h) / im.shape[0])
    resized = cv2.resize(im, (new_w, new_h))
    boxed = np.ones((h, w, im.shape[2]), dtype=np.float32)
    boxed *= 0.5
    boxed[int((h-new_h)/2):int((h-new_h)/2)+resized.shape[0], int((w-new_w)/2):int((w-new_w)/2)+resized.shape[1]] = resized
    return boxed


def det(pic):
    net.blobs['data'].reshape(1, 3, 416, 416)

    image = caffe.io.load_image(pic)
    orig_width = image.shape[1]
    orig_height = image.shape[0]
    lb_image = letterbox_image(image, 416, 416)
    lb_image = lb_image.transpose((2, 0, 1))
    net.blobs['data'].data[...] = lb_image

    output = net.forward()

    im = cv2.imread(pic)

    boxes = output['box']
    prob = output['prob']
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[1]):
            for k in range(boxes.shape[2]):
                boxes[i][j][k][0] = (boxes[i][j][k][0] - (orig_width - 416)) / 2 / orig_width / (416 / orig_width)
                boxes[i][j][k][1] = (boxes[i][j][k][1] - (orig_height - 416)) / 2 / orig_height / (416 / orig_height)
                boxes[i][j][k][2] *= orig_width / 416
                boxes[i][j][k][3] *= orig_height / 416

    boxes = apply_nms(boxes, 0.3)

    for box in boxes:
        left = (box[0] - box[2] / 2.0) * orig_width
        right = (box[0] + box[2] / 2.0) * orig_width
        top = (box[1] - box[3] / 2.0) * orig_height
        bot = (box[1] + box[3] / 2.0) * orig_height
        if left < 0:
            left = 0
        if right > orig_width:
            right = orig_width
        if top < 0:
            top = 0
        if bot > orig_height:
            bot = orig_height
        color = (255, 242, 35)
        cv2.putText(im, voc_names[box[4]-1], (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.rectangle(im, (int(left), int(top)), (int(right), int(bot)), color, 2)

    cv2.imshow('src', im)
    cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test yolo v2')
    parser.add_argument('-d', '--deploy', action='store', dest='deploy',
                        required=True, help='deploy prototxt')
    parser.add_argument('-m', '--model', action='store', dest='model',
                        required=True, help='caffemodel')
    parser.add_argument('pic_path', help='input test picture')
    args = parser.parse_args()

    caffe.set_mode_cpu()
    net = caffe.Net(args.deploy, args.model, caffe.TEST)

    det(args.pic_path)


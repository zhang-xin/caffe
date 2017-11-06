#!/usr/bin/env python

from __future__ import print_function, division
import sys

import lmdb
import caffe
import cv2
import numpy as np


env = lmdb.open(sys.argv[1], readonly=True)
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)

        data = datum.data
        fdata = datum.float_data

        image = cv2.imdecode(np.fromstring(data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        objs = int(len(fdata) / 6)
        for i in range(objs):
            w = int(fdata[i*6+4] * image.shape[1])
            h = int(fdata[i*6+5] * image.shape[0])
            x = int(fdata[i*6+2] * image.shape[1] - w / 2)
            y = int(fdata[i*6+3] * image.shape[0] - h / 2)
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255))
        cv2.imshow('image', image)
        if cv2.waitKey(0) == ord('q'):
            break

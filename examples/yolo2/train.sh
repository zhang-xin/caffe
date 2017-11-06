#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=./solver.prototxt
WEIGHTS=./yolo-voc.caffemodel
$CAFFE_HOME/build/tools/caffe train --solver=$SOLVER --weights=$WEIGHTS

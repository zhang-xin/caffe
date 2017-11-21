#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=./solver.prototxt
WEIGHTS=./tiny-yolo-conv1-7.caffemodel
$CAFFE_HOME/build/tools/caffe train --solver=$SOLVER --weights=$WEIGHTS

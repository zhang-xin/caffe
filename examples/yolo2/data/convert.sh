#!/usr/bin/env sh

CAFFE_ROOT=../..
DATA_DIR=/home/xzhang84/projects/voc_data/
LABEL_FILE=$CAFFE_ROOT/examples/yolo2/data/label_map.txt

# 2007 + 2012 trainval
LIST_FILE=$DATA_DIR/trainval.txt
LMDB_DIR=$CAFFE_ROOT/examples/yolo2/trainval_lmdb
SHUFFLE=true

RESIZE_W=416
RESIZE_H=416

$CAFFE_ROOT/build/tools/convert_box_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
  --label_file=$LABEL_FILE $DATA_DIR $LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

# 2007 test
LIST_FILE=$DATA_DIR/test_2007.txt
LMDB_DIR=$CAFFE_ROOT/examples/yolo2/test2007_lmdb
SHUFFLE=true

$CAFFE_ROOT/build/tools/convert_box_data --resize_width=$RESIZE_W --resize_height=$RESIZE_H \
  --label_file=$LABEL_FILE $DATA_DIR $LIST_FILE $LMDB_DIR --encoded=true --encode_type=jpg --shuffle=$SHUFFLE

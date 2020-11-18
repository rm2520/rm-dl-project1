#!/usr/bin/env bash

CAFFE=caffe
DATASET=CUHK03
NET=CaffeNet
SNAPSHOTS_DIR=output/${DATASET}_train

LOG="experiments/logs/${DATASET}_re-id_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"

cd $(dirname ${BASH_SOURCE[0]})/../../

mkdir -p ${SNAPSHOTS_DIR}
mkdir -p experiments/logs/

GLOG_logtostderr=1 ${CAFFE}/build/tools/caffe train \
  -solver models/${DATASET}/${NET}/${NET}_labeled_solver.prototxt \
  -weights data/imagenet_models/${NET}.v2.caffemodel  2>&1 | tee ${LOG}

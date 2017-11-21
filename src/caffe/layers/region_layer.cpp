#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include "caffe/layer.hpp"
#include "caffe/layers/region_layer.hpp"
#include "caffe/util/region_util.hpp"

namespace caffe {

template <typename Dtype>
void RegionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  output_.ReshapeLike(*bottom[0]);

  // reshpe top blob
  vector<int> box_shape(4);
  box_shape[0] = num_;
  box_shape[1] = height_;
  box_shape[2] = width_;
  box_shape[3] = coords_;
  top[0]->Reshape(box_shape);

  vector<int> prob_shape(4);
  prob_shape[0] = num_;
  prob_shape[1] = height_;
  prob_shape[2] = width_;
  prob_shape[3] = num_class_ + 1;
  top[1]->Reshape(prob_shape);
}

template <typename Dtype>
void RegionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  RegionLossParameter param = this->layer_param_.region_loss_param();

  height_     = param.side();
  width_      = param.side();
  num_class_  = param.num_classes();
  coords_     = param.coords();
  num_        = param.num();
  softmax_    = param.softmax();
  batch_      = param.batch();
  thresh_     = param.threshold();

  for (int c = 0; c < param.biases_size(); ++c) {
    biases_.push_back(param.biases(c));
  }

  int input_count = bottom[0]->count(1);
  int tmp_input_count = width_ * height_ * num_ * (coords_ + num_class_ + 1);
  CHECK_EQ(input_count, tmp_input_count);
}

template <typename Dtype>
void RegionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  Dtype* box_data = top[0]->mutable_cpu_data();
  Dtype* prob_data = top[1]->mutable_cpu_data();

  Dtype* output = output_.mutable_cpu_data();
  caffe_copy(output_.count(), input_data, output);
  forward_softmax<Dtype>(input_data, output, batch_, num_, width_, height_, coords_, num_class_, softmax_);

  get_region_boxes<Dtype>(output, box_data, prob_data, num_, width_, height_, coords_, num_class_,
                          thresh_, biases_);
}

INSTANTIATE_CLASS(RegionLayer);
REGISTER_LAYER_CLASS(Region);

}  // namespace caffe

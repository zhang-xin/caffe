#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include "caffe/layer.hpp"
#include "caffe/layers/region_layer.hpp"

namespace caffe {

template <typename Dtype>
void RegionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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
inline void softmax_op(Dtype* input, int classes, int stride) {
  Dtype sum = 0;
  Dtype large = input[0];
  for (int i = 0; i < classes; ++i) {
    if (input[i * stride] > large)
      large = input[i * stride];
  }
  for (int i = 0; i < classes; ++i) {
      Dtype e = exp(input[i * stride] - large);
      sum += e;
      input[i * stride] = e;
  }
  for (int i = 0; i < classes; ++i) {
      input[i * stride] /= sum;
  }
}

template <typename Dtype>
inline void softmax_cpu(Dtype *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride) {
  for (int b = 0; b < batch; ++b) {
    for (int g = 0; g < groups; ++g) {
      softmax_op(input + b*batch_offset + g*group_offset, n, stride);
    }
  }
}

template <typename Dtype>
vector<Dtype> get_region_box(Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int w, int h, int stride) {
  vector<Dtype> b;
  b.push_back((i + x[index + 0 * stride]) / w);
  b.push_back((j + x[index + 1 * stride]) / h);
  b.push_back(exp(x[index + 2 * stride]) * biases[2 * n] / w);
  b.push_back(exp(x[index + 3 * stride]) * biases[2 * n + 1] / h);
  return b;
}

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void RegionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Layer<Dtype>::LayerSetUp(bottom, top);
  RegionLossParameter param = this->layer_param_.region_loss_param();
  
  height_ = param.side();
  width_ =  param.side();
  bias_match_ = param.bias_match();
  num_class_ = param.num_classes();
  coords_ = param.coords();
  num_ = param.num();
  softmax_ = param.softmax();
  batch_ = param.batch();
  jitter_ = param.jitter(); 
  rescore_ = param.rescore();
  
  for (int c = 0; c < param.biases_size(); ++c) {
	  biases_.push_back(param.biases(c));
  }
  
  absolute_ = param.absolute();
  thresh_ = param.threshold();
  random_ = param.random();

  int input_count = bottom[0]->count(1);
  // outputs: classes, iou, coordinates
  int tmp_input_count = width_ * height_ * num_ * (coords_ + num_class_ + 1);
  CHECK_EQ(input_count, tmp_input_count);
}

template <typename Dtype>
void RegionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Dtype* input_data = bottom[0]->mutable_cpu_data();
  Dtype* box_data = top[0]->mutable_cpu_data();
  Dtype* prob_data = top[1]->mutable_cpu_data();
  for (int b = 0; b < batch_; b++) {
    for (int n = 0; n < num_; n++) {
      int index = entry_index(b, n * width_ * height_, 0);
      for (int k = 0; k < 2 * width_ * height_; k++) {
        input_data[index + k] = sigmoid(input_data[index + k]);
      }
      index = entry_index(b, n * width_ * height_, coords_);
      for (int k = 0; k < width_ * height_; k++) {
        input_data[index + k] = sigmoid(input_data[index + k]);
      }
    }
  }
  if (softmax_) {
    int index = entry_index(0, 0, coords_ + 1);
    softmax_cpu(input_data + index, num_class_, batch_ * num_, height_ * width_ * (num_class_ + coords_ + 1),
                width_ * height_, 1, width_ * height_);
  }

  for (int i = 0; i < width_ * height_; ++i) {
	int row = i / width_;
	int col = i % width_;
	for (int n = 0; n < num_; ++n) {
      int index = n * width_ * height_ + i;
	  for (int j = 0; j < num_class_ + 1; ++j) {
        prob_data[index * (num_class_ + 1) + j] = 0;
      }
      int obj_index = entry_index(0, n * width_ * height_ + i, coords_);
      int box_index = entry_index(0, n * width_ * height_ + i, 0);
      float scale = input_data[obj_index];
      vector<Dtype> box = get_region_box(input_data, biases_, n, box_index, col, row, width_, height_,
                                         width_ * height_);
      for (int k = 0; k < box.size(); k++)
        box_data[index * coords_ + k] = box[k];
      float max_prob = 0;
      for (int j = 0; j < num_class_; ++j) {
        int class_index = entry_index(0, n * width_ * height_ + i, coords_ + 1 + j);
        float prob = scale * input_data[class_index];
        prob_data[index * (num_class_ + 1) + j] = (prob > thresh_) ? prob : 0;
	    if (prob > max_prob) max_prob = prob;
      }
      prob_data[index * (num_class_ + 1) + num_class_] = (max_prob > thresh_) ? max_prob : 0;
    }
  }
}

INSTANTIATE_CLASS(RegionLayer);
REGISTER_LAYER_CLASS(Region);

}  // namespace caffe

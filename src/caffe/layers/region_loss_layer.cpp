#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/region_util.hpp"
#include "caffe/util/bbox_util.hpp"

int iter = 0;

namespace caffe {

template <typename Dtype>
Dtype delta_region_box(vector<Dtype> truth, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j,
                       int w, int h, Dtype* delta, float scale, int stride) {
  vector<Dtype> pred;
  pred = get_region_box(x, biases, n, index, i, j, w, h, stride);

  float iou = box_iou(pred, truth);
  float tx = truth[0] * w - i;
  float ty = truth[1] * h - j;
  float tw = log(truth[2] * w / biases[2 * n]);
  float th = log(truth[3] * h / biases[2 * n + 1]);

  delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
  delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
  delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
  delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
  return iou;
}

template <typename Dtype>
void delta_region_class(Dtype* data, Dtype* &diff, int index, int class_label, int classes,
                        string softmax_tree, tree *t, float scale, int stride, Dtype* avg_cat, bool tag) {
  if (softmax_tree != "") {
  } else {
    if (diff[index] && tag) {
      diff[index + stride * class_label] = scale * (1 - data[index + stride * class_label]);
      return;
    }
    for (int n = 0; n < classes; ++n) {
      diff[index + stride * n] = scale * (((n == class_label) ? 1 : 0) - data[index + stride * n]);
      if (n == class_label) {
        *avg_cat += data[index + stride * n];
      }
    }
  }
}

template <typename Dtype>
Dtype get_hierarchy_prob(Dtype* data, tree *t, int c, int stride) {
  float p = 1;
  while (c >= 0) {
    p = p * data[c * stride];
    c = t->parent[c];
  }
  return p;
}

template <typename Dtype>
void gradient_array(const Dtype* x, const int n, Dtype* delta)
{
  for (int i = 0; i < n; i++)
    delta[i] *= x[i] * (1 - x[i]);
}

template <typename Dtype>
void RegionLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  RegionLossParameter param = this->layer_param_.region_loss_param();

  height_     = param.side();
  width_      = param.side();
  num_class_  = param.num_classes();
  coords_     = param.coords();
  num_        = param.num();
  softmax_    = param.softmax();
  batch_      = param.batch();
  thresh_     = param.threshold();

  bias_match_ = param.bias_match();
  jitter_     = param.jitter();
  rescore_    = param.rescore();
  absolute_   = param.absolute();
  random_     = param.random();

  for (int c = 0; c < param.biases_size(); ++c) {
    biases_.push_back(param.biases(c));
  }

  object_scale_   = param.object_scale();
  noobject_scale_ = param.noobject_scale();
  class_scale_    = param.class_scale();
  coord_scale_    = param.coord_scale();

  softmax_tree_ = param.softmax_tree();
  if (softmax_tree_ != "")
    t_ = tree(softmax_tree_);

  class_map_ = param.class_map();
  if (class_map_ != "") {
    string line;
    std::fstream fin(class_map_.c_str());
    if (!fin){
      LOG(INFO) << "no map file";
    }

    int index = 0;
    int id = 0;
    while (getline(fin, line)) {
      stringstream ss;
      ss << line;
      ss >> id;

      cls_map_[index] = id;
      index ++;
    }
    fin.close();
  }

  int label_count = bottom[1]->count(1);
  int tmp_label_count = 30 * num_;
  CHECK_EQ(label_count, tmp_label_count);
  int input_count = bottom[0]->count(1);
  int tmp_input_count = width_ * height_ * num_ * (coords_ + num_class_ + 1);
  CHECK_EQ(input_count, tmp_input_count);
}


template <typename Dtype>
void RegionLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  delta_.ReshapeLike(*bottom[0]);
  output_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  batch_ = bottom[0]->num();

  Dtype* output = output_.mutable_cpu_data();
  caffe_copy(output_.count(), input_data, output);
  forward_softmax<Dtype>(input_data, output, batch_, num_, width_, height_, coords_, num_class_, softmax_);

  Dtype* delta = delta_.mutable_cpu_data();
  caffe_set(delta_.count(), Dtype(0.0), delta);
  Dtype avg_anyobj(0.0), avg_obj(0.0), avg_iou(0.0), avg_cat(0.0), recall(0.0), loss(0.0);
  int count = 0;
  int class_count = 0;
  for (int b = 0; b < batch_; b++) {
    for (int j = 0; j < height_; j++) {
      for (int i = 0; i < width_; i++) {
        for (int n = 0; n < num_; n++) {
          int box_index = entry_index(b, n * width_ * height_ + j * width_ + i, 0, num_, width_, height_, coords_, num_class_);
          vector<Dtype> pred = get_region_box(output, biases_, n, box_index, i, j, width_, height_, width_ * height_);
          float best_iou = 0;
          for(int t = 0; t < 30; ++t) {
            vector<Dtype> truth;
            Dtype x = label_data[b * 30 * (coords_ + 1) + t * (coords_ + 1) + 1];
            Dtype y = label_data[b * 30 * (coords_ + 1) + t * (coords_ + 1) + 2];
            Dtype w = label_data[b * 30 * (coords_ + 1) + t * (coords_ + 1) + 3];
            Dtype h = label_data[b * 30 * (coords_ + 1) + t * (coords_ + 1) + 4];

            if (x == 0.0f) break;
            truth.push_back(x);
            truth.push_back(y);
            truth.push_back(w);
            truth.push_back(h);
            Dtype iou = box_iou(pred, truth);
            if (iou > best_iou) {
              best_iou = iou;
            }
          }
          int obj_index = entry_index(b, n * width_ * height_ + j * width_ + i, coords_, num_, width_, height_, coords_, num_class_);
          avg_anyobj += output[obj_index];
          delta[obj_index] = noobject_scale_ * (0 - output[obj_index]);
          if (best_iou > thresh_) {
            delta[obj_index] = 0;
          }
          /*
          if (iter < 12800 / batch_) {
            vector<Dtype> truth;
            truth.push_back((i + .5) / width_);
            truth.push_back((j + .5) / height_);
            truth.push_back((biases_[2 * n]) / width_);
            truth.push_back((biases_[2 * n + 1]) / height_);
            delta_region_box(truth, output, biases_, n, box_index, i, j, width_, height_, delta, .01f, width_ * height_);
          }
          */
        }
      }
    }

    for (int t = 0; t < 30; t++) {
      vector<Dtype> truth;
      int class_label = label_data[b * 30 * (coords_ + 1) + t * (coords_ + 1) + 0];
      Dtype x = label_data[b * 30 * (coords_ + 1) + t * (coords_ + 1) + 1];
      Dtype y = label_data[b * 30 * (coords_ + 1) + t * (coords_ + 1) + 2];
      Dtype w = label_data[b * 30 * (coords_ + 1) + t * (coords_ + 1) + 3];
      Dtype h = label_data[b * 30 * (coords_ + 1) + t * (coords_ + 1) + 4];
      if (x == 0.0f) break;
      truth.push_back(x);
      truth.push_back(y);
      truth.push_back(w);
      truth.push_back(h);

      float best_iou = 0;
      int best_n = 0;
      int i = truth[0] * width_;
      int j = truth[1] * height_;
      vector<Dtype> truth_shift;
      truth_shift.push_back(0);
      truth_shift.push_back(0);
      truth_shift.push_back(w);
      truth_shift.push_back(h);

      for (int n = 0; n < num_; n++) {
        int box_index = entry_index(b, n * width_ * height_ + j * width_ + i, 0,
                                    num_, width_, height_, coords_, num_class_);
        vector<Dtype> pred = get_region_box(output, biases_, n, box_index, i, j, width_, height_, width_ * height_);
        if (bias_match_) {
          pred[2] = biases_[2 * n] / width_;
          pred[3] = biases_[2 * n + 1] / height_;
        }
        pred[0] = 0;
        pred[1] = 0;
        float iou = box_iou(pred, truth_shift);
        if (iou > best_iou) {
          best_iou = iou;
          best_n = n;
        }
      }

      int box_index = entry_index(b, best_n * width_ * height_ + j * width_ + i, 0,
                                  num_, width_, height_, coords_, num_class_);
      float iou = delta_region_box(truth, output, biases_, best_n, box_index, i, j, width_, height_, delta,
                                   coord_scale_ * (2 - truth[2] * truth[3]), width_ * height_);
      if (iou > 0.5) recall += 1;
      avg_iou += iou;

      int obj_index = entry_index(b, best_n * width_ * height_ + j * width_ + i, coords_,
                                  num_, width_, height_, coords_, num_class_);
      avg_obj += output[obj_index];
      delta[obj_index] = object_scale_ * (1 - output[obj_index]);
      if (rescore_)
        delta[obj_index] = object_scale_ * (iou - output[obj_index]);

      if (class_map_ != "") class_label = cls_map_[class_label];
      int class_index = entry_index(b, best_n * width_ * height_ + j * width_ + i, coords_ + 1,
                                    num_, width_, height_, coords_, num_class_);
      delta_region_class(output, delta, class_index, class_label, num_class_, softmax_tree_, &t_, class_scale_, width_ * height_, &avg_cat, !softmax_);

      ++count;
      ++class_count;
    }
  }

  for (int i = 0; i < delta_.count(); ++i) {
    loss += delta[i] * delta[i];
  }
  top[0]->mutable_cpu_data()[0] = loss;
  iter++;
  if (!(iter % 100)) {
    LOG(INFO) << "avg_noobj: "<< avg_anyobj / (width_ * height_ * num_ * batch_)
              << " avg_obj: " << avg_obj / count
              <<" avg_iou: " << avg_iou / count
              << " avg_cat: " << avg_cat / class_count
              << " recall: " << recall / count
              << " class_count: "<< class_count;
  }
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    /*
    for (int b = 0; b < bottom[0]->num(); b++) {
      for (int n = 0; n < num_; n++) {
        int index = entry_index(b, n * width_ * height_, 0, num_, width_, height_, coords_, num_class_);
        gradient_array(output_.cpu_data() + index, 2 * width_ * height_, delta_.mutable_cpu_data() + index);
        index = entry_index(b, n * width_ * height_, coords_, num_, width_, height_, coords_, num_class_);
        gradient_array(output_.cpu_data() + index, width_ * height_, delta_.mutable_cpu_data() + index);
      }
    }
    */
    const Dtype alpha = -1 * top[0]->cpu_diff()[0];
    caffe_cpu_axpby(bottom[0]->count(), alpha, delta_.cpu_data(), Dtype(0), bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(RegionLossLayer);
#endif

INSTANTIATE_CLASS(RegionLossLayer);
REGISTER_LAYER_CLASS(RegionLoss);

}  // namespace caffe

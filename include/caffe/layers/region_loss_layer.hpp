#ifndef CAFFE_REGION_LOSS_LAYER_HPP_
#define CAFFE_REGION_LOSS_LAYER_HPP_

#include <vector>
#include <string>
#include <map>
#include "caffe/util/tree.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class RegionLossLayer : public LossLayer<Dtype> {
 public:
  explicit RegionLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), delta_(), output_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RegionLoss"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int height_;
  int width_;
  int num_class_;
  int num_;
  int coords_;
  bool softmax_;
  int batch_;
  float thresh_;
  vector<Dtype> biases_;

  int bias_match_;
  float jitter_;
  int rescore_;
  int absolute_;
  int random_;

  float object_scale_;
  float class_scale_;
  float noobject_scale_;
  float coord_scale_;

  Blob<Dtype> delta_;
  Blob<Dtype> output_;

  string softmax_tree_;
  tree t_;
  string class_map_;
  map<int, int> cls_map_;
};

}  // namespace caffe

#endif  // CAFFE_REGION_LOSS_LAYER_HPP_

#ifndef CAFFE_REGION_LAYER_HPP_
#define CAFFE_REGION_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class RegionLayer : public Layer<Dtype> {
 public:
  explicit RegionLayer(const LayerParameter& param)
	  : Layer<Dtype>(param), output_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Region"; }

  int height_;
  int width_;
  int num_class_;
  int coords_;
  int num_;
  bool softmax_;
  int batch_;
  float thresh_;
  vector<Dtype> biases_;
  Blob<Dtype> output_;


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }
};

}  // namespace caffe

#endif  // CAFFE_REGION_LAYER_HPP_

#ifndef CAFFE_REGION_LOSS_LAYER_HPP_
#define CAFFE_REGION_LOSS_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class RegionLayer : public Layer<Dtype> {
 public:
  explicit RegionLayer(const LayerParameter& param)
	  : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Region"; }

  int height_;
  int width_;
  int bias_match_;
  int num_class_;
  int coords_;
  int num_;
  int softmax_;
  int batch_;
  float jitter_;
  int rescore_;
  int absolute_;
  float thresh_;
  int random_;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
	  for (int i = 0; i < propagate_down.size(); ++i) {
		  if (propagate_down[i]) { NOT_IMPLEMENTED; }
	  }
  }

  vector<Dtype> biases_;

  inline int entry_index(int batch, int location, int entry) {
    int n = location / (width_ * height_);
    int loc = location % (width_ * height_);
    int outputs = height_ * width_ * num_ * (num_class_ + coords_ + 1);
    return batch * outputs + n * outputs / num_ + entry * width_ * height_ + loc;
  }
};

}  // namespace caffe

#endif  // CAFFE_REGION_LOSS_LAYER_HPP_

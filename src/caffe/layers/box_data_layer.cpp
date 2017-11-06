#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/box_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
BoxDataLayer<Dtype>::BoxDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    offset_() {
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
BoxDataLayer<Dtype>::~BoxDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void BoxDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->box_label_ = true;
  const DataParameter param = this->layer_param_.data_param();
  const int batch_size = param.batch_size();
  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->multi_label_.clear();
    }
    vector<int> label_shape(1, batch_size);
    int label_size = 30 * 5; // (coords + 1) for region loss layer
    label_shape.push_back(label_size);
    top[1]->Reshape(label_shape);
    for (int j = 0; j < this->prefetch_.size(); ++j) {
      shared_ptr<Blob<Dtype> > tmp_blob;
      tmp_blob.reset(new Blob<Dtype>(label_shape));
      this->prefetch_[j]->multi_label_.push_back(tmp_blob);
    }
  }
}

template <typename Dtype>
bool BoxDataLayer<Dtype>::Skip() {
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template<typename Dtype>
void BoxDataLayer<Dtype>::Next() {
  cursor_->Next();
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
    << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template<typename Dtype>
void BoxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  Datum datum;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }

    vector<BoxLabel> box_labels;

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    // Copy label.
    if (this->output_labels_) {
      // rand sample a patch, adjust box labels
      this->data_transformer_->Transform(datum, &(this->transformed_data_), &box_labels);
      // transform label
      Dtype* top_label = batch->multi_label_[0]->mutable_cpu_data();
      int label_offset = batch->multi_label_[0]->offset(item_id);
      int count = batch->multi_label_[0]->count(1);
      transform_label(count, top_label + label_offset, box_labels);
    } else {
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    trans_time += timer.MicroSeconds();
    Next();
  }
  timer.Stop();
  batch_timer.Stop();
  //DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
void BoxDataLayer<Dtype>::transform_label(int count, Dtype* top_label,
    const vector<BoxLabel>& box_labels) {
  CHECK_EQ(count, 30*5) << "side and count not match";
  caffe_set(30*5, Dtype(0), top_label);
  int index = 0;
  for (int i = 0; i < box_labels.size(); ++i) {
    float difficult = box_labels[i].difficult_;
    if (difficult != 0. && difficult != 1.) {
      LOG(WARNING) << "Difficult must be 0 or 1";
    }
    float class_label = box_labels[i].class_label_;
    CHECK_GE(class_label, 0) << "class_label must >= 0";
    top_label[index++] = class_label;
    for (int j = 0; j < 4; ++j) {
	  top_label[index + j] = box_labels[i].box_[j];
    }
    index += 4;
  }
}

INSTANTIATE_CLASS(BoxDataLayer);
REGISTER_LAYER_CLASS(BoxData);

}  // namespace caffe

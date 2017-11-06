#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/layers/region_layer.hpp"
#include "caffe/util/bbox_util.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/db.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::string;
using caffe::vector;
using std::map;
using std::pair;
using namespace std;
using namespace cv;

DEFINE_string(input, "", "Input image for run dection");
DEFINE_string(model, "", "The model definition protocol buffer text file.");
DEFINE_string(label, "", "The label file.");
DEFINE_string(weights, "", "The pretrained weights to initialize finetuning.");
DEFINE_double(nms, 0.30, "The thresh of nms.");

static int num_class = 0;

void preprocess_image(Net<float>& net, Mat &orig_image, int width, int height, int roi_width, int roi_height)
{
  Mat resized, resized_float;

  if (roi_width != orig_image.cols || roi_height != orig_image.rows) {
    resize(orig_image, resized, Size(roi_width, roi_height));
  } else {
    resized = orig_image;
  }
  resized.convertTo(resized_float, CV_32FC3);

  // letterbox image
  Mat boxed = cv::Mat::ones(height, width, CV_32FC3);
  boxed.setTo(Scalar(0.5, 0.5, 0.5));
  resized_float.copyTo(boxed(Rect((width - roi_width) / 2, (height - roi_height) / 2, roi_width, roi_height)));

  Blob<float>* input_layer = net.input_blobs()[0];
  int num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
  Size input_geometry_ = Size(input_layer->width(), input_layer->height());
  input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net.Reshape();

  vector<Mat> input_channels;
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    Mat channel(height, width, CV_32FC1, input_data);
    input_channels.push_back(channel);
    input_data += input_layer->width() * input_layer->height();
  }

  split(boxed, input_channels);
  for (int i = 0; i < input_layer->channels(); ++i)
    normalize(input_channels[i], input_channels[i], 1.0, 0.0, NORM_MINMAX);

  CHECK(reinterpret_cast<float*>(input_channels.at(0).data) == net.input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

typedef struct {
  int index_;
  int class_;
  float *probs_;
} sortable_bbox;

int nms_comparator(const void *pa, const void *pb)
{
  sortable_bbox a = *(sortable_bbox *)pa;
  sortable_bbox b = *(sortable_bbox *)pb;
  float diff = a.probs_[a.index_ * (num_class + 1) + a.class_] - b.probs_[b.index_ * (num_class + 1) + b.class_];
  if (diff < 0) return 1;
  else if (diff > 0) return -1;
  return 0;
}

void do_nms_sort(float *box_data, float *prob_data, int total_size, int num_classes) {
  sortable_bbox *s = new sortable_bbox[total_size];

  for (int i = 0; i < total_size; ++i) {
    s[i].index_ = i;
    s[i].class_ = 0;
    s[i].probs_ = prob_data;
  }

  for (int c = 0; c < num_classes; c++) {
    for (int i = 0; i < total_size; i++)
      s[i].class_ = c;

    qsort(s, (size_t)total_size, sizeof(sortable_bbox), nms_comparator);
    for (int i = 0; i < total_size; ++i) {
      if (prob_data[s[i].index_*(num_classes + 1) + c] == 0) continue;
      vector<float> a;
      a.push_back(box_data[s[i].index_ * 4 + 0]);
      a.push_back(box_data[s[i].index_ * 4 + 1]);
      a.push_back(box_data[s[i].index_ * 4 + 2]);
      a.push_back(box_data[s[i].index_ * 4 + 3]);
      for (int j = i + 1; j < total_size; ++j) {
        vector<float> b;
        b.push_back(box_data[s[j].index_ * 4 + 0]);
        b.push_back(box_data[s[j].index_ * 4 + 1]);
        b.push_back(box_data[s[j].index_ * 4 + 2]);
        b.push_back(box_data[s[j].index_ * 4 + 3]);
        if (caffe::box_iou(a, b) > FLAGS_nms) {
          prob_data[s[j].index_*(num_classes + 1) + c] = 0;
        }
      }
    }
  }

  delete[] s;
}

void draw_detections(const string &input, int num, const float *boxes, const float *probs, int classes) {
  CHECK_GT(FLAGS_label.size(), 0) << "Need label file to show result.";
  vector<string> labels;
  ifstream in;
  in.open(FLAGS_label.c_str());
  for (int i =0; i < classes; i++) {
    string name;
    getline(in, name);
    labels.push_back(name);
  }
  in.close();
  Mat orig_image = imread(input, CV_LOAD_IMAGE_COLOR);
  for (int i = 0; i < num; ++i) {
    int class_id = -1;
    if (probs[i * (classes + 1) + classes] != 0) {
      for (int j = 0; j < classes; j++)
        if (probs[i * (classes + 1) + classes] == probs[i * (classes + 1) + j])
          class_id = j;
      if (class_id == -1)
        continue;
      const float* b = &boxes[i * 4];
      cout << labels[class_id] << ": " << probs[i * (classes + 1) + class_id] << endl;
      int left = int((*b - *(b + 2) / 2.) * orig_image.cols);
      int right = int((*b + *(b + 2) / 2.) * orig_image.cols);
      int top = int((*(b + 1) - *(b + 3) / 2.) * orig_image.rows);
      int bot = int((*(b + 1) + *(b + 3) / 2.) * orig_image.rows);
      if (left < 0) left = 0;
      if (right > orig_image.cols - 1) right = orig_image.cols - 1;
      if (top < 0) top = 0;
      if (bot > orig_image.rows - 1) bot = orig_image.rows - 1;
      rectangle(orig_image, Point(left, top), Point(right, bot), Scalar(0, 0, 255), 2);
      putText(orig_image, labels[class_id], Point(left, top + 20), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
    }
  }
  imshow("Result", orig_image);
  waitKey(0);
}

// Test: score a model.
int yolo_detection() {
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";

#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  Blob<float> *input = caffe_net.input_blobs()[0];
  int input_w = input->shape(3);
  int input_h = input->shape(2);

  // letterbox image size
  int new_w = 0;
  int new_h = 0;
  Mat orig_image = imread(FLAGS_input, CV_LOAD_IMAGE_COLOR);
  if (((float)input_w / orig_image.cols) < ((float)input_h / orig_image.rows)) {
    new_w = input_w;
    new_h = (orig_image.rows * input_w) / orig_image.cols;
  } else {
    new_h = input_h;
    new_w = (orig_image.cols * input_h) / orig_image.rows;
  }

  // preprocess image
  preprocess_image(caffe_net, orig_image, input_w, input_h, new_w, new_h);

  const vector<Blob<float>*>& result = caffe_net.Forward();

  float* box_data = result[0]->mutable_cpu_data();
  int box_size = result[0]->count();
  float* prob_data = result[1]->mutable_cpu_data();

  for (int i = 0; i < box_size; i += 4) {
    box_data[i] = (box_data[i] - (input_w - new_w) / 2.0f / input_w) / ((float)new_w / input_w);
    box_data[i + 1] = (box_data[i + 1] - (input_h - new_h) / 2.0f / input_h) / ((float)new_h / input_h);
    box_data[i + 2] *= (float)input_w / new_w;
    box_data[i + 3] *= (float)input_h / new_h;
  }

  caffe::RegionLayer<float> *layer = dynamic_cast<caffe::RegionLayer<float> *>(caffe_net.layer_by_name("region").get());
  int total_size = layer->height_ * layer->width_ * layer->num_;
  num_class = layer->num_class_;
  if (FLAGS_nms) {
    do_nms_sort(box_data, prob_data, total_size, num_class);
  }

  draw_detections(FLAGS_input, total_size, result[0]->cpu_data(), result[1]->cpu_data(), num_class);

  return 0;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = true;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  // Usage message.
  gflags::SetUsageMessage("Test a object detection model\n"
        "Usage:\n"
        "    yolo_detection [FLAGS] \n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  return yolo_detection();
}

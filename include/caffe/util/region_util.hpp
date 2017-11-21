#ifndef CAFFE_REGION_UTIL_HPP_
#define CAFFE_REGION_UTIL_HPP_

#include <cmath>
#include <vector>
#include <cfloat>

namespace caffe {

template <typename Dtype>
inline void softmax_op(const Dtype* input, int classes, int stride, Dtype* output) {
  Dtype sum = 0;
  Dtype large = -FLT_MAX;
  for (int i = 0; i < classes; ++i) {
    if (input[i * stride] > large)
      large = input[i * stride];
  }
  for (int i = 0; i < classes; ++i) {
    Dtype e = exp(input[i * stride] - large);
    sum += e;
    output[i * stride] = e;
  }
  for (int i = 0; i < classes; ++i) {
    output[i * stride] /= sum;
  }
}

template <typename Dtype>
inline void softmax_cpu(const Dtype *input, Dtype *output, int n, int batch, int batch_offset, int groups,
                        int group_offset, int stride) {
  for (int b = 0; b < batch; ++b) {
    for (int g = 0; g < groups; ++g) {
      softmax_op(input + b * batch_offset + g * group_offset, n, stride, output + b * batch_offset + g * group_offset);
    }
  }
}

template <typename Dtype>
inline std::vector<Dtype> get_region_box(Dtype* x, std::vector<Dtype> biases, int n, int index,
                                         int i, int j, int w, int h, int stride) {
  std::vector<Dtype> b;
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

inline int entry_index(int batch, int location, int entry, int num, int width, int height, int coords, int num_class) {
  int n = location / (width * height);
  int loc = location % (width * height);
  int outputs = height * width * num * (num_class + coords + 1);
  return batch * outputs + n * outputs / num + entry * width * height + loc;
}

template <typename Dtype>
inline void forward_softmax(const Dtype *input_data, Dtype *output_data, int batch, int num, int width, int height,
                            int coords, int num_class, bool softmax) {
  for (int b = 0; b < batch; b++) {
    for (int n = 0; n < num; n++) {
      int index = entry_index(b, n * width * height, 0, num, width, height, coords, num_class);
      for (int k = 0; k < 2 * width * height; k++) {
        output_data[index + k] = sigmoid(input_data[index + k]);
      }
      index = entry_index(b, n * width * height, coords, num, width, height, coords, num_class);
      for (int k = 0; k < width * height; k++) {
        output_data[index + k] = sigmoid(input_data[index + k]);
      }
    }
  }
  if (softmax) {
    int index = entry_index(0, 0, coords + 1, num, width, height, coords, num_class);
    softmax_cpu(input_data + index, output_data + index, num_class, batch * num,
                height * width * (num_class + coords + 1), width * height, 1, width * height);
  }
}

template <typename Dtype>
inline void get_region_boxes(Dtype *input_data, Dtype *box_data, Dtype *prob_data, int num, int width, int height,
                             int coords, int num_class, float thresh, std::vector<Dtype> &biases) {
  for (int i = 0; i < width * height; ++i) {
    int row = i / width;
    int col = i % width;
    for (int n = 0; n < num; ++n) {
      int index = n * width * height + i;
      for (int j = 0; j < num_class + 1; ++j) {
        prob_data[index * (num_class + 1) + j] = 0;
      }
      int obj_index = entry_index(0, n * width * height + i, coords, num, width, height, coords, num_class);
      int box_index = entry_index(0, n * width * height + i, 0, num, width, height, coords, num_class);
      float scale = input_data[obj_index];
      std::vector<Dtype> box = get_region_box(input_data, biases, n, box_index, col, row, width, height,
                                         width * height);
      for (int k = 0; k < box.size(); k++)
        box_data[index * coords + k] = box[k];
      float max_prob = 0;
      for (int j = 0; j < num_class; ++j) {
        int class_index = entry_index(0, n * width * height + i, coords + 1 + j, num, width, height, coords, num_class);
        float prob = scale * input_data[class_index];
        prob_data[index * (num_class + 1) + j] = (prob > thresh) ? prob : 0;
        if (prob > max_prob) max_prob = prob;
      }
      prob_data[index * (num_class + 1) + num_class] = (max_prob > thresh) ? max_prob : 0;
    }
  }
}

}  // namespace caffe

#endif  // CAFFE_REGION_UTIL_HPP_

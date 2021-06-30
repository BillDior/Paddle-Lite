// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/backends/arm/math/fp16/prior_box_fp16.h"
#include <algorithm>
#include <cmath>
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

void ExpandAspectRatios(const std::vector<float16_t>& input_aspect_ratior,
                        bool flip,
                        std::vector<float16_t>* output_aspect_ratior) {
  constexpr float16_t epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back((float16_t)1);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float16_t ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back((float16_t)1 / ar);
      }
    }
  }
}

void DensityPriorBox(const lite::Tensor* input,
                     const lite::Tensor* image,
                     lite::Tensor* boxes,
                     lite::Tensor* variances,
                     const std::vector<float16_t>& min_size_,
                     const std::vector<float16_t>& fixed_size_,
                     const std::vector<float16_t>& fixed_ratio_,
                     const std::vector<int>& density_size_,
                     const std::vector<float16_t>& max_size_,
                     const std::vector<float16_t>& aspect_ratio_,
                     const std::vector<float16_t>& variance_,
                     int img_w_,
                     int img_h_,
                     float16_t step_w_,
                     float16_t step_h_,
                     float16_t offset_,
                     int prior_num_,
                     bool is_flip_,
                     bool is_clip_,
                     const std::vector<std::string>& order_,
                     bool min_max_aspect_ratios_order) {
  // compute output shape
  int win1 = input->dims()[3];
  int hin1 = input->dims()[2];
  DDim shape_out({hin1, win1, prior_num_, 4});
  boxes->Resize(shape_out);
  variances->Resize(shape_out);

  float16_t* _cpu_data = boxes->mutable_data<float16_t>();
  float16_t* _variance_data = variances->mutable_data<float16_t>();

  const int width = win1;
  const int height = hin1;
  int img_width = img_w_;
  int img_height = img_h_;
  if (img_width == 0 || img_height == 0) {
    img_width = image->dims()[3];
    img_height = image->dims()[2];
  }
  float16_t step_w = step_w_;
  float16_t step_h = step_h_;
  if (step_w == 0 || step_h == 0) {
    step_w = static_cast<float16_t>(img_width) / width;
    step_h = static_cast<float16_t>(img_height) / height;
  }
  float16_t offset = offset_;
  int step_average = static_cast<int>((step_w + step_h) * 0.5);  // add
  int channel_size = height * width * prior_num_ * 4;
  int idx = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      float16_t center_x = static_cast<float16_t>((w + offset) * step_w);
      float16_t center_y = static_cast<float16_t>((h + offset) * step_h);
      float16_t box_width;
      float16_t box_height;
      if (fixed_size_.size() > 0) {
        // add
        for (size_t s = 0; s < fixed_size_.size(); ++s) {
          int fixed_size = fixed_size_[s];
          box_width = fixed_size;
          box_height = fixed_size;

          if (fixed_ratio_.size() > 0) {
            for (size_t r = 0; r < fixed_ratio_.size(); ++r) {
              float16_t ar = static_cast<float16_t>(fixed_ratio_[r]);
              int density = density_size_[s];
              int shift = step_average / density;
              float16_t box_width_ratio =
                  static_cast<float16_t>(fixed_size_[s] * sqrt(ar));
              float16_t box_height_ratio =
                  static_cast<float16_t>(fixed_size_[s] / sqrt(ar));

              for (int p = 0; p < density; ++p) {
                for (int c = 0; c < density; ++c) {
                  float16_t center_x_temp = center_x -
                                            step_average / (float16_t)2 +
                                            shift / (float16_t)2 + c * shift;
                  float16_t center_y_temp = center_y -
                                            step_average / (float16_t)2 +
                                            shift / (float16_t)2 + p * shift;
                  // xmin
                  _cpu_data[idx++] =
                      (center_x_temp - box_width_ratio / (float16_t)2) /
                                  img_width >=
                              0
                          ? (center_x_temp - box_width_ratio / (float16_t)2) /
                                img_width
                          : 0;
                  // ymin
                  _cpu_data[idx++] =
                      (center_y_temp - box_height_ratio / (float16_t)2) /
                                  img_height >=
                              0
                          ? (center_y_temp - box_height_ratio / (float16_t)2) /
                                img_height
                          : 0;
                  // xmax
                  _cpu_data[idx++] =
                      (center_x_temp + box_width_ratio / (float16_t)2) /
                                  img_width <=
                              1
                          ? (center_x_temp + box_width_ratio / (float16_t)2) /
                                img_width
                          : 1;
                  // ymax
                  _cpu_data[idx++] =
                      (center_y_temp + box_height_ratio / (float16_t)2) /
                                  img_height <=
                              1
                          ? (center_y_temp + box_height_ratio / (float16_t)2) /
                                img_height
                          : 1;
                }
              }
            }
          } else {
            // this code for density anchor box
            if (density_size_.size() > 0) {
              CHECK_EQ(fixed_size_.size(), density_size_.size())
                  << "fixed_size_ should be same with density_size_";
              int density = density_size_[s];
              int shift = fixed_size_[s] / density;

              for (int r = 0; r < density; ++r) {
                for (int c = 0; c < density; ++c) {
                  float16_t center_x_temp = center_x -
                                            fixed_size / (float16_t)2 +
                                            shift / (float16_t)2 + c * shift;
                  float16_t center_y_temp = center_y -
                                            fixed_size / (float16_t)2 +
                                            shift / (float16_t)2 + r * shift;
                  // xmin
                  _cpu_data[idx++] =
                      (center_x_temp - box_width / (float16_t)2) / img_width >=
                              0
                          ? (center_x_temp - box_width / (float16_t)2) /
                                img_width
                          : 0;
                  // ymin
                  _cpu_data[idx++] =
                      (center_y_temp - box_height / (float16_t)2) /
                                  img_height >=
                              0
                          ? (center_y_temp - box_height / (float16_t)2) /
                                img_height
                          : 0;
                  // xmax
                  _cpu_data[idx++] =
                      (center_x_temp + box_width / (float16_t)2) / img_width <=
                              1
                          ? (center_x_temp + box_width / (float16_t)2) /
                                img_width
                          : 1;
                  // ymax
                  _cpu_data[idx++] =
                      (center_y_temp + box_height / (float16_t)2) /
                                  img_height <=
                              1
                          ? (center_y_temp + box_height / (float16_t)2) /
                                img_height
                          : 1;
                }
              }
            }

            // rest of priors: will never come here!!!
            for (size_t r = 0; r < aspect_ratio_.size(); ++r) {
              float16_t ar = aspect_ratio_[r];

              if (fabs(ar - (float16_t)1) < 1e-6) {
                continue;
              }

              int density = density_size_[s];
              int shift = fixed_size_[s] / density;
              float16_t box_width_ratio = fixed_size_[s] * sqrt(ar);
              float16_t box_height_ratio = fixed_size_[s] / sqrt(ar);

              for (int p = 0; p < density; ++p) {
                for (int c = 0; c < density; ++c) {
                  float16_t center_x_temp = center_x -
                                            fixed_size / (float16_t)2 +
                                            shift / (float16_t)2 + c * shift;
                  float16_t center_y_temp = center_y -
                                            fixed_size / (float16_t)2 +
                                            shift / (float16_t)2 + p * shift;
                  // xmin
                  _cpu_data[idx++] =
                      (center_x_temp - box_width_ratio / (float16_t)2) /
                                  img_width >=
                              0
                          ? (center_x_temp - box_width_ratio / (float16_t)2) /
                                img_width
                          : 0;
                  // ymin
                  _cpu_data[idx++] =
                      (center_y_temp - box_height_ratio / (float16_t)2) /
                                  img_height >=
                              0
                          ? (center_y_temp - box_height_ratio / (float16_t)2) /
                                img_height
                          : 0;
                  // xmax
                  _cpu_data[idx++] =
                      (center_x_temp + box_width_ratio / (float16_t)2) /
                                  img_width <=
                              1
                          ? (center_x_temp + box_width_ratio / (float16_t)2) /
                                img_width
                          : 1;
                  // ymax
                  _cpu_data[idx++] =
                      (center_y_temp + box_height_ratio / (float16_t)2) /
                                  img_height <=
                              1
                          ? (center_y_temp + box_height_ratio / (float16_t)2) /
                                img_height
                          : 1;
                }
              }
            }
          }
        }
      } else {
        float16_t* min_buf = reinterpret_cast<float16_t*>(
            TargetWrapper<TARGET(kHost)>::Malloc(sizeof(float16_t) * 4));
        float16_t* max_buf = reinterpret_cast<float16_t*>(
            TargetWrapper<TARGET(kHost)>::Malloc(sizeof(float16_t) * 4));
        float16_t* com_buf =
            reinterpret_cast<float16_t*>(TargetWrapper<TARGET(kHost)>::Malloc(
                sizeof(float16_t) * aspect_ratio_.size() * 4));

        for (size_t s = 0; s < min_size_.size(); ++s) {
          int min_idx = 0;
          int max_idx = 0;
          int com_idx = 0;
          int min_size = min_size_[s];
          // first prior: aspect_ratio = 1, size = min_size
          box_width = box_height = min_size;
          //! xmin
          min_buf[min_idx++] =
              (center_x - box_width / (float16_t)2) / img_width;
          //! ymin
          min_buf[min_idx++] =
              (center_y - box_height / (float16_t)2) / img_height;
          //! xmax
          min_buf[min_idx++] =
              (center_x + box_width / (float16_t)2) / img_width;
          //! ymax
          min_buf[min_idx++] =
              (center_y + box_height / (float16_t)2) / img_height;

          if (max_size_.size() > 0) {
            int max_size = max_size_[s];
            //! second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
            box_width = box_height = sqrtf(min_size * max_size);
            //! xmin
            max_buf[max_idx++] =
                (center_x - box_width / (float16_t)2) / img_width;
            //! ymin
            max_buf[max_idx++] =
                (center_y - box_height / (float16_t)2) / img_height;
            //! xmax
            max_buf[max_idx++] =
                (center_x + box_width / (float16_t)2) / img_width;
            //! ymax
            max_buf[max_idx++] =
                (center_y + box_height / (float16_t)2) / img_height;
          }

          //! rest of priors
          for (size_t r = 0; r < aspect_ratio_.size(); ++r) {
            float16_t ar = aspect_ratio_[r];
            if (fabs(ar - (float16_t)1) < 1e-6) {
              continue;
            }
            box_width = min_size * sqrt(ar);
            box_height = min_size / sqrt(ar);
            //! xmin
            com_buf[com_idx++] =
                (center_x - box_width / (float16_t)2) / img_width;
            //! ymin
            com_buf[com_idx++] =
                (center_y - box_height / (float16_t)2) / img_height;
            //! xmax
            com_buf[com_idx++] =
                (center_x + box_width / (float16_t)2) / img_width;
            //! ymax
            com_buf[com_idx++] =
                (center_y + box_height / (float16_t)2) / img_height;
          }
          if (min_max_aspect_ratios_order) {
            memcpy(_cpu_data + idx, min_buf, sizeof(float16_t) * min_idx);
            idx += min_idx;
            memcpy(_cpu_data + idx, max_buf, sizeof(float16_t) * max_idx);
            idx += max_idx;
            memcpy(_cpu_data + idx, com_buf, sizeof(float16_t) * com_idx);
            idx += com_idx;
          } else {
            memcpy(_cpu_data + idx, min_buf, sizeof(float16_t) * min_idx);
            idx += min_idx;
            memcpy(_cpu_data + idx, com_buf, sizeof(float16_t) * com_idx);
            idx += com_idx;
            memcpy(_cpu_data + idx, max_buf, sizeof(float16_t) * max_idx);
            idx += max_idx;
          }
        }
        TargetWrapper<TARGET(kHost)>::Free(min_buf);
        TargetWrapper<TARGET(kHost)>::Free(max_buf);
        TargetWrapper<TARGET(kHost)>::Free(com_buf);
      }
    }
  }
  //! clip the prior's coordinate such that it is within [0, 1]
  if (is_clip_) {
    for (int d = 0; d < channel_size; ++d) {
      _cpu_data[d] =
          (std::min)((std::max)(_cpu_data[d], float16_t(0)), float16_t(1));
    }
  }
  //! set the variance.
  int count = 0;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int i = 0; i < prior_num_; ++i) {
        for (int j = 0; j < 4; ++j) {
          _variance_data[count] = variance_[j];
          ++count;
        }
      }
    }
  }
}

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

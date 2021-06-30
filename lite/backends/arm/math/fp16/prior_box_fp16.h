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

#pragma once
#include <string>
#include <vector>
#include "lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {
namespace fp16 {

typedef __fp16 float16_t;
void ExpandAspectRatios(const std::vector<float16_t>& input_aspect_ratior,
                        bool flip,
                        std::vector<float16_t>* output_aspect_ratior);

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
                     bool min_max_aspect_ratios_order);

}  // namespace fp16
}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle

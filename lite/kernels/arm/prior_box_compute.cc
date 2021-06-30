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

#include "lite/kernels/arm/prior_box_compute.h"
#include <algorithm>
#include <string>
#include <vector>

#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/prior_box_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#ifdef ENABLE_ARM_FP16

using float16_t = __fp16;

void PriorBoxCompute::ReInitWhenNeeded() {
  auto& param = this->template Param<param_t>();
  auto input_dims = param.input->dims();
  auto image_dims = param.image->dims();
  if (last_input_shape_ == input_dims && last_image_shape_ == image_dims) {
    return;
  }
  bool is_flip = param.flip;
  bool is_clip = param.clip;
  std::vector<float16_t> min_size(param.min_sizes.size());
  std::vector<float16_t> max_size(param.max_sizes.size());
  std::vector<float16_t> aspect_ratio(param.aspect_ratios.size());
  std::vector<float16_t> variance(param.variances_.size());

  auto fp32_to_fp16 = [](float x) { return static_cast<float16_t>(x); };
  std::transform(param.min_sizes.begin(),
                 param.min_sizes.end(),
                 min_size.begin(),
                 fp32_to_fp16);

  std::transform(param.max_sizes.begin(),
                 param.max_sizes.end(),
                 max_size.begin(),
                 fp32_to_fp16);

  std::transform(param.aspect_ratios.begin(),
                 param.aspect_ratios.end(),
                 aspect_ratio.begin(),
                 fp32_to_fp16);

  std::transform(param.variances_.begin(),
                 param.variances_.end(),
                 variance.begin(),
                 fp32_to_fp16);
  int img_w = param.img_w;
  int img_h = param.img_h;
  float16_t step_w = param.step_w;
  float16_t step_h = param.step_h;
  float16_t offset = param.offset;
  std::vector<float16_t> aspect_ratios_vec;
  lite::arm::math::fp16::ExpandAspectRatios(
      aspect_ratio, is_flip, &aspect_ratios_vec);
  size_t prior_num = aspect_ratios_vec.size() * min_size.size();
  prior_num += max_size.size();
  std::vector<std::string> order = param.order;
  bool min_max_aspect_ratios_order = param.min_max_aspect_ratios_order;
  lite::arm::math::fp16::DensityPriorBox(param.input,
                                         param.image,
                                         &boxes_tmp_,
                                         &variances_tmp_,
                                         min_size,
                                         std::vector<float16_t>(),
                                         std::vector<float16_t>(),
                                         std::vector<int>(),
                                         max_size,
                                         aspect_ratios_vec,
                                         variance,
                                         img_w,
                                         img_h,
                                         step_w,
                                         step_h,
                                         offset,
                                         prior_num,
                                         is_flip,
                                         is_clip,
                                         order,
                                         min_max_aspect_ratios_order);
  last_input_shape_ = input_dims;
  last_image_shape_ = image_dims;
}

void PriorBoxCompute::Run() {
  auto& param = this->template Param<param_t>();
  param.boxes->CopyDataFrom(boxes_tmp_);
  param.variances->CopyDataFrom(variances_tmp_);
}
#endif

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef ENABLE_ARM_FP16
REGISTER_LITE_KERNEL(prior_box,
                     kARM,
                     kFP16,
                     kNCHW,
                     paddle::lite::kernels::arm::PriorBoxCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindInput("Image", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Boxes",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .BindOutput("Variances",
                {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
    .Finalize();
#endif  // ENABLE_ARM_FP16

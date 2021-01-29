// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/p_norm_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

inline void GetDims(
    const DDim& dim, int axis, int* pre, int* n, int* post, bool asvector) {
  *pre = 1;
  *post = 1;
  *n = dim[axis];
  if (asvector) {
    *n = dim.production();
  } else {
    for (int i = 0; i < axis; ++i) {
      (*pre) *= dim[i];
    }
    for (int i = axis + 1; i < dim.size(); ++i) {
      (*post) *= dim[i];
    }
  }
}

void PNormCompute::Run() {
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto& param = Param<operators::PNormParam>();
  auto x = param.X;
  auto xdim = x->dims();
  float porder = param.porder;
  int axis = param.axis;
  const auto* x_data = x->data<float>();
  auto* out_data = param.Out->mutable_data<float>();
  if (axis < 0) {
    axis += xdim.size();
  }
  int pre, n, post;
  GetDims(xdim, axis, &pre, &n, &post, param.asvector);
  lite::arm::math::norm(x_data, pre, n, post, param.epsilon, out_data, &ctx);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    p_norm, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::PNormCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
    .Finalize();

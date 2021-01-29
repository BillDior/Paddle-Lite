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

#include "lite/operators/p_norm_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool PNormOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}
bool PNormOpLite::InferShapeImpl() const {
  auto x_dim = param_.X->dims();
  auto x_rank = x_dim.size();
  int& axis = param_.axis;
  bool& keepdim = param_.keepdim;

  CHECK_GE(axis, -x_rank)
      << "Attr(axis) value should be in range [-R, R-1], R is "
      << "the rank of Input(X). But received axis: " << axis
      << ", R: " << x_rank << ". "
      << "Current Input(X)'s shape is=[" << x_dim << "].";

  CHECK_LT(axis, x_rank)
      << "Attr(axis) value should be in range [-R, R-1], R is "
      << "the rank of Input(X). But received axis: " << axis
      << ", R: " << x_rank << ". "
      << "Current Input(X)'s shape is=[" << x_dim << "].";

  std::vector<int64_t> reduce_dims;
  const bool asvector = param_.asvector;
  if (asvector) {
    reduce_dims.emplace_back(1);
    if (keepdim) {
      for (int64_t i = 1; i < x_dim.size(); ++i) {
        reduce_dims.emplace_back(1);
      }
      x_dim.ConstructFrom(reduce_dims);
    }
  } else {
    if (axis < 0) {
      axis = x_dim.size() + axis;
    }
    for (int i = 0; i < x_dim.size(); ++i) {
      if (i != axis) reduce_dims.emplace_back(x_dim[i]);
    }
    if (reduce_dims.size() == 0) {
      reduce_dims.emplace_back(1);
    }
  }
  x_dim[axis] = 1;

  if (keepdim) {
    param_.Out->Resize(x_dim);
  } else {
    param_.Out->Resize(reduce_dims);
  }
  return true;
}

bool PNormOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto x_var = scope->FindVar(opdesc.Input("X").front());
  CHECK(x_var != nullptr);
  param_.X = x_var->GetMutable<Tensor>();

  auto out_var = scope->FindVar(opdesc.Output("Out").front());
  CHECK(out_var != nullptr);
  param_.Out = out_var->GetMutable<Tensor>();

  if (opdesc.HasAttr("keepdim")) {
    param_.keepdim = opdesc.GetAttr<bool>("keepdim");
  }
  if (opdesc.HasAttr("axis")) {
    param_.axis = opdesc.GetAttr<int>("axis");
  }
  if (opdesc.HasAttr("epsilon")) {
    param_.epsilon = opdesc.GetAttr<float>("epsilon");
  }
  if (opdesc.HasAttr("asvector")) {
    param_.asvector = opdesc.GetAttr<bool>("asvector");
  }
  if (opdesc.HasAttr("porder")) {
    param_.porder = opdesc.GetAttr<float>("porder");
  }
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(p_norm, paddle::lite::operators::PNormOpLite);

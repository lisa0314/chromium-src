// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_EXECUTION_IMPL_CL_DNN_H_
#define SERVICES_ML_EXECUTION_IMPL_CL_DNN_H_

#include <memory>
#include <vector>

#include "base/macros.h"
#include "mojo/public/cpp/system/buffer.h"
#include "services/ml/common.h"
#include "services/ml/public/mojom/execution.mojom.h"
#include "third_party/clDNN/api/C/cldnn.h"

namespace ml {

class CompilationDelegateClDnn;

class ExecutionImplClDnn : public mojom::Execution {
 public:
  ExecutionImplClDnn(const CompilationDelegateClDnn*,
                     mojom::ExecutionInitParamsPtr params);
  ~ExecutionImplClDnn() override;

  void StartCompute(StartComputeCallback callback) override;

 private:
  mojom::ExecutionInitParamsPtr params_;

  std::vector<cldnn_memory> input_memories_;
  cldnn_network network_;

  DISALLOW_COPY_AND_ASSIGN(ExecutionImplClDnn);
};

}  // namespace ml

#endif  // SERVICES_ML_EXECUTION_IMPL_CL_DNN_H_
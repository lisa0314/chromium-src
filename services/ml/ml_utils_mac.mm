// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/ml/common.h"
#include "services/ml/ml_utils_mac.h"

namespace ml {

OperandMac::OperandMac() = default;
OperandMac::OperandMac(const OperandMac& operand) = default;
OperandMac::OperandMac(const mojom::OperandPtr& operand_ptr) {
  type = operand_ptr->type;
  dimensions = operand_ptr->dimensions;
  scale = operand_ptr->scale;
  zeroPoint = operand_ptr->zeroPoint;
}

OperandMac::OperandMac(const ml::Operand& operand) {
  type = operand.type;
  dimensions = operand.dimensions;
  scale = operand.scale;
  zeroPoint = operand.zeroPoint;
}

OperandMac::~OperandMac() = default;

OperationMac::OperationMac() = default;
OperationMac::OperationMac(const OperationMac& operation) = default;
OperationMac::OperationMac(const mojom::OperationPtr& operation_ptr) {
  type = operation_ptr->type;
  inputs = operation_ptr->inputs;
  outputs = operation_ptr->outputs;
  local_operation = KBNNSFilter;
}

OperationMac::OperationMac(const ml::Operation& operation) {
  type = operation.type;
  inputs = operation.inputs;
  outputs = operation.outputs;
  local_operation = KBNNSFilter;
}

OperationMac::~OperationMac() = default;

uint32_t OperandMac::requiredSize() const {
  return GetRequiredSize(type, dimensions);
}

CompiledModel::CompiledModel() = default;
CompiledModel::~CompiledModel() = default;

void CompileForModel(const mojom::ModelInfoPtr& model,
                     CompiledModel* compiled_model) {
  compiled_model->operands_.reserve(model->operands.size());
  for (uint32_t i = 0; i < model->operands.size(); ++i) {
    OperandMac operand(model->operands[i]);
    compiled_model->operands_.push_back(operand);
  }
  compiled_model->operations_.reserve(model->operations.size());
  for (uint32_t i = 0; i < model->operations.size(); ++i) {
    OperationMac operation(model->operations[i]);
    operation.filter = nullptr;
    compiled_model->operations_.push_back(operation);
  }

  compiled_model->inputs_ = model->inputs;
  compiled_model->outputs_ = model->outputs;
}

bool ParameterExtracterForConv(const OperationMac& operation,
                               const std::vector<uint32_t>& inputs,
                               const std::vector<uint32_t>& outputs,
                               const std::map<uint32_t, ValueInfo>& values,
                               const std::unique_ptr<int8_t[]>& memory,
                               const std::vector<OperandMac>& operands,
                               int32_t& input_batch_size,
                               int32_t& input_width,
                               int32_t& input_height,
                               int32_t& output_width,
                               int32_t& output_height,
                               bool& implicit_padding,
                               int32_t& padding_left,
                               int32_t& padding_right,
                               int32_t& padding_top,
                               int32_t& padding_bottom,
                               int32_t& stride_width,
                               int32_t& stride_height,
                               int32_t& padding_code,
                               int32_t& fuse_code,
                               int32_t& depth_out,
                               int32_t& filter_height,
                               int32_t& filter_width,
                               int32_t& depth_in,
                               int32_t& depthwise_multiplier,
                               bool depthwise) {
  uint32_t output_idx = outputs[0];
  const OperandMac& output = operands[output_idx];
  output_height = output.dimensions[1];
  output_width = output.dimensions[2];
  int32_t index = 0;
  int32_t input_idx = inputs[index++];
  const OperandMac& input = operands[input_idx];
  // depth_in is the fourth dimension of input that shape is
  // [batches, height, width, depth_in].
  input_batch_size = input.dimensions[0];
  input_height = input.dimensions[1];
  input_width = input.dimensions[2];
  depth_in = input.dimensions[3];

  const OperandMac& filter = operands[inputs[index++]];
  if (depthwise) {
    depth_out = filter.dimensions[3];
  } else {
    depth_out = filter.dimensions[0];
  }
  filter_height = filter.dimensions[1];
  filter_width = filter.dimensions[2];

  const OperandMac& bias = operands[inputs[index++]];
  DLOG(INFO) << "  bias length: " << bias.dimensions[0];

  if ((!depthwise && inputs.size() == 10) ||
      (depthwise && inputs.size() == 11)) {
    implicit_padding = false;
    padding_left = getScalarInt32(values, inputs[index++], memory.get());
    padding_right = getScalarInt32(values, inputs[index++], memory.get());
    padding_top = getScalarInt32(values, inputs[index++], memory.get());
    padding_bottom = getScalarInt32(values, inputs[index++], memory.get());
  } else if ((!depthwise && inputs.size() == 7) ||
             (depthwise && inputs.size() == 8)) {
    implicit_padding = true;
    padding_code = getScalarInt32(values, inputs[index++], memory.get());
  } else {
    DLOG(ERROR) << "  inputs size is incorrect";
    return false;
  }
  stride_width = getScalarInt32(values, inputs[index++], memory.get());
  stride_height = getScalarInt32(values, inputs[index++], memory.get());
  if (depthwise == true) {
    depthwise_multiplier =
        getScalarInt32(values, inputs[index++], memory.get());
    if (depthwise_multiplier != 1) {
      DLOG(ERROR) << "  depthwise_multiplier " << depthwise_multiplier
                  << " is not supported.";
      return false;
    }
  }
  fuse_code = getScalarInt32(values, inputs[index++], memory.get());
  return true;
}

void SetupOperandInfoForOperands(
    std::vector<std::unique_ptr<OperandInfo>>& opearnd_info_array,
    std::vector<OperandMac>& operands,
    const std::vector<uint32_t>& operands_index_array,
    mojo::ScopedSharedBufferHandle& memory,
    uint32_t& mapped_length) {
  for (size_t i = 0; i < operands_index_array.size(); ++i) {
    const uint32_t length = operands[operands_index_array[i]].requiredSize();
    mojo::ScopedSharedBufferMapping mapping =
        memory->MapAtOffset(length, mapped_length);
    std::unique_ptr<OperandInfo> info(
        new OperandInfo(mapped_length, length, std::move(mapping)));
    opearnd_info_array.push_back(std::move(info));
    mapped_length += length;
  }
}
}

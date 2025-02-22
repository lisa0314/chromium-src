// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/ml/compilation_delegate_ie.h"

#include <string>
#include <utility>

#include "base/strings/string_number_conversions.h"
#include "services/ml/execution_impl_ie.h"
#include "third_party/libinference_engine/dldt/inference-engine/include/ie_builders.hpp"
#include "third_party/libinference_engine/dldt/inference-engine/include/ie_utils.hpp"
#include "third_party/libinference_engine/dldt/inference-engine/include/inference_engine.hpp"

namespace ie = InferenceEngine;

namespace ml {

namespace {
static float asfloat(uint32_t v) {
  union {
    float f;
    std::uint32_t u;
  } converter = {0};
  converter.u = v;
  return converter.f;
}

static short f32tof16(float x) {
  static float min16 = asfloat((127 - 14) << 23);

  static float max16 = asfloat(((127 + 15) << 23) | 0x007FE000);
  static uint32_t max16f16 = ((15 + 15) << 10) | 0x3FF;

  static constexpr std::uint32_t EXP_MASK_F32 = 0x7F800000U;

  union {
    float f;
    uint32_t u;
  } v = {0};
  v.f = x;

  uint32_t s = (v.u >> 16) & 0x8000;

  v.u &= 0x7FFFFFFF;

  if ((v.u & EXP_MASK_F32) == EXP_MASK_F32) {
    if (v.u & 0x007FFFFF) {
      return static_cast<short>(s | (v.u >> (23 - 10)) | 0x0200);
    } else {
      return static_cast<short>(s | (v.u >> (23 - 10)));
    }
  }

  float halfULP = asfloat(v.u & EXP_MASK_F32) * asfloat((127 - 11) << 23);
  v.f += halfULP;

  if (v.f < min16 * 0.5f) {
    return static_cast<short>(s);
  }

  if (v.f < min16) {
    return static_cast<short>(s | (1 << 10));
  }

  if (v.f >= max16) {
    return static_cast<short>(max16f16 | s);
  }

  v.u -= ((127 - 15) << 23);

  v.u >>= (23 - 10);

  return static_cast<short>(v.u | s);
}
}  // namespace

CompilationDelegateIe::CompilationDelegateIe(const CompilationImpl* compilation)
    : CompilationDelegate(),
      compilation_(compilation),
      builder_(nullptr),
      network_(nullptr) {}

CompilationDelegateIe::~CompilationDelegateIe() {
  DLOG(INFO) << "CompilationDelegateIe::~CompilationDelegateIe()";
}

int32_t CompilationDelegateIe::Compile() {
  DLOG(INFO) << "CompilationDelegateIe::Compile";

  int32_t result = Init();
  if (result != mojom::NOT_ERROR) {
    return result;
  }

  result = BuildNetwork();
  if (result != mojom::NOT_ERROR) {
    return result;
  }

  DLOG(INFO) << "CompilationDelegateIe::Compile ends";
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::CreateExecution(
    std::unique_ptr<mojom::Execution>& execution,
    mojom::ExecutionInitParamsPtr params) {
  execution = std::make_unique<ExecutionImplIe>(this, std::move(params));
  return static_cast<ExecutionImplIe*>(execution.get())
      ->Init(compilation_->GetPreference());
}

int32_t CompilationDelegateIe::Init() {
  const ie::Version* version = ie::GetInferenceEngineVersion();
  DLOG(INFO) << "[IE] inference engine build number " << version->buildNumber;
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::BuildNetwork() {
  try {
    builder_.reset(new ie::Builder::Network("webnn"));
    DLOG(INFO) << "[IE] succeed to create network";
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] exception " << ex.what();
    return mojom::OP_FAILED;
  }

  int32_t result;
  const mojom::ModelInfoPtr& model = compilation_->GetModel();
  for (size_t i = 0; i < model->inputs.size(); ++i) {
    result = AddInput(model->inputs[i]);
    if (result != mojom::NOT_ERROR) {
      return result;
    }
  }

  for (size_t i = 0; i < model->operations.size(); ++i) {
    const mojom::OperationPtr& operation = model->operations[i];
    const int32_t type = operation->type;

    if (operation->outputs.size() != 1) {
      LOG(ERROR) << "Only 1 output is supported";
      return mojom::BAD_DATA;
    }

    int32_t result = mojom::NOT_ERROR;
    if (type == mojom::ADD || type == mojom::MUL) {
      result = AddElementwise(operation);
    } else if (type == mojom::CONV_2D || type == mojom::DEPTHWISE_CONV_2D ||
               type == mojom::ATROUS_CONV_2D ||
               type == mojom::ATROUS_DEPTHWISE_CONV_2D) {
      result = AddConvolution(operation);
    } else if (type == mojom::AVERAGE_POOL_2D || type == mojom::MAX_POOL_2D) {
      result = AddPooling(operation);
    } else if (type == mojom::SOFTMAX) {
      result = AddSoftmax(operation);
    } else if (type == mojom::RESHAPE) {
      result = AddReshape(operation);
    } else if (type == mojom::CONCATENATION) {
      result = AddConcatenation(operation);
    } else if (type == mojom::FULLY_CONNECTED) {
      result = AddFullyConnected(operation);
    } else if (type == mojom::RESIZE_BILINEAR) {
      result = AddResizeBilinear(operation);
    } else {
      LOG(ERROR) << "Operation type " << type << " is not supported.";
      return mojom::BAD_DATA;
    }

    if (result != mojom::NOT_ERROR) {
      return result;
    }
  }

  for (size_t i = 0; i < model->outputs.size(); ++i) {
    result = AddOutput(model->outputs[i]);
    if (result != mojom::NOT_ERROR) {
      return result;
    }
  }

  try {
    network_.reset(new ie::CNNNetwork(
        ie::Builder::convertToICNNNetwork(builder_->build())));
    DLOG(INFO) << "[IE] succeed to build network";

    ie::InputsDataMap input_info(network_->getInputsInfo());
    DLOG(INFO) << "[IE] inputs data map size " << input_info.size();
    for (auto itr : input_info) {
      if (itr.second->getLayout() == ie::Layout::NCHW) {
        itr.second->setLayout(ie::Layout::NHWC);
      }
      itr.second->setPrecision(ie::Precision::FP32);
      ie::SizeVector dims = itr.second->getDims();
      DLOG(INFO) << "      input item name " << itr.first << " precision "
                 << itr.second->getPrecision() << " layout "
                 << itr.second->getLayout() << " dims "
                 << VectorToString(dims.data(), dims.size());
    }

    ie::OutputsDataMap output_info(network_->getOutputsInfo());
    DLOG(INFO) << "[IE] outputs data map size " << output_info.size();
    for (auto itr : output_info) {
      ie::SizeVector dims = itr.second->getDims();
      if (itr.second->getLayout() == ie::Layout::NCHW) {
        itr.second->setLayout(ie::Layout::NHWC);
      }
      itr.second->setPrecision(ie::Precision::FP32);
      DLOG(INFO) << "      output item name " << itr.first << " precision "
                 << itr.second->getPrecision() << " layout "
                 << itr.second->getLayout() << " dims "
                 << VectorToString(dims.data(), dims.size());
    }

    size_t batch_size = network_->getBatchSize();
    DLOG(INFO) << "[IE] Batch size is " << batch_size;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] exception " << ex.what();
    return mojom::OP_FAILED;
  }
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::GetDims(const std::vector<uint32_t>& dimensions,
                                       std::vector<size_t>& dims) {
  // IE default layout is NCHW
  dims.resize(dimensions.size());
  if (dimensions.size() == 1) {
    dims[0] = dimensions[0];
  } else if (dimensions.size() == 2) {
    dims[0] = dimensions[0];
    dims[1] = dimensions[1];
  } else if (dimensions.size() == 3) {
    // HWC -> CHW
    dims[0] = dimensions[2];
    dims[1] = dimensions[0];
    dims[2] = dimensions[1];
  } else if (dimensions.size() == 4) {
    // NHWC -> NCHW
    dims[0] = dimensions[0];
    dims[1] = dimensions[3];
    dims[2] = dimensions[1];
    dims[3] = dimensions[2];
  } else {
    LOG(ERROR) << "Tensor rank " << dimensions.size() << " is not supproted";
    return mojom::BAD_DATA;
  }
  return mojom::NOT_ERROR;
}

template <typename T>
int32_t CompilationDelegateIe::Reorder(T* dst,
                                       const float* src,
                                       std::vector<uint32_t>& dims,
                                       bool nhwc_to_nchw) {
  if (!(std::is_same<T, float>::value || std::is_same<T, int16_t>::value)) {
    LOG(ERROR) << "Data type is not supported";
    return mojom::BAD_DATA;
  }
  if (dims.size() == 1 || dims.size() == 2) {
    size_t size = product(dims);
    if (std::is_same<T, float>::value) {
      const size_t buffer_length = size * sizeof(T);
      memcpy(static_cast<void*>(dst), static_cast<const void*>(src),
             buffer_length);
    } else if (std::is_same<T, int16_t>::value) {
      for (size_t i = 0; i < size; ++i) {
        dst[i] = f32tof16(src[i]);
      }
    }
  } else if (dims.size() == 3 || dims.size() == 4) {
    // dims is in NHWC
    const bool rank3 = dims.size() == 3;
    const uint32_t batches = rank3 ? 1 : dims[0];
    const uint32_t channels = rank3 ? dims[2] : dims[3];
    const uint32_t height = rank3 ? dims[0] : dims[1];
    const uint32_t width = rank3 ? dims[1] : dims[2];

    for (uint32_t b = 0; b < batches; ++b) {
      for (uint32_t c = 0; c < channels; ++c) {
        for (uint32_t y = 0; y < height; ++y) {
          for (uint32_t x = 0; x < width; ++x) {
            uint32_t dst_index, src_index;
            if (nhwc_to_nchw) {
              dst_index = b * channels * height * width + c * height * width +
                          y * width + x;
              src_index = b * height * width * channels + y * width * channels +
                          x * channels + c;
            } else {
              dst_index = b * height * width * channels + y * width * channels +
                          x * channels + c;
              src_index = b * channels * height * width + c * height * width +
                          y * width + x;
            }
            if (std::is_same<T, float>::value) {
              dst[dst_index] = src[src_index];
            } else if (std::is_same<T, int16_t>::value) {
              dst[dst_index] = f32tof16(src[src_index]);
            }
          }
        }
      }
    }
  } else {
    LOG(ERROR) << "Tensor rank " << dims.size() << " is not supproted";
    return mojom::BAD_DATA;
  }
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::CreateBlob(
    uint32_t index,
    std::shared_ptr<InferenceEngine::Blob>& blob) {
  try {
    const mojom::ModelInfoPtr& model = compilation_->GetModel();
    const mojom::OperandPtr& operand = model->operands[index];
    ie::SizeVector dims;
    int32_t result = GetDims(operand->dimensions, dims);
    if (result != mojom::NOT_ERROR) {
      return result;
    }
    int32_t pref = compilation_->GetPreference();
    if (pref != mojom::PREFER_LOW_POWER) {
      blob = ie::make_shared_blob<float>(ie::Precision::FP32, dims);
    } else {
      // MYRIAD only accepts FP16 precision
      blob = ie::make_shared_blob<int16_t>(ie::Precision::FP16, dims);
    }
    blob->allocate();
    DLOG(INFO) << "Create blob with size " << blob->size()
               << " for operand index " << index;
    if (pref != mojom::PREFER_LOW_POWER) {
      float* dst = blob->buffer().as<float*>();
      auto mapping = compilation_->MapMemory(index);
      const float* src = reinterpret_cast<const float*>(mapping.get());
      result = Reorder<float>(dst, src, operand->dimensions);
      if (result != mojom::NOT_ERROR) {
        return result;
      }
    } else {
      int16_t* dst = blob->buffer().as<int16_t*>();
      auto mapping = compilation_->MapMemory(index);
      const float* src = reinterpret_cast<const float*>(mapping.get());
      result = Reorder<int16_t>(dst, src, operand->dimensions);
      if (result != mojom::NOT_ERROR) {
        return result;
      }
    }
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to create blob " << ex.what();
    return mojom::OP_FAILED;
  }
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddInput(uint32_t index) {
  DLOG(INFO) << "AddInput for operand index " << index;
  const mojom::ModelInfoPtr& model = compilation_->GetModel();
  const mojom::OperandPtr& operand = model->operands[index];
  ie::SizeVector dims;
  int32_t result = GetDims(operand->dimensions, dims);
  if (result != mojom::NOT_ERROR) {
    return result;
  }
  std::string name(base::NumberToString(index));
  try {
    size_t layer_id = builder_->addLayer(
        ie::Builder::InputLayer(name).setPort(ie::Port(dims)));
    layer_id_map_[index] = layer_id;
    DLOG(INFO) << "[IE] succeed to add input layer id " << layer_id
               << " for operand index " << index << " with dims "
               << VectorToString(dims.data(), dims.size());
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add input layer " << ex.what();
    return mojom::OP_FAILED;
  }
  DLOG(INFO) << "AddInput ends";
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddOutput(uint32_t index) {
  DLOG(INFO) << "AddOutput for operand index " << index;
  if (layer_id_map_.find(index) == layer_id_map_.end()) {
    LOG(ERROR) << "Layer for output index " << index << " is not ready";
    return mojom::BAD_DATA;
  }
  const size_t input_layer_id = layer_id_map_[index];
  DLOG(INFO) << "[IE] input port layer id " << input_layer_id
             << " for operand index " << index;
  std::string name(base::NumberToString(index));
  name += std::string("_out");
  try {
    size_t output_id =
        builder_->addLayer({{input_layer_id}}, ie::Builder::OutputLayer(name));
    DLOG(INFO) << "[IE] succeed to add output layer id " << output_id
               << " for operand index " << index;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add output layer " << ex.what();
    return mojom::OP_FAILED;
  }
  DLOG(INFO) << "AddOutput ends";
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddConstant(uint32_t index) {
  DLOG(INFO) << "AddConstant for operand index " << index;
  const mojom::ModelInfoPtr& model = compilation_->GetModel();
  const mojom::OperandPtr& operand = model->operands[index];
  ie::SizeVector dims;
  int32_t result = GetDims(operand->dimensions, dims);
  if (result != mojom::NOT_ERROR) {
    return result;
  }
  std::string name(base::NumberToString(index));
  try {
    ie::Blob::Ptr blob;
    result = CreateBlob(index, blob);
    if (result != mojom::NOT_ERROR) {
      return result;
    }
    size_t layer_id = builder_->addLayer(
        ie::Builder::ConstLayer(name).setPort(ie::Port(dims)).setData(blob));
    layer_id_map_[index] = layer_id;
    DLOG(INFO) << "[IE] succeed to add const layer id " << layer_id
               << " for operand index " << index << " with dims "
               << VectorToString(dims.data(), dims.size());
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add const layer " << ex.what();
    return mojom::OP_FAILED;
  }
  DLOG(INFO) << "AddConstant ends";
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddActivationByFusedCode(
    int32_t fuse_code,
    size_t input_layer,
    const std::string& name,
    size_t& activiation_layer_id) {
  try {
    if (fuse_code == mojom::FUSED_RELU) {
      activiation_layer_id =
          builder_->addLayer({{input_layer}}, ie::Builder::ReLULayer(name));
      DLOG(INFO) << "[IE] succeed to add relu layer id "
                 << activiation_layer_id;
    } else if (fuse_code == mojom::FUSED_RELU1) {
      activiation_layer_id = builder_->addLayer(
          {{input_layer}},
          ie::Builder::ClampLayer(name).setMinValue(-1.0).setMaxValue(1.0));
      DLOG(INFO) << "[IE] succeed to add clamp layer id "
                 << activiation_layer_id;
    } else if (fuse_code == mojom::FUSED_RELU6) {
      activiation_layer_id = builder_->addLayer(
          {{input_layer}},
          ie::Builder::ClampLayer(name).setMinValue(0.0).setMaxValue(6.0));
      DLOG(INFO) << "[IE] succeed to add clamp layer id "
                 << activiation_layer_id;
    } else {
      LOG(ERROR) << "Fuse code " << fuse_code << " is not supported";
      return mojom::BAD_DATA;
    }
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add relu layer " << ex.what();
    return mojom::OP_FAILED;
  }
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddElementwise(
    const mojom::OperationPtr& operation) {
  DLOG(INFO) << "AddElementwise for operation type " << operation->type;
  // Setup element-wise parameters.
  ElementWiseParams params;
  int32_t result = compilation_->GetElementWiseParams(operation, params);
  if (result != mojom::NOT_ERROR)
    return result;
  DLOG(INFO) << "    fuse_code: " << params.fuse_code;

  // Check boardcasting
  const mojom::ModelInfoPtr& model = compilation_->GetModel();
  const mojom::OperandPtr& input0 = model->operands[operation->inputs[0]];
  const mojom::OperandPtr& input1 = model->operands[operation->inputs[1]];
  if (input0->dimensions != input1->dimensions) {
    LOG(ERROR) << "Boardcasting is not supported";
    return mojom::BAD_DATA;
  }

  // Binary op
  std::vector<ie::PortInfo> input_port_infos;
  std::vector<ie::Port> input_ports;
  for (size_t i = 0; i < 2; ++i) {
    const uint32_t input_index = operation->inputs[i];
    if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
      // Setup constants
      const mojom::ModelInfoPtr& model = compilation_->GetModel();
      if (model->values.find(base::NumberToString(input_index)) !=
          model->values.end()) {
        result = AddConstant(input_index);
        if (result != mojom::NOT_ERROR) {
          return result;
        }
      } else {
        LOG(ERROR) << "The layer for operand index " << input_index
                   << " is not ready";
        return mojom::BAD_DATA;
      }
    }
    const size_t layer_id = layer_id_map_[input_index];
    input_port_infos.push_back({layer_id});
    input_ports.push_back(builder_->getLayer(layer_id).getOutputPorts()[0]);
    DLOG(INFO) << "[IE] input " << i << " layer id " << layer_id
               << " operand index " << input_index;
  }
  const uint32_t output_index = operation->outputs[0];
  std::string output_name(base::NumberToString(output_index));
  std::string name(output_name);
  if (params.fuse_code != mojom::FUSED_NONE) {
    name = name + "_pre_fuse";
  }
  ie::Builder::EltwiseLayer::EltwiseType type;
  if (operation->type == mojom::ADD) {
    type = ie::Builder::EltwiseLayer::SUM;
  } else if (operation->type == mojom::MUL) {
    type = ie::Builder::EltwiseLayer::MUL;
  } else {
    LOG(ERROR) << "Operation type " << operation->type << " is not supported";
    return mojom::BAD_DATA;
  }
  try {
    size_t layer_id =
        builder_->addLayer(input_port_infos, ie::Builder::EltwiseLayer(name)
                                                 .setInputPorts(input_ports)
                                                 .setEltwiseType(type));
    DLOG(INFO) << "[IE] succeed to add eltwise layer id " << layer_id
               << " for output operand index " << output_index << " with type "
               << type;

    if (params.fuse_code != mojom::FUSED_NONE) {
      result = AddActivationByFusedCode(params.fuse_code, layer_id, output_name,
                                        layer_id);
      if (result != mojom::NOT_ERROR) {
        return result;
      }
    }
    layer_id_map_[output_index] = layer_id;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add eltwise layer " << ex.what();
    return mojom::OP_FAILED;
  }
  DLOG(INFO) << "AddElementwise ends.";
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddConvolution(
    const mojom::OperationPtr& operation) {
  // Setup convolution params.
  ConvParams params;
  int32_t result = compilation_->GetConvParams(operation, params);
  if (result != mojom::NOT_ERROR)
    return result;
  if (params.depthwise && params.depthwise_multiplier != 1) {
    LOG(ERROR) << "depthwise_multiplier " << params.depthwise_multiplier
               << " is not supported";
    return mojom::BAD_DATA;
  }
  const uint32_t input_index = operation->inputs[0];
  if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
    LOG(ERROR) << "The layer for operand index " << input_index
               << " is not ready";
    return mojom::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation->outputs[0];
    std::string output_name(base::NumberToString(output_index));
    std::string name(output_name);
    if (params.fuse_code != mojom::FUSED_NONE) {
      name = name + "_pre_fuse";
    }
    const uint32_t weights_index = operation->inputs[1];
    ie::Blob::Ptr weights;
    result = CreateBlob(weights_index, weights);
    if (result != mojom::NOT_ERROR) {
      return result;
    }
    const uint32_t bias_index = operation->inputs[2];
    ie::Blob::Ptr bias;
    result = CreateBlob(bias_index, bias);
    if (result != mojom::NOT_ERROR) {
      return result;
    }
    const size_t input_layer_id = layer_id_map_[input_index];
    DLOG(INFO) << "[IE] input port layer id " << input_layer_id
               << " for operand index " << input_index;
    size_t layer_id = builder_->addLayer(
        {{input_layer_id}},
        ie::Builder::ConvolutionLayer(name)
            .setKernel({params.filter_height, params.filter_width})
            .setGroup(params.depthwise ? params.depth_out : 1)
            .setOutDepth(params.depth_out)
            .setDilation({params.dilation_width, params.dilation_height})
            .setStrides({params.stride_width, params.stride_height})
            .setPaddingsBegin({params.padding_top, params.padding_left})
            .setPaddingsEnd({params.padding_bottom, params.padding_right})
            .setWeights(weights)
            .setBiases(bias));
    if (params.fuse_code != mojom::FUSED_NONE) {
      result = AddActivationByFusedCode(params.fuse_code, layer_id, output_name,
                                        layer_id);
      if (result != mojom::NOT_ERROR) {
        return result;
      }
    }
    layer_id_map_[output_index] = layer_id;
    DLOG(INFO) << "[IE] succeed to add convolution layer id " << layer_id
               << " for output operand index " << output_index;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add convolution layer " << ex.what();
    return mojom::OP_FAILED;
  }
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddPooling(
    const mojom::OperationPtr& operation) {
  // Setup pooling params.
  PoolingParams params;
  int32_t result = compilation_->GetPoolingParams(operation, params);
  if (result != mojom::NOT_ERROR)
    return result;
  const uint32_t input_index = operation->inputs[0];
  if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
    LOG(ERROR) << "The layer for operand index " << input_index
               << " is not ready";
    return mojom::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation->outputs[0];
    std::string output_name(base::NumberToString(output_index));
    std::string name(output_name);
    if (params.fuse_code != mojom::FUSED_NONE) {
      name = name + "_pre_fuse";
    }
    const size_t input_layer_id = layer_id_map_[input_index];
    DLOG(INFO) << "[IE] input port layer id " << input_layer_id
               << " for operand index " << input_index;
    size_t layer_id = builder_->addLayer(
        {{input_layer_id}},
        ie::Builder::PoolingLayer(name)
            .setExcludePad(true)
            .setKernel({params.filter_height, params.filter_width})
            .setStrides({params.stride_width, params.stride_height})
            .setPaddingsBegin({params.padding_top, params.padding_left})
            .setPaddingsEnd({params.padding_bottom, params.padding_right})
            .setPoolingType(operation->type == mojom::AVERAGE_POOL_2D
                                ? ie::Builder::PoolingLayer::PoolingType::AVG
                                : ie::Builder::PoolingLayer::PoolingType::MAX)
            .setRoundingType(ie::Builder::PoolingLayer::RoundingType::FLOOR));
    if (params.fuse_code != mojom::FUSED_NONE) {
      result = AddActivationByFusedCode(params.fuse_code, layer_id, output_name,
                                        layer_id);
      if (result != mojom::NOT_ERROR) {
        return result;
      }
    }
    layer_id_map_[output_index] = layer_id;
    DLOG(INFO) << "[IE] succeed to add pooling layer id " << layer_id
               << " for output operand index " << output_index;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add pooling layer " << ex.what();
    return mojom::OP_FAILED;
  }
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddSoftmax(
    const mojom::OperationPtr& operation) {
  // Setup softmax params.
  SoftmaxParams params;
  int32_t result = compilation_->GetSoftmaxParams(operation, params);
  if (result != mojom::NOT_ERROR)
    return result;
  // Check beta.
  if (params.beta != 1.0) {
    LOG(ERROR) << "beta " << params.beta << " is not supported.";
    return mojom::BAD_DATA;
  }
  const uint32_t input_index = operation->inputs[0];
  if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
    LOG(ERROR) << "The layer for operand index " << input_index
               << " is not ready";
    return mojom::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation->outputs[0];
    std::string name(base::NumberToString(output_index));
    const size_t input_layer_id = layer_id_map_[input_index];
    DLOG(INFO) << "[IE] input port layer id " << input_layer_id
               << " for operand index " << input_index;
    size_t layer_id = builder_->addLayer(
        {{input_layer_id}}, ie::Builder::SoftMaxLayer(name).setAxis(1));
    layer_id_map_[output_index] = layer_id;
    DLOG(INFO) << "[IE] succeed to add softmax layer id " << layer_id
               << " for output operand index " << output_index;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add softmax layer " << ex.what();
    return mojom::OP_FAILED;
  }
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddReshape(
    const mojom::OperationPtr& operation) {
  const uint32_t input_index = operation->inputs[0];
  if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
    LOG(ERROR) << "The layer for operand index " << input_index
               << " is not ready";
    return mojom::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation->outputs[0];
    const mojom::ModelInfoPtr& model = compilation_->GetModel();
    const mojom::OperandPtr& output = model->operands[output_index];
    ie::SizeVector dims;
    int32_t result = GetDims(output->dimensions, dims);
    if (result != mojom::NOT_ERROR) {
      return result;
    }
    std::vector<int> cdims;
    for (auto d : dims)
      cdims.push_back(static_cast<int>(d));
    std::string name(base::NumberToString(output_index));
    const size_t input_layer_id = layer_id_map_[input_index];
    DLOG(INFO) << "[IE] input port layer id " << input_layer_id
               << " for operand index " << input_index;
    size_t layer_id = builder_->addLayer(
        {{input_layer_id}}, ie::Builder::ReshapeLayer(name).setDims(cdims));
    layer_id_map_[output_index] = layer_id;
    DLOG(INFO) << "[IE] succeed to add reshape layer id " << layer_id
               << " for output operand index " << output_index;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add reshape layer " << ex.what();
    return mojom::OP_FAILED;
  }
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddConcatenation(
    const mojom::OperationPtr& operation) {
  // Setup concatenation params.
  ConcatParams params;
  int32_t result = compilation_->GetConcatParams(operation, params);
  if (result != mojom::NOT_ERROR)
    return result;

  const mojom::ModelInfoPtr& model = compilation_->GetModel();
  std::vector<ie::PortInfo> input_port_infos;
  std::vector<ie::Port> input_ports;
  for (size_t i = 0; i < operation->inputs.size() - 1; ++i) {
    const uint32_t input_index = operation->inputs[i];
    if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
      // Setup constants
      const mojom::ModelInfoPtr& model = compilation_->GetModel();
      if (model->values.find(base::NumberToString(input_index)) !=
          model->values.end()) {
        result = AddConstant(input_index);
        if (result != mojom::NOT_ERROR) {
          return result;
        }
      } else {
        LOG(ERROR) << "The layer for operand index " << input_index
                   << " is not ready";
        return mojom::BAD_DATA;
      }
    }
    const size_t layer_id = layer_id_map_[input_index];
    input_port_infos.push_back({layer_id});
    input_ports.push_back(builder_->getLayer(layer_id).getOutputPorts()[0]);
    DLOG(INFO) << "[IE] input " << i << " layer id " << layer_id
               << " operand index " << input_index;
  }

  int axis = 0;
  const uint32_t rank =
      model->operands[operation->inputs[0]]->dimensions.size();
  if (rank == 1 || rank == 2) {
    axis = params.axis;
  } else if (rank == 3) {
    // HWC -> CHW
    if (params.axis == 0) {
      axis = 1;
    } else if (params.axis == 1) {
      axis = 2;
    } else if (params.axis == 2) {
      axis = 0;
    }
  } else if (rank == 4) {
    // NHWC -> NCHW
    if (params.axis == 0) {
      axis = 0;
    } else if (params.axis == 1) {
      axis = 2;
    } else if (params.axis == 2) {
      axis = 3;
    } else if (params.axis == 3) {
      axis = 1;
    }
  } else {
    LOG(ERROR) << "rank " << rank << " is not supported.";
    return mojom::BAD_DATA;
  }

  const uint32_t output_index = operation->outputs[0];
  std::string name(base::NumberToString(output_index));
  try {
    size_t layer_id =
        builder_->addLayer(input_port_infos, ie::Builder::ConcatLayer(name)
                                                 .setInputPorts(input_ports)
                                                 .setAxis(axis));
    DLOG(INFO) << "[IE] succeed to add concat layer id " << layer_id
               << " for output operand index " << output_index << "with axis"
               << axis;
    layer_id_map_[output_index] = layer_id;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add concat layer " << ex.what();
    return mojom::OP_FAILED;
  }
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddFullyConnected(
    const mojom::OperationPtr& operation) {
  // Setup fully connected params.
  FullyConnectedParams params;
  int32_t result = compilation_->GetFullyConnectedParams(operation, params);
  if (result != mojom::NOT_ERROR)
    return result;
  const uint32_t input_index = operation->inputs[0];
  if (layer_id_map_.find(input_index) == layer_id_map_.end()) {
    LOG(ERROR) << "The layer for operand index " << input_index
               << " is not ready";
    return mojom::BAD_DATA;
  }
  try {
    const uint32_t output_index = operation->outputs[0];
    std::string output_name(base::NumberToString(output_index));
    std::string name(output_name);
    if (params.fuse_code != mojom::FUSED_NONE) {
      name = name + "_pre_fuse";
    }
    const uint32_t weights_index = operation->inputs[1];
    ie::Blob::Ptr weights;
    result = CreateBlob(weights_index, weights);
    if (result != mojom::NOT_ERROR) {
      return result;
    }
    const uint32_t bias_index = operation->inputs[2];
    ie::Blob::Ptr bias;
    result = CreateBlob(bias_index, bias);
    if (result != mojom::NOT_ERROR) {
      return result;
    }
    size_t input_layer_id = layer_id_map_[input_index];
    DLOG(INFO) << "[IE] input port layer id " << input_layer_id
               << " for operand index " << input_index;
    std::string reshape_name = name + "-reshape";
    size_t layer_id = builder_->addLayer(
        {{input_layer_id}},
        ie::Builder::ReshapeLayer(reshape_name)
            .setDims({params.input_batch_size, params.input_size}));
    layer_id = builder_->addLayer({{layer_id}},
                                  ie::Builder::FullyConnectedLayer(name)
                                      .setOutputNum(params.output_num_units)
                                      .setWeights(weights)
                                      .setBiases(bias));
    if (params.fuse_code != mojom::FUSED_NONE) {
      result = AddActivationByFusedCode(params.fuse_code, layer_id, output_name,
                                        layer_id);
      if (result != mojom::NOT_ERROR) {
        return result;
      }
    }
    layer_id_map_[output_index] = layer_id;
    DLOG(INFO) << "[IE] succeed to add fc layer id " << layer_id
               << " for output operand index " << output_index;
  } catch (const std::exception& ex) {
    LOG(ERROR) << "[IE] failed to add fc layer " << ex.what();
    return mojom::OP_FAILED;
  }
  return mojom::NOT_ERROR;
}

int32_t CompilationDelegateIe::AddResizeBilinear(
    const mojom::OperationPtr& operation) {
  // Setup resize bilinear params.
  ResizeBilinearParams params;
  int32_t result = compilation_->GetResizeBilinearParams(operation, params);
  if (result != mojom::NOT_ERROR)
    return result;

  LOG(ERROR) << "Operation type " << operation->type << " is not supported.";
  return mojom::BAD_DATA;
}

}  // namespace ml

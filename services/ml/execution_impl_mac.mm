// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/mac/scoped_nsautorelease_pool.h"
#include "services/ml/execution_impl_mac.h"
#include "services/ml/mpscnn_context.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace ml {

NSString*
API_AVAILABLE(macosx(10.13)) KernelFor(const MPSImage* X, NSString* arrayKernel, NSString* nonArrayKernel) {
    if (X.featureChannels > 4) {
        return arrayKernel;
    }
    if (X.numberOfImages > 1) {
        return arrayKernel;
    }
    return nonArrayKernel;
}

auto divRoundUp(uint x, uint y) -> uint {
    return (x + y - 1) / y;
}

struct LaunchParams {
    MTLSize threadsPerThreadgroup;
    MTLSize threadgroupsPerGrid;
};

LaunchParams API_AVAILABLE(macosx(10.13)) SpatialPointwiseKernelLaunchParams(
                                                id<MTLComputePipelineState> pipeline,
                                                const MPSImage* im) {
    //const auto maxThreadsPerThreadgroup =
    //[pipeline maxTotalThreadsPerThreadgroup];
    //const auto threadExecutionWidth = [pipeline threadExecutionWidth];
    const auto threadsPerThreadgroup = MTLSizeMake(
                                                   8 /* threadExecutionWidth */,
                                                   4 /* maxThreadsPerThreadgroup / threadExecutionWidth */,
                                                   1);
    const auto threadgroupsPerGrid = MTLSizeMake(
                                                 divRoundUp(im.width, threadsPerThreadgroup.width),
                                                 divRoundUp(im.height, threadsPerThreadgroup.height),
                                                 im.numberOfImages * divRoundUp(im.featureChannels, 4));
    return {threadsPerThreadgroup, threadgroupsPerGrid};
};

bool GetMPSImageInfo(const OperandMac& operand, uint32_t& n, uint32_t& width, uint32_t& height, uint32_t& channels) {
  const std::vector<uint32_t>& dimensions = operand.dimensions;
  if (dimensions.size() == 4) {
    n = dimensions[0];
    height = dimensions[1];
    width = dimensions[2];
    channels = dimensions[3];
    return true;
  } else if (dimensions.size() == 2) {
    n = dimensions[0];
    channels = dimensions[1];
    height = 1;
    width = 1;
    return true;
  } else {
    DLOG(ERROR) << "dimension " << dimensions.size() << " is not supported";
    return false;
  }
}

MPSImageDescriptor* API_AVAILABLE(macosx(10.13)) CreateMPSImageDescriptor(const OperandMac& operand) {
  int32_t type = operand.type;
  MPSImageDescriptor* mpsimage_desc = nullptr;
  if (type != mojom::TENSOR_FLOAT32) {
    DLOG(ERROR) << "type " << type << " is not supported";
    return mpsimage_desc;
  }
  uint32_t n, width, height, channels;
  if (!GetMPSImageInfo(operand, n, width, height, channels)) {
    return mpsimage_desc;
  }
  if (n != 1) {
    DLOG(ERROR) << "number of images " << n << " is not supported";
    return mpsimage_desc;
  }
  mpsimage_desc = [MPSImageDescriptor
      imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
      width:width
      height:height
      featureChannels:channels
      numberOfImages:n
      usage:MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite];
  DLOG(INFO) << "Create MPSImageDescriptor " << mpsimage_desc
      << " [" << width << ", " << height << ", " << channels << "]";
  return mpsimage_desc;
}

ExecutionImplMac::ExecutionImplMac(CompilationImplMac* compilation, mojo::ScopedSharedBufferHandle memory) {
  compilation_ = compilation;
  memory_ = std::move(memory);
  uint32_t total_length = 0;
  uint32_t inputs_size = compilation_->inputs_.size();
  if (@available(macOS 10.13, *)) {
    input_mpsimages_.resize(inputs_size);
    input_mtlbuffers_.resize(inputs_size);
  }
  for (size_t i = 0; i < inputs_size; ++i) {
    OperandMac& operand = compilation_->operands_[compilation_->inputs_[i]];
    uint32_t offset = total_length;
    uint32_t length = operand.requiredSize();
    mojo::ScopedSharedBufferMapping mapping = memory_->MapAtOffset(length, offset);
    std::unique_ptr<OperandInfo> info(new OperandInfo(offset, length, std::move(mapping)));
    inputs_info_.push_back(std::move(info));
    total_length += length;
    if (@available(macOS 10.13, *)) {
      MPSImage* mps_img = [[MPSImage alloc]
          initWithDevice:GetMPSCNNContext().device
          imageDescriptor:CreateMPSImageDescriptor(operand)];
      input_mpsimages_[i].reset(mps_img);
      input_mtlbuffers_[i] = [GetMPSCNNContext().device
          newBufferWithLength:length
          options:MTLResourceOptionCPUCacheModeWriteCombined];
    }
  }
  uint32_t outputs_size = compilation_->outputs_.size();
  if (@available(macOS 10.13, *)) {
    output_mpsimages_.resize(outputs_size);
    output_mtlbuffers_.resize(outputs_size);
  }
  for (size_t i = 0; i < outputs_size; ++i) {
    OperandMac& operand = compilation_->operands_[compilation_->outputs_[i]];
    uint32_t offset = total_length;
    uint32_t length = operand.requiredSize();
    mojo::ScopedSharedBufferMapping mapping = memory_->MapAtOffset(length, offset);
    std::unique_ptr<OperandInfo> info(new OperandInfo(offset, length, std::move(mapping)));
    outputs_info_.push_back(std::move(info));
    total_length += length;
    if (@available(macOS 10.13, *)) {
      MPSImage* mps_img = [[MPSImage alloc]
          initWithDevice:GetMPSCNNContext().device
          imageDescriptor:CreateMPSImageDescriptor(operand)];
      output_mpsimages_[i].reset(mps_img);
      output_mtlbuffers_[i] = [GetMPSCNNContext().device
          newBufferWithLength:length
          options:MTLResourceOptionCPUCacheModeWriteCombined];
    }
  }
}

ExecutionImplMac::~ExecutionImplMac() {}

void ExecutionImplMac::_base_softmax(const float *in, float *out, int size, float beta) {
  if (size == 0) { 
    return; 
  }
  float max_val = *(in);
  for (int i = 0; i < size; i++) {
    float cur_val = *(in + i);
    if (cur_val > max_val) {
      max_val = cur_val ;
    }
  }
  float sum_val = 0.0;
  for (int i = 0; i < size; i++) {
    float cur_val = *(in + i) ;
    sum_val += expf((cur_val - max_val) * beta);
  }
  for (int i = 0; i < size; i++) {
    float cur_val = *(in + i);
    *(out + i) = expf((cur_val - max_val) * beta) / sum_val;
  }
}

void ExecutionImplMac::startCompute(startComputeCallback callback) {
  DLOG(INFO) << "ExecutionImplMac::startCompute";
  bool success = true;
  if (@available(macOS 10.13, *)) {
    do {
      @autoreleasepool {  
        if (compilation_->inputs_.size() > 1) {
          DLOG(ERROR) << "Only input size 1 is supported";
          success = false;
          break;
        }

        if (compilation_->is_BNNS) {
          const uint32_t input_idx = compilation_->inputs_[0];
          std::unique_ptr<OperandInfo>& input_data = inputs_info_[0];
          // input data pointer
          void* in = input_data->mapping.get();
          float* bnns_input = nullptr ;
          // void* current_input ;
          // void* current_output ;

          for (size_t i = 0; i < compilation_->operations_.size(); i++) {
            const OperationMac& operation = compilation_->operations_[i];
            float *src = nullptr;
            float *des = nullptr;
            if (operation.local_operation == BNNSFilter) {
              uint32_t operation_input_idx = operation.inputs[0];
              const OperandMac& operation_input = compilation_->operands_[operation_input_idx];

              int32_t input_batch = operation_input.dimensions[0];
              int32_t input_height = operation_input.dimensions[1];
              int32_t input_width = operation_input.dimensions[2];
              int32_t input_depth = operation_input.dimensions[3];
              int32_t input_row_stride = input_width;
              int32_t input_image_stride = input_height;
              bnns_input = (float*) malloc(sizeof(float) * input_batch * input_height * input_width * input_depth);

              if (operation_input_idx == input_idx) {  
                float *raw_input = (float*) in;
      
                // With Metal we could just load the source image into an MTLTexture but
                // we cannot use such textures directly with BNNS. Here, we load the RGBA
                // pixel data and convert it to an array of Floats that first has all the
                // R values, then all the G values, and then all the B values (instead of
                // interleaved RGBA values as in MTLTexture).
                for ( int b = 0; b < input_batch; b ++ ) {
                  for ( int h = 0; h < input_height; h ++ ) {
                    for ( int w = 0; w < input_width; w ++ ) {
                      for ( int d = 0; d < input_depth; d ++ ) {
                        int batch_offset = b * input_height * input_width * input_depth;
                        int raw_index =  batch_offset + h * input_width * input_depth + w * input_depth + d;
                        int bnns_index = batch_offset + w + h * input_row_stride + d * input_image_stride ;
                        *(bnns_input + bnns_index) = *(raw_input + raw_index) ;
                      }
                    }
                  }
                }
              }  
              src = bnns_input;
              uint32_t operation_output_idx = operation.outputs[0];
              const OperandMac& operation_output = compilation_->operands_[operation_output_idx];
              int32_t output_batch = operation_output.dimensions[0];
              int32_t output_height = operation_output.dimensions[1];
              int32_t output_width = operation_output.dimensions[2];
              int32_t output_depth = operation_output.dimensions[3];
              // int32_t output_row_stride = output_width;
              // int32_t output_image_stride = output_height;

              des = (float*) malloc(sizeof(float) * output_batch * output_height * output_width * output_depth);
              BNNSFilterApply(operation.filter, src, des);
            } else if (operation.local_operation == SoftMax) {
              const OperandMac& shape = compilation_->operands_[operation.inputs[0]];
              uint32_t batch_count = shape.dimensions[0];
              uint32_t size = shape.dimensions[1];
              uint32_t beta = operation.beta_softmax;
              for (size_t i = 0; i < batch_count; i ++) {
                const float *sub_in = src + (i * size);
                float *sub_out = des + (i * size) ;
                _base_softmax(sub_in, sub_out, size, beta) ;
              }
            }
          }    
        } else {
          id<MTLCommandBuffer> command_buffer = [GetMPSCNNContext().command_queue commandBuffer];
          const uint32_t input_idx = compilation_->inputs_[0];
          std::unique_ptr<OperandInfo>& input_data = inputs_info_[0];
          MPSImage* input_img = input_mpsimages_[0].get();
          id<MTLBuffer> input_buffer = input_mtlbuffers_[0];

          {
            memcpy([input_buffer contents], input_data->mapping.get(), input_data->length);
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            id<MTLComputePipelineState> state =
                GetMPSCNNContext().GetSpecializedPipelineState(
                    KernelFor(input_img, @"copy_nhwc_to_metal", @"copy_nhwc_to_metal_nonarray"),
                    {{ushort(input_img.height), ushort(input_img.width), ushort(input_img.featureChannels)}});
            [encoder setComputePipelineState:state];
            [encoder setBuffer:input_buffer offset:0 atIndex:0];
            [encoder setTexture:[input_img texture] atIndex:0];
            const auto& inputLaunchParams =
                SpatialPointwiseKernelLaunchParams(state, input_img);
            [encoder dispatchThreadgroups:inputLaunchParams.threadgroupsPerGrid
                    threadsPerThreadgroup:inputLaunchParams.threadsPerThreadgroup];
            [encoder endEncoding];
          }

          std::map<uint32_t, MPSTemporaryImage*> tmp_mpsimage_cache;
          for (size_t i = 0; i < compilation_->operations_.size(); i++) {
            const OperationMac& operation = compilation_->operations_[i];
            MPSCNNKernel* kernel = operation.mpscnn_kernel.get();
            if (!kernel) {
              DLOG(INFO) << "No kernel compiled for operation " << i << " type " << operation.type;
              continue;
            }
            MPSImage* src_img = nullptr;
            MPSImage* dst_img = nullptr;
            uint32_t operation_input_idx = operation.inputs[0];
            const OperandMac& operation_input = compilation_->operands_[operation_input_idx];
            if (operation_input_idx == input_idx) {
              src_img = input_img;
            }
            uint32_t operation_output_idx = operation.outputs[0];
            const OperandMac& operation_output = compilation_->operands_[operation_output_idx];
            for (size_t j = 0; j < compilation_->outputs_.size(); ++j) {
              if (operation_output_idx == compilation_->outputs_[j]) {
                dst_img = output_mpsimages_[j];
              }
            }
            if (!src_img) {
              if (tmp_mpsimage_cache.find(operation_input_idx) == tmp_mpsimage_cache.end()) {
                MPSTemporaryImage* temp_image = [MPSTemporaryImage
                    temporaryImageWithCommandBuffer:command_buffer
                    imageDescriptor:CreateMPSImageDescriptor(operation_input)];
                DLOG(INFO) << "Set readCount as " << operation_input.read_count;
                temp_image.readCount = operation_input.read_count;
                tmp_mpsimage_cache[operation_input_idx] = temp_image;
              }
              src_img = tmp_mpsimage_cache[operation_input_idx];
            }
            if (!dst_img) {
              if (tmp_mpsimage_cache.find(operation_output_idx) == tmp_mpsimage_cache.end()) {
                MPSTemporaryImage* temp_image = [MPSTemporaryImage
                    temporaryImageWithCommandBuffer:command_buffer
                    imageDescriptor:CreateMPSImageDescriptor(operation_output)];
                DLOG(INFO) << "Set readCount as " << operation_output.read_count;
                temp_image.readCount = operation_output.read_count;
                tmp_mpsimage_cache[operation_output_idx] = temp_image;
              }
              dst_img = tmp_mpsimage_cache[operation_output_idx];
            }
            DLOG(INFO) << "Encode operation " << i << " with kernel " <<
                kernel << " src " << operation_input_idx << " sourceImage " << src_img <<
                " dst " << operation_output_idx << " destinationImage " << dst_img;
            if (operation.fuse_code == mojom::FUSED_RELU1 || operation.fuse_code == mojom::FUSED_RELU6) {
              // Insert relu layer
              MPSTemporaryImage* relu_input = [MPSTemporaryImage
                    temporaryImageWithCommandBuffer:command_buffer
                    imageDescriptor:CreateMPSImageDescriptor(operation_output)];
              [kernel encodeToCommandBuffer:command_buffer
                  sourceImage:src_img
                  destinationImage:relu_input];
              short threshold = 6;
              if (operation.fuse_code == mojom::FUSED_RELU1) {
                threshold = 1;
              }
              {
                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                id<MTLComputePipelineState> state =
                    GetMPSCNNContext().GetSpecializedPipelineState(
                        KernelFor(dst_img, @"relu", @"relu_nonarray"),
                        {ushort(threshold)});
                [encoder setComputePipelineState:state];
                [encoder setTexture:[relu_input texture] atIndex:0];
                [encoder setTexture:[dst_img texture] atIndex:1];
                const auto& inputLaunchParams =
                    SpatialPointwiseKernelLaunchParams(state, relu_input);
                [encoder dispatchThreadgroups:inputLaunchParams.threadgroupsPerGrid
                        threadsPerThreadgroup:inputLaunchParams.threadsPerThreadgroup];
                [encoder endEncoding];
              }
            } else {
              [kernel encodeToCommandBuffer:command_buffer
                  sourceImage:src_img
                  destinationImage:dst_img];
            }
          }

          for (size_t i = 0; i < compilation_->outputs_.size(); ++i)
          {
            MPSImage* output_img = output_mpsimages_[i];
            id<MTLBuffer> output_buffer = output_mtlbuffers_[i];

            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            id<MTLComputePipelineState> state = GetMPSCNNContext().GetSpecializedPipelineState(
                KernelFor(output_img, @"copy_metal_to_nhwc", @"copy_metal_to_nhwc_nonarray"),
                {{ushort(output_img.height), ushort(output_img.width), ushort(output_img.featureChannels)}});
                
            [encoder setComputePipelineState:state];
            [encoder setBuffer:output_buffer offset:0 atIndex:0];
            [encoder setTexture:[output_img texture] atIndex:0];
              
            const auto& outputLaunchParams = SpatialPointwiseKernelLaunchParams(state, output_img);
            [encoder dispatchThreadgroups:outputLaunchParams.threadgroupsPerGrid
                    threadsPerThreadgroup:outputLaunchParams.threadsPerThreadgroup];
            [encoder endEncoding];
          }

          [command_buffer commit];
          [command_buffer waitUntilCompleted];

          for (size_t i = 0; i < compilation_->outputs_.size(); ++i) {
            std::unique_ptr<OperandInfo>& output_data = outputs_info_[i];
            id<MTLBuffer> output_buffer = output_mtlbuffers_[i];
            //DLOG(INFO) << "Copy memory back from output buffer with length " << output_buffer.length;
            memcpy(output_data->mapping.get(), [output_buffer contents], output_data->length);
            //OperandMac& operand = compilation_->operands_[compilation_->outputs_[i]];
            //PrintOperand(operand, output_data);
          }
        }
      }  // @autoreleasepool
    } while(0);
  }

  if (success) {
    std::move(callback).Run(mojom::NO_ERROR);
  } else {
    std::move(callback).Run(mojom::BAD_DATA);
  }
}

}  // namespace ml


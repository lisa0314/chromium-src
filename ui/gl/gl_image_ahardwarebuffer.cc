// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ui/gl/gl_image_ahardwarebuffer.h"

#include "base/android/android_hardware_buffer_compat.h"
#include "base/android/scoped_hardware_buffer_fence_sync.h"
#include "ui/gl/gl_bindings.h"
#include "ui/gl/gl_fence_android_native_fence_sync.h"
#include "ui/gl/gl_utils.h"

namespace gl {
namespace {

uint32_t GetBufferFormat(const AHardwareBuffer* buffer) {
  AHardwareBuffer_Desc desc = {};
  base::AndroidHardwareBufferCompat::GetInstance().Describe(buffer, &desc);
  return desc.format;
}

unsigned int GLInternalFormat(uint32_t buffer_format) {
  switch (buffer_format) {
    case AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM:
    case AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT:
    case AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM:
      return GL_RGBA;
    case AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM:
    case AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM:
    case AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM:
      return GL_RGB;
    default:
      // For all other buffer formats, use GL_RGBA as internal format.
      return GL_RGBA;
  }
}

void InsertGLFence(base::ScopedFD fence_fd) {
  if (!fence_fd.is_valid())
    return;

  gfx::GpuFenceHandle handle;
  handle.type = gfx::GpuFenceHandleType::kAndroidNativeFenceSync;
  handle.native_fd =
      base::FileDescriptor(fence_fd.release(), /*auto_close=*/true);
  gfx::GpuFence gpu_fence(handle);
  auto gl_fence = GLFence::CreateFromGpuFence(gpu_fence);
  gl_fence->ServerWait();
}

}  // namespace

class GLImageAHardwareBuffer::ScopedHardwareBufferFenceSyncImpl
    : public base::android::ScopedHardwareBufferFenceSync {
 public:
  ScopedHardwareBufferFenceSyncImpl(
      scoped_refptr<GLImageAHardwareBuffer> image,
      base::android::ScopedHardwareBufferHandle handle,
      base::ScopedFD fence_fd)
      : ScopedHardwareBufferFenceSync(std::move(handle), std::move(fence_fd)),
        image_(std::move(image)) {}
  ~ScopedHardwareBufferFenceSyncImpl() override = default;

  void SetReadFence(base::ScopedFD fence_fd) override {
    image_->release_fence_ =
        MergeFDs(std::move(image_->release_fence_), std::move(fence_fd));
  }

 private:
  scoped_refptr<GLImageAHardwareBuffer> image_;
};

GLImageAHardwareBuffer::GLImageAHardwareBuffer(const gfx::Size& size)
    : GLImageEGL(size) {}

GLImageAHardwareBuffer::~GLImageAHardwareBuffer() {}

bool GLImageAHardwareBuffer::Initialize(AHardwareBuffer* buffer,
                                        bool preserved) {
  handle_ = base::android::ScopedHardwareBufferHandle::Create(buffer);
  uint32_t buffer_format = GetBufferFormat(buffer);
  internal_format_ = GLInternalFormat(buffer_format);
  EGLint attribs[] = {EGL_IMAGE_PRESERVED_KHR, preserved ? EGL_TRUE : EGL_FALSE,
                      EGL_NONE};
  EGLClientBuffer client_buffer = eglGetNativeClientBufferANDROID(buffer);
  return GLImageEGL::Initialize(EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID,
                                client_buffer, attribs);
}

unsigned GLImageAHardwareBuffer::GetInternalFormat() {
  return internal_format_;
}

bool GLImageAHardwareBuffer::BindTexImage(unsigned target) {
  InsertGLFence(std::move(release_fence_));
  return GLImageEGL::BindTexImage(target);
}

bool GLImageAHardwareBuffer::CopyTexImage(unsigned target) {
  return false;
}

bool GLImageAHardwareBuffer::CopyTexSubImage(unsigned target,
                                             const gfx::Point& offset,
                                             const gfx::Rect& rect) {
  return false;
}

bool GLImageAHardwareBuffer::ScheduleOverlayPlane(
    gfx::AcceleratedWidget widget,
    int z_order,
    gfx::OverlayTransform transform,
    const gfx::Rect& bounds_rect,
    const gfx::RectF& crop_rect,
    bool enable_blend,
    std::unique_ptr<gfx::GpuFence> gpu_fence) {
  return false;
}

void GLImageAHardwareBuffer::Flush() {}

void GLImageAHardwareBuffer::OnMemoryDump(
    base::trace_event::ProcessMemoryDump* pmd,
    uint64_t process_tracing_id,
    const std::string& dump_name) {}

std::unique_ptr<base::android::ScopedHardwareBufferFenceSync>
GLImageAHardwareBuffer::GetAHardwareBuffer() {
  return std::make_unique<ScopedHardwareBufferFenceSyncImpl>(
      this, base::android::ScopedHardwareBufferHandle::Create(handle_.get()),
      std::move(release_fence_));
}

}  // namespace gl

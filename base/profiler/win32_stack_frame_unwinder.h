// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef BASE_PROFILER_WIN32_STACK_FRAME_UNWINDER_H_
#define BASE_PROFILER_WIN32_STACK_FRAME_UNWINDER_H_

#include <windows.h>

#include <memory>

#include "base/base_export.h"
#include "base/macros.h"
#include "base/sampling_heap_profiler/module_cache.h"
#include "build/build_config.h"

namespace base {

#if !defined(_WIN64)
// Allows code to compile for x86. Actual support for x86 will require either
// refactoring these interfaces or separate architecture-specific interfaces.
struct RUNTIME_FUNCTION {
  DWORD BeginAddress;
  DWORD EndAddress;
};
using PRUNTIME_FUNCTION = RUNTIME_FUNCTION*;
#endif  // !defined(_WIN64)

#if defined(ARCH_CPU_64_BITS)
inline ULONG64 ContextPC(CONTEXT* context) {
#if defined(ARCH_CPU_X86_64)
  return context->Rip;
#elif defined(ARCH_CPU_ARM64)
  return context->Pc;
#else
#error Unsupported Windows 64-bit Arch
#endif
}
#endif

// Instances of this class are expected to be created and destroyed for each
// stack unwinding. This class is not used while the target thread is suspended,
// so may allocate from the default heap.
class BASE_EXPORT Win32StackFrameUnwinder {
 public:
  // Interface for Win32 unwind-related functionality this class depends
  // on. Provides a seam for testing.
  class BASE_EXPORT UnwindFunctions {
   public:
    virtual ~UnwindFunctions();

    virtual PRUNTIME_FUNCTION LookupFunctionEntry(DWORD64 program_counter,
                                                  PDWORD64 image_base) = 0;
    virtual void VirtualUnwind(DWORD64 image_base,
                               DWORD64 program_counter,
                               PRUNTIME_FUNCTION runtime_function,
                               CONTEXT* context) = 0;

   protected:
    UnwindFunctions();

   private:
    DISALLOW_COPY_AND_ASSIGN(UnwindFunctions);
  };

  explicit Win32StackFrameUnwinder();
  ~Win32StackFrameUnwinder();

  // Attempts to unwind the frame represented by |context|, where the
  // instruction pointer is known to be in |module|. Updates |context| if
  // successful.
  bool TryUnwind(CONTEXT* context, const ModuleCache::Module* module);

 private:
  static bool TryUnwindImpl(UnwindFunctions* unwind_functions,
                            bool at_top_frame,
                            CONTEXT* context,
                            const ModuleCache::Module* module);

  // This function is for internal and test purposes only.
  Win32StackFrameUnwinder(std::unique_ptr<UnwindFunctions> unwind_functions);
  friend class Win32StackFrameUnwinderTest;

  // State associated with each stack unwinding.
  bool at_top_frame_ = true;

  std::unique_ptr<UnwindFunctions> unwind_functions_;

  DISALLOW_COPY_AND_ASSIGN(Win32StackFrameUnwinder);
};

}  // namespace base

#endif  // BASE_PROFILER_WIN32_STACK_FRAME_UNWINDER_H_

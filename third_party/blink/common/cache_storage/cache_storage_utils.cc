// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/public/common/cache_storage/cache_storage_utils.h"

#include "base/atomic_sequence_num.h"
#include "base/process/process_handle.h"

namespace blink {
namespace cache_storage {

int64_t CreateTraceId() {
  // The top 32-bits are the unique process identifier.
  int64_t id = base::GetUniqueIdForProcess();
  id <<= 32;

  // The bottom 32-bits are an atomic number sequence specific to this
  // process.
  static base::AtomicSequenceNumber seq;
  id += (seq.GetNext() & 0x0ffff);

  return id;
}

}  // namespace cache_storage
}  // namespace blink

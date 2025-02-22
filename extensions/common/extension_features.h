// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef EXTENSIONS_COMMON_EXTENSION_FEATURES_H_
#define EXTENSIONS_COMMON_EXTENSION_FEATURES_H_

#include "base/feature_list.h"

namespace extensions_features {

extern const base::Feature kBypassCorbOnlyForExtensionsAllowlist;
extern const char kBypassCorbAllowlistParamName[];
extern const base::Feature kNewExtensionUpdaterService;
extern const base::Feature kRuntimeHostPermissions;
extern const base::Feature kForceWebRequestProxyForTest;

}  // namespace extensions_features

#endif  // EXTENSIONS_COMMON_EXTENSION_FEATURES_H_

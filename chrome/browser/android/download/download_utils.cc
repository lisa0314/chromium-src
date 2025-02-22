// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/android/download/download_utils.h"

#include "base/android/jni_string.h"
#include "base/metrics/field_trial_params.h"
#include "base/strings/string_number_conversions.h"
#include "chrome/browser/android/chrome_feature_list.h"
#include "chrome/browser/download/download_offline_content_provider.h"
#include "chrome/browser/download/offline_item_utils.h"
#include "chrome/browser/offline_items_collection/offline_content_aggregator_factory.h"
#include "chrome/grit/generated_resources.h"
#include "components/download/public/common/download_utils.h"
#include "components/offline_items_collection/core/offline_content_aggregator.h"
#include "content/public/browser/browser_context.h"
#include "jni/DownloadUtils_jni.h"
#include "ui/base/l10n/l10n_util.h"

using base::android::ConvertUTF16ToJavaString;
using base::android::JavaParamRef;
using base::android::ScopedJavaLocalRef;
using OfflineContentAggregator =
    offline_items_collection::OfflineContentAggregator;

namespace {
// If received bytes is more than the size limit and resumption will restart
// from the beginning, throttle it.
int kDefaultAutoResumptionSizeLimit = 10 * 1024 * 1024;  // 10 MB
const char kAutoResumptionSizeLimitParamName[] = "AutoResumptionSizeLimit";

static DownloadOfflineContentProvider* g_download_provider = nullptr;
static DownloadOfflineContentProvider* g_download_provider_incognito = nullptr;

}  // namespace

static ScopedJavaLocalRef<jstring> JNI_DownloadUtils_GetFailStateMessage(
    JNIEnv* env,
    jint fail_state) {
  base::string16 message = OfflineItemUtils::GetFailStateMessage(
      static_cast<offline_items_collection::FailState>(fail_state));
  l10n_util::GetStringFUTF16(IDS_DOWNLOAD_STATUS_INTERRUPTED, message);
  return ConvertUTF16ToJavaString(env, message);
}

static jint JNI_DownloadUtils_GetResumeMode(
    JNIEnv* env,
    const base::android::JavaParamRef<jstring>& jurl,
    jint failState) {
  std::string url = ConvertJavaStringToUTF8(env, jurl);
  auto reason = OfflineItemUtils::ConvertFailStateToDownloadInterruptReason(
      static_cast<offline_items_collection::FailState>(failState));
  return static_cast<jint>(download::GetDownloadResumeMode(
      GURL(std::move(url)), reason, false /* restart_required */,
      true /* user_action_required */));
}

// static
base::FilePath DownloadUtils::GetUriStringForPath(
    const base::FilePath& file_path) {
  JNIEnv* env = base::android::AttachCurrentThread();
  auto uri_jstring = Java_DownloadUtils_getUriStringForPath(
      env,
      base::android::ConvertUTF8ToJavaString(env, file_path.AsUTF8Unsafe()));
  return base::FilePath(
      base::android::ConvertJavaStringToUTF8(env, uri_jstring));
}

// static
int DownloadUtils::GetAutoResumptionSizeLimit() {
  std::string value = base::GetFieldTrialParamValueByFeature(
      chrome::android::kDownloadAutoResumptionThrottling,
      kAutoResumptionSizeLimitParamName);
  int size_limit;
  return base::StringToInt(value, &size_limit)
             ? size_limit
             : kDefaultAutoResumptionSizeLimit;
}

// static
DownloadOfflineContentProvider*
DownloadUtils::GetDownloadOfflineContentProvider(
    content::BrowserContext* browser_context) {
  OfflineContentAggregator* aggregator =
      OfflineContentAggregatorFactory::GetForBrowserContext(browser_context);
  bool is_off_the_record =
      browser_context ? browser_context->IsOffTheRecord() : false;

  // Only create the provider if it is needed for the given |browser_context|.
  if (!is_off_the_record && !g_download_provider) {
    std::string name_space = OfflineContentAggregator::CreateUniqueNameSpace(
        OfflineItemUtils::GetDownloadNamespacePrefix(false), false);
    g_download_provider =
        new DownloadOfflineContentProvider(aggregator, name_space);
  }

  if (is_off_the_record && !g_download_provider_incognito) {
    std::string name_space = OfflineContentAggregator::CreateUniqueNameSpace(
        OfflineItemUtils::GetDownloadNamespacePrefix(true), true);
    g_download_provider_incognito =
        new DownloadOfflineContentProvider(aggregator, name_space);
  }

  return is_off_the_record ? g_download_provider_incognito
                           : g_download_provider;
}

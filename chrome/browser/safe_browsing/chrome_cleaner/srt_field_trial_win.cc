// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/safe_browsing/chrome_cleaner/srt_field_trial_win.h"

#include "base/metrics/field_trial.h"
#include "base/metrics/field_trial_params.h"
#include "base/metrics/histogram_macros.h"
#include "base/strings/string_util.h"
#include "base/win/windows_version.h"
#include "components/variations/variations_associated_data.h"
#include "url/origin.h"

namespace {

// Field trial strings.
constexpr char kSRTPromptOffGroup[] = "Off";
constexpr char kSRTPromptSeedParam[] = "Seed";

constexpr char kSRTElevationTrial[] = "SRTElevation";
constexpr char kSRTElevationAsNeededGroup[] = "AsNeeded";

// The download links of the Software Removal Tool.
constexpr char kDownloadRootPath[] =
    "https://dl.google.com/dl/softwareremovaltool/win/";

constexpr char kSRTX86StableDownloadURL[] =
    "https://dl.google.com/dl"
    "/softwareremovaltool/win/x86/stable/chrome_cleanup_tool.exe";

constexpr char kSRTX64StableDownloadURL[] =
    "https://dl.google.com/dl"
    "/softwareremovaltool/win/x64/stable/chrome_cleanup_tool.exe";

}  // namespace

namespace safe_browsing {

constexpr char kSRTPromptTrial[] = "SRTPromptFieldTrial";

const base::Feature kRebootPromptDialogFeature{
    "RebootPromptDialog", base::FEATURE_DISABLED_BY_DEFAULT};

const base::Feature kChromeCleanupDistributionFeature{
    "ChromeCleanupDistribution", base::FEATURE_DISABLED_BY_DEFAULT};

const base::Feature kChromeCleanupExtensionsFeature{
    "ChromeCleanupExtensions", base::FEATURE_DISABLED_BY_DEFAULT};

bool IsInSRTPromptFieldTrialGroups() {
  return !base::StartsWith(base::FieldTrialList::FindFullName(kSRTPromptTrial),
                           kSRTPromptOffGroup, base::CompareCase::SENSITIVE);
}

bool SRTPromptNeedsElevationIcon() {
  return !base::StartsWith(
      base::FieldTrialList::FindFullName(kSRTElevationTrial),
      kSRTElevationAsNeededGroup, base::CompareCase::SENSITIVE);
}

GURL GetStableDownloadURL() {
  const std::string url = base::win::OSInfo::GetArchitecture() ==
                                  base::win::OSInfo::X86_ARCHITECTURE
                              ? kSRTX86StableDownloadURL
                              : kSRTX64StableDownloadURL;
  return GURL(url);
}

GURL GetSRTDownloadURL() {
  constexpr char kCleanerDownloadGroupParam[] = "cleaner_download_group";
  const std::string download_group = base::GetFieldTrialParamValueByFeature(
      kChromeCleanupDistributionFeature, kCleanerDownloadGroupParam);
  if (download_group.empty())
    return GetStableDownloadURL();

  std::string architecture = base::win::OSInfo::GetArchitecture() ==
                                     base::win::OSInfo::X86_ARCHITECTURE
                                 ? "x86"
                                 : "x64";

  // Construct download URL using the following pattern:
  // https://dl.google.com/.../win/{arch}/{group}/chrome_cleanup_tool.exe
  std::string download_url_str = std::string(kDownloadRootPath) + architecture +
                                 "/" + download_group +
                                 "/chrome_cleanup_tool.exe";
  GURL download_url(download_url_str);

  // Ensure URL construction didn't change origin.
  const GURL download_root(kDownloadRootPath);
  const url::Origin known_good_origin = url::Origin::Create(download_root);
  url::Origin current_origin = url::Origin::Create(download_url);
  if (!current_origin.IsSameOriginWith(known_good_origin))
    return GetStableDownloadURL();

  return download_url;
}

std::string GetIncomingSRTSeed() {
  return variations::GetVariationParamValue(kSRTPromptTrial,
                                            kSRTPromptSeedParam);
}

std::string GetSRTFieldTrialGroupName() {
  return base::FieldTrialList::FindFullName(kSRTPromptTrial);
}

RebootPromptType GetRebootPromptType() {
  constexpr char kIsModalParam[] = "modal_reboot_prompt";
  if (!base::FeatureList::IsEnabled(kRebootPromptDialogFeature))
    return REBOOT_PROMPT_TYPE_OPEN_SETTINGS_PAGE;
  if (base::GetFieldTrialParamByFeatureAsBool(kRebootPromptDialogFeature,
                                              kIsModalParam,
                                              /*default_value=*/false)) {
    return REBOOT_PROMPT_TYPE_SHOW_MODAL_DIALOG;
  } else {
    return REBOOT_PROMPT_TYPE_SHOW_NON_MODAL_DIALOG;
  }
}

void RecordPromptShownWithTypeHistogram(PromptTypeHistogramValue value) {
  UMA_HISTOGRAM_ENUMERATION("SoftwareReporter.PromptShownWithType", value,
                            PROMPT_TYPE_MAX);
}

void RecordPromptNotShownWithReasonHistogram(
    NoPromptReasonHistogramValue value) {
  UMA_HISTOGRAM_ENUMERATION("SoftwareReporter.NoPromptReason", value,
                            NO_PROMPT_REASON_MAX);
}

}  // namespace safe_browsing

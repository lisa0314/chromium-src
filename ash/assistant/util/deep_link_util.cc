// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ash/assistant/util/deep_link_util.h"

#include <array>
#include <set>

#include "ash/assistant/util/i18n_util.h"
#include "base/stl_util.h"
#include "base/strings/string_split.h"
#include "base/strings/string_util.h"
#include "net/base/escape.h"
#include "net/base/url_util.h"
#include "url/gurl.h"

namespace ash {
namespace assistant {
namespace util {

namespace {

// Supported deep link param keys. These values must be kept in sync with the
// server. See more details at go/cros-assistant-deeplink.
constexpr char kIdParamKey[] = "id";
constexpr char kQueryParamKey[] = "q";
constexpr char kPageParamKey[] = "page";
constexpr char kRelaunchParamKey[] = "relaunch";

// Supported deep link prefixes. These values must be kept in sync with the
// server. See more details at go/cros-assistant-deeplink.
constexpr char kChromeSettingsPrefix[] = "googleassistant://chrome-settings";
constexpr char kAssistantFeedbackPrefix[] = "googleassistant://send-feedback";
constexpr char kAssistantListsPrefix[] = "googleassistant://lists";
constexpr char kAssistantNotesPrefix[] = "googleassistant://notes";
constexpr char kAssistantOnboardingPrefix[] = "googleassistant://onboarding";
constexpr char kAssistantQueryPrefix[] = "googleassistant://send-query";
constexpr char kAssistantRemindersPrefix[] = "googleassistant://reminders";
constexpr char kAssistantScreenshotPrefix[] =
    "googleassistant://take-screenshot";
constexpr char kAssistantSettingsPrefix[] = "googleassistant://settings";
constexpr char kAssistantTaskManagerPrefix[] = "googleassistant://task-manager";
constexpr char kAssistantWhatsOnMyScreenPrefix[] =
    "googleassistant://whats-on-my-screen";

}  // namespace

// Utilities -------------------------------------------------------------------

GURL CreateAssistantQueryDeepLink(const std::string& query) {
  return net::AppendOrReplaceQueryParameter(GURL(kAssistantQueryPrefix),
                                            kQueryParamKey, query);
}

GURL CreateAssistantSettingsDeepLink() {
  return GURL(kAssistantSettingsPrefix);
}

GURL CreateWhatsOnMyScreenDeepLink() {
  return GURL(kAssistantWhatsOnMyScreenPrefix);
}

std::map<std::string, std::string> GetDeepLinkParams(const GURL& deep_link) {
  std::map<std::string, std::string> params;

  if (!IsDeepLinkUrl(deep_link))
    return params;

  if (!deep_link.has_query())
    return params;

  // Key-value pairs are '&' delimited and the keys/values are '=' delimited.
  // Example: "googleassistant://onboarding?k1=v1&k2=v2".
  base::StringPairs pairs;
  if (!base::SplitStringIntoKeyValuePairs(deep_link.query(), '=', '&',
                                          &pairs)) {
    return params;
  }

  for (const auto& pair : pairs)
    params[pair.first] = pair.second;

  return params;
}

base::Optional<std::string> GetDeepLinkParam(
    const std::map<std::string, std::string>& params,
    DeepLinkParam param) {
  // Map of supported deep link params to their keys.
  static const std::map<DeepLinkParam, std::string> kDeepLinkParamKeys = {
      {DeepLinkParam::kId, kIdParamKey},
      {DeepLinkParam::kPage, kPageParamKey},
      {DeepLinkParam::kQuery, kQueryParamKey},
      {DeepLinkParam::kRelaunch, kRelaunchParamKey}};

  const std::string& key = kDeepLinkParamKeys.at(param);
  const auto it = params.find(key);
  return it != params.end()
             ? base::Optional<std::string>(net::UnescapeURLComponent(
                   it->second,
                   net::UnescapeRule::URL_SPECIAL_CHARS_EXCEPT_PATH_SEPARATORS |
                       net::UnescapeRule::REPLACE_PLUS_WITH_SPACE))
             : base::nullopt;
}

base::Optional<bool> GetDeepLinkParamAsBool(
    const std::map<std::string, std::string>& params,
    DeepLinkParam param) {
  const base::Optional<std::string>& value = GetDeepLinkParam(params, param);
  if (value == "true")
    return true;

  if (value == "false")
    return false;

  return base::nullopt;
}

DeepLinkType GetDeepLinkType(const GURL& url) {
  // Map of supported deep link types to their prefixes.
  static const std::map<DeepLinkType, std::string> kSupportedDeepLinks = {
      {DeepLinkType::kChromeSettings, kChromeSettingsPrefix},
      {DeepLinkType::kFeedback, kAssistantFeedbackPrefix},
      {DeepLinkType::kLists, kAssistantListsPrefix},
      {DeepLinkType::kNotes, kAssistantNotesPrefix},
      {DeepLinkType::kOnboarding, kAssistantOnboardingPrefix},
      {DeepLinkType::kQuery, kAssistantQueryPrefix},
      {DeepLinkType::kReminders, kAssistantRemindersPrefix},
      {DeepLinkType::kScreenshot, kAssistantScreenshotPrefix},
      {DeepLinkType::kSettings, kAssistantSettingsPrefix},
      {DeepLinkType::kTaskManager, kAssistantTaskManagerPrefix},
      {DeepLinkType::kWhatsOnMyScreen, kAssistantWhatsOnMyScreenPrefix}};

  for (const auto& supported_deep_link : kSupportedDeepLinks) {
    if (base::StartsWith(url.spec(), supported_deep_link.second,
                         base::CompareCase::SENSITIVE)) {
      return supported_deep_link.first;
    }
  }
  return DeepLinkType::kUnsupported;
}

bool IsDeepLinkType(const GURL& url, DeepLinkType type) {
  return GetDeepLinkType(url) == type;
}

bool IsDeepLinkUrl(const GURL& url) {
  return GetDeepLinkType(url) != DeepLinkType::kUnsupported;
}

base::Optional<GURL> GetAssistantUrl(DeepLinkType type,
                                     const base::Optional<std::string>& id) {
  std::string top_level_url;
  std::string by_id_url;

  switch (type) {
    case DeepLinkType::kLists:
      top_level_url =
          std::string("https://assistant.google.com/lists/mainview");
      by_id_url = std::string("https://assistant.google.com/lists/list/");
      break;
    case DeepLinkType::kNotes:
      top_level_url = std::string(
          "https://assistant.google.com/lists/mainview?note_tap=true");
      by_id_url = std::string("https://assistant.google.com/lists/note/");
      break;
    case DeepLinkType::kReminders:
      top_level_url =
          std::string("https://assistant.google.com/reminders/mainview");
      by_id_url = std::string("https://assistant.google.com/reminders/id/");
      break;
    default:
      NOTREACHED();
      return base::nullopt;
  }

  return (id && !id.value().empty())
             ? CreateLocalizedGURL(by_id_url + id.value())
             : CreateLocalizedGURL(top_level_url);
}

GURL GetChromeSettingsUrl(const base::Optional<std::string>& page) {
  static constexpr char kChromeSettingsUrl[] = "chrome://settings/";

  // Note that we only allow deep linking to a subset of pages. If a deep link
  // requests a page not contained in this array, we fallback gracefully to
  // top-level Chrome Settings.
  static constexpr std::array<char[16], 2> kAllowedPages = {"googleAssistant",
                                                            "languages"};

  return page && std::find(kAllowedPages.begin(), kAllowedPages.end(),
                           page.value()) != kAllowedPages.end()
             ? GURL(kChromeSettingsUrl + page.value())
             : GURL(kChromeSettingsUrl);
}

base::Optional<GURL> GetWebUrl(const GURL& deep_link) {
  return GetWebUrl(GetDeepLinkType(deep_link), GetDeepLinkParams(deep_link));
}

base::Optional<GURL> GetWebUrl(
    DeepLinkType type,
    const std::map<std::string, std::string>& params) {
  static constexpr char kAssistantSettingsWebUrl[] =
      "https://assistant.google.com/settings/mainpage";

  if (!IsWebDeepLinkType(type))
    return base::nullopt;

  switch (type) {
    case DeepLinkType::kLists:
    case DeepLinkType::kNotes:
    case DeepLinkType::kReminders: {
      const auto id = GetDeepLinkParam(params, DeepLinkParam::kId);
      return GetAssistantUrl(type, id);
    }
    case DeepLinkType::kSettings:
      return CreateLocalizedGURL(kAssistantSettingsWebUrl);
    case DeepLinkType::kUnsupported:
    case DeepLinkType::kChromeSettings:
    case DeepLinkType::kFeedback:
    case DeepLinkType::kOnboarding:
    case DeepLinkType::kQuery:
    case DeepLinkType::kScreenshot:
    case DeepLinkType::kTaskManager:
    case DeepLinkType::kWhatsOnMyScreen:
      NOTREACHED();
      return base::nullopt;
  }

  NOTREACHED();
  return base::nullopt;
}

bool IsWebDeepLink(const GURL& deep_link) {
  return IsWebDeepLinkType(GetDeepLinkType(deep_link));
}

bool IsWebDeepLinkType(DeepLinkType type) {
  // Set of deep link types which open web contents in the Assistant UI.
  static const std::set<DeepLinkType> kWebDeepLinks = {
      DeepLinkType::kLists, DeepLinkType::kNotes, DeepLinkType::kReminders,
      DeepLinkType::kSettings};

  return base::ContainsKey(kWebDeepLinks, type);
}

}  // namespace util
}  // namespace assistant
}  // namespace ash

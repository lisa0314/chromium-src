// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/browser_switcher/browser_switcher_prefs.h"

#include "base/bind.h"
#include "base/callback.h"
#include "base/metrics/histogram_macros.h"
#include "base/threading/thread_task_runner_handle.h"
#include "build/build_config.h"
#include "chrome/browser/policy/profile_policy_connector.h"
#include "chrome/browser/policy/profile_policy_connector_factory.h"
#include "chrome/browser/profiles/profile.h"
#include "components/pref_registry/pref_registry_syncable.h"
#include "components/prefs/pref_service.h"
#include "content/public/browser/browser_thread.h"

namespace browser_switcher {

RuleSet::RuleSet() = default;
RuleSet::~RuleSet() = default;

BrowserSwitcherPrefs::BrowserSwitcherPrefs(Profile* profile)
    : BrowserSwitcherPrefs(
          profile->GetPrefs(),
          policy::ProfilePolicyConnectorFactory::GetForBrowserContext(profile)
              ->policy_service()) {}

BrowserSwitcherPrefs::BrowserSwitcherPrefs(
    PrefService* prefs,
    policy::PolicyService* policy_service)
    : policy_service_(policy_service), prefs_(prefs), weak_ptr_factory_(this) {
  filtering_change_registrar_.Init(prefs_);

  const struct {
    const char* pref_name;
    base::RepeatingCallback<void(BrowserSwitcherPrefs*)> callback;
  } hooks[] = {
    {prefs::kAlternativeBrowserPath,
     base::BindRepeating(&BrowserSwitcherPrefs::AlternativeBrowserPathChanged)},
    {prefs::kAlternativeBrowserParameters,
     base::BindRepeating(
         &BrowserSwitcherPrefs::AlternativeBrowserParametersChanged)},
    {prefs::kUrlList,
     base::BindRepeating(&BrowserSwitcherPrefs::UrlListChanged)},
    {prefs::kUrlGreylist,
     base::BindRepeating(&BrowserSwitcherPrefs::GreylistChanged)},
#if defined(OS_WIN)
    {prefs::kChromePath,
     base::BindRepeating(&BrowserSwitcherPrefs::ChromePathChanged)},
    {prefs::kChromeParameters,
     base::BindRepeating(&BrowserSwitcherPrefs::ChromeParametersChanged)},
#endif
  };

  // Listen for pref changes, and run all the hooks once to initialize state.
  for (const auto& hook : hooks) {
    auto callback = base::BindRepeating(hook.callback, base::Unretained(this));
    filtering_change_registrar_.Add(hook.pref_name, callback);
    callback.Run();
  }

  // When any pref changes, mark this object as 'dirty' for the purpose of
  // triggering observers.
  notifying_change_registrar_.Init(prefs_);
  const char* all_prefs[] = {
    prefs::kEnabled,
    prefs::kAlternativeBrowserPath,
    prefs::kAlternativeBrowserParameters,
    prefs::kKeepLastTab,
    prefs::kUrlList,
    prefs::kUrlGreylist,
    prefs::kExternalSitelistUrl,
#if defined(OS_WIN)
    prefs::kUseIeSitelist,
    prefs::kChromePath,
    prefs::kChromeParameters,
#endif
  };
  for (const char* pref_name : all_prefs) {
    notifying_change_registrar_.Add(
        pref_name, base::BindRepeating(&BrowserSwitcherPrefs::MarkDirty,
                                       base::Unretained(this)));
  }

  if (policy_service_)
    policy_service_->AddObserver(policy::POLICY_DOMAIN_CHROME, this);
}

BrowserSwitcherPrefs::~BrowserSwitcherPrefs() = default;

void BrowserSwitcherPrefs::Shutdown() {
  if (policy_service_)
    policy_service_->RemoveObserver(policy::POLICY_DOMAIN_CHROME, this);
}

// static
void BrowserSwitcherPrefs::RegisterProfilePrefs(
    user_prefs::PrefRegistrySyncable* registry) {
  registry->RegisterBooleanPref(prefs::kEnabled, false);
  registry->RegisterIntegerPref(prefs::kDelay, 0);
  registry->RegisterStringPref(prefs::kAlternativeBrowserPath, "");
  registry->RegisterListPref(prefs::kAlternativeBrowserParameters);
  registry->RegisterBooleanPref(prefs::kKeepLastTab, true);
  registry->RegisterListPref(prefs::kUrlList);
  registry->RegisterListPref(prefs::kUrlGreylist);
  registry->RegisterStringPref(prefs::kExternalSitelistUrl, "");
#if defined(OS_WIN)
  registry->RegisterBooleanPref(prefs::kUseIeSitelist, false);
  registry->RegisterStringPref(prefs::kChromePath, "");
  registry->RegisterListPref(prefs::kChromeParameters);
#endif
}

bool BrowserSwitcherPrefs::IsEnabled() const {
  return prefs_->GetBoolean(prefs::kEnabled) &&
         prefs_->IsManagedPreference(prefs::kEnabled);
}

const std::string& BrowserSwitcherPrefs::GetAlternativeBrowserPath() const {
  return alt_browser_path_;
}

const std::vector<std::string>&
BrowserSwitcherPrefs::GetAlternativeBrowserParameters() const {
  return alt_browser_params_;
}

bool BrowserSwitcherPrefs::KeepLastTab() const {
  return prefs_->GetBoolean(prefs::kKeepLastTab);
}

int BrowserSwitcherPrefs::GetDelay() const {
  return prefs_->GetInteger(prefs::kDelay);
}

const RuleSet& BrowserSwitcherPrefs::GetRules() const {
  return rules_;
}

GURL BrowserSwitcherPrefs::GetExternalSitelistUrl() const {
  if (!prefs_->IsManagedPreference(prefs::kExternalSitelistUrl))
    return GURL();
  return GURL(prefs_->GetString(prefs::kExternalSitelistUrl));
}

#if defined(OS_WIN)
bool BrowserSwitcherPrefs::UseIeSitelist() const {
  if (!prefs_->IsManagedPreference(prefs::kUseIeSitelist))
    return false;
  return prefs_->GetBoolean(prefs::kUseIeSitelist);
}

const std::string& BrowserSwitcherPrefs::GetChromePath() const {
  return chrome_path_;
}

const std::vector<std::string>& BrowserSwitcherPrefs::GetChromeParameters()
    const {
  return chrome_params_;
}
#endif

void BrowserSwitcherPrefs::OnPolicyUpdated(const policy::PolicyNamespace& ns,
                                           const policy::PolicyMap& previous,
                                           const policy::PolicyMap& current) {
  // Let all the other policy observers run first, so that prefs are up-to-date
  // when we run our own callbacks.
  base::ThreadTaskRunnerHandle::Get()->PostTask(
      FROM_HERE, base::BindOnce(&BrowserSwitcherPrefs::RunCallbacksIfDirty,
                                weak_ptr_factory_.GetWeakPtr()));
}

std::unique_ptr<BrowserSwitcherPrefs::CallbackSubscription>
BrowserSwitcherPrefs::RegisterPrefsChangedCallback(
    BrowserSwitcherPrefs::PrefsChangedCallback cb) {
  return callback_list_.Add(cb);
}

void BrowserSwitcherPrefs::RunCallbacksIfDirty() {
  DCHECK_CURRENTLY_ON(content::BrowserThread::UI);
  if (dirty_)
    callback_list_.Notify(this);
  dirty_ = false;
}

void BrowserSwitcherPrefs::MarkDirty() {
  dirty_ = true;
}

void BrowserSwitcherPrefs::AlternativeBrowserPathChanged() {
  alt_browser_path_.clear();
  if (prefs_->IsManagedPreference(prefs::kAlternativeBrowserPath))
    alt_browser_path_ = prefs_->GetString(prefs::kAlternativeBrowserPath);
}

void BrowserSwitcherPrefs::AlternativeBrowserParametersChanged() {
  alt_browser_params_.clear();
  if (!prefs_->IsManagedPreference(prefs::kAlternativeBrowserParameters))
    return;
  const base::ListValue* params =
      prefs_->GetList(prefs::kAlternativeBrowserParameters);
  for (const auto& param : *params) {
    std::string param_string = param.GetString();
    alt_browser_params_.push_back(param_string);
  }
}

void BrowserSwitcherPrefs::UrlListChanged() {
  rules_.sitelist.clear();

  if (!prefs_->IsManagedPreference(prefs::kUrlList))
    return;

  UMA_HISTOGRAM_COUNTS_100000(
      "BrowserSwitcher.UrlListSize",
      prefs_->GetList(prefs::kUrlList)->GetList().size());

  bool has_wildcard = false;
  for (const auto& url : *prefs_->GetList(prefs::kUrlList)) {
    rules_.sitelist.push_back(url.GetString());
    if (url.GetString() == "*")
      has_wildcard = true;
  }

  UMA_HISTOGRAM_BOOLEAN("BrowserSwitcher.UrlListWildcard", has_wildcard);
}

void BrowserSwitcherPrefs::GreylistChanged() {
  rules_.greylist.clear();

  // This pref is sensitive. Only set through policies.
  if (!prefs_->IsManagedPreference(prefs::kUrlGreylist))
    return;

  UMA_HISTOGRAM_COUNTS_100000(
      "BrowserSwitcher.GreylistSize",
      prefs_->GetList(prefs::kUrlGreylist)->GetList().size());

  bool has_wildcard = false;
  for (const auto& url : *prefs_->GetList(prefs::kUrlGreylist)) {
    rules_.greylist.push_back(url.GetString());
    if (url.GetString() == "*")
      has_wildcard = true;
  }

  UMA_HISTOGRAM_BOOLEAN("BrowserSwitcher.UrlListWildcard", has_wildcard);
}

#if defined(OS_WIN)
void BrowserSwitcherPrefs::ChromePathChanged() {
  chrome_path_.clear();
  if (prefs_->IsManagedPreference(prefs::kChromePath))
    chrome_path_ = prefs_->GetString(prefs::kChromePath);
}

void BrowserSwitcherPrefs::ChromeParametersChanged() {
  chrome_params_.clear();
  if (!prefs_->IsManagedPreference(prefs::kChromeParameters))
    return;
  const base::ListValue* params = prefs_->GetList(prefs::kChromeParameters);
  for (const auto& param : *params) {
    std::string param_string = param.GetString();
    chrome_params_.push_back(param_string);
  }
}
#endif

namespace prefs {

// Path to the executable of the alternative browser, or one of "${chrome}",
// "${ie}", "${firefox}", "${opera}", "${safari}".
const char kAlternativeBrowserPath[] =
    "browser_switcher.alternative_browser_path";

// Arguments to pass to the alternative browser when invoking it via
// |ShellExecute()|.
const char kAlternativeBrowserParameters[] =
    "browser_switcher.alternative_browser_parameters";

// If true, always keep at least one tab open after switching.
const char kKeepLastTab[] = "browser_switcher.keep_last_tab";

// List of host domain names to be opened in an alternative browser.
const char kUrlList[] = "browser_switcher.url_list";

// List of hosts that should not trigger a transition in either browser.
const char kUrlGreylist[] = "browser_switcher.url_greylist";

// URL with an external XML sitelist file to load.
const char kExternalSitelistUrl[] = "browser_switcher.external_sitelist_url";

#if defined(OS_WIN)
// If set to true, use the IE Enterprise Mode Sitelist policy.
const char kUseIeSitelist[] = "browser_switcher.use_ie_sitelist";

// Path to the Chrome executable for the alternative browser.
const char kChromePath[] = "browser_switcher.chrome_path";

// Arguments the alternative browser should pass to Chrome when launching it.
const char kChromeParameters[] = "browser_switcher.chrome_parameters";
#endif

// Disable browser_switcher unless this is set to true.
const char kEnabled[] = "browser_switcher.enabled";

// How long to wait on chrome://browser-switch (milliseconds).
const char kDelay[] = "browser_switcher.delay";

}  // namespace prefs
}  // namespace browser_switcher

// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/notifications/notification_trigger_scheduler.h"

#include <memory>

#include "chrome/browser/browser_process.h"
#include "chrome/browser/notifications/platform_notification_service_impl.h"
#include "chrome/browser/profiles/profile.h"
#include "chrome/browser/profiles/profile_manager.h"
#include "chrome/common/pref_names.h"
#include "components/prefs/pref_service.h"
#include "content/public/browser/platform_notification_context.h"
#include "content/public/browser/storage_partition.h"

using content::BrowserContext;
using content::BrowserThread;

namespace {

void TriggerNotificationsForProfile(Profile* profile) {
  auto* service = PlatformNotificationServiceImpl::GetInstance();
  base::Time next_trigger = service->ReadNextTriggerTimestamp(profile);

  // Skip this profile if there are no pending notifications.
  if (next_trigger > base::Time::Now()) {
    // Reschedule in case there are some in the future.
    if (next_trigger < base::Time::Max())
      service->ScheduleTrigger(profile, next_trigger);
    return;
  }

  // Reset the next trigger time. It will be set again if there are more
  // scheduled notifications for any storage partition of this profile.
  profile->GetPrefs()->SetTime(prefs::kNotificationNextTriggerTime,
                               base::Time::Max());

  // Unretained is safe here as PlatformNotificationServiceImpl is a singleton
  // and owns its |trigger_scheduler_| until process exit.
  BrowserContext::ForEachStoragePartition(
      profile,
      base::BindRepeating(
          &NotificationTriggerScheduler::
              TriggerNotificationsForStoragePartition,
          base::Unretained(service->GetNotificationTriggerScheduler())));
}

}  // namespace

// static
std::unique_ptr<NotificationTriggerScheduler>
NotificationTriggerScheduler::Create() {
  return base::WrapUnique(new NotificationTriggerScheduler());
}

// static
void NotificationTriggerScheduler::TriggerNotifications() {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  // Skip if the browser process is already in shutdown path.
  if (!g_browser_process || g_browser_process->IsShuttingDown())
    return;
  auto profiles = g_browser_process->profile_manager()->GetLoadedProfiles();
  for (Profile* profile : profiles) {
    TriggerNotificationsForProfile(profile);
    // Notifications are technically not supported in Incognito, but in case we
    // ever change that lets handle these profiles too.
    if (profile->HasOffTheRecordProfile())
      TriggerNotificationsForProfile(profile->GetOffTheRecordProfile());
  }
}

NotificationTriggerScheduler::NotificationTriggerScheduler()
    : weak_ptr_factory_(this) {}

NotificationTriggerScheduler::~NotificationTriggerScheduler() = default;

void NotificationTriggerScheduler::ScheduleTrigger(base::Time timestamp) {
  base::TimeDelta delay = timestamp - base::Time::Now();
  if (delay.InMicroseconds() < 0)
    delay = base::TimeDelta();

  if (trigger_timer_.IsRunning() && trigger_timer_.GetCurrentDelay() <= delay)
    return;

  trigger_timer_.Start(
      FROM_HERE, delay,
      base::BindOnce(&NotificationTriggerScheduler::TriggerNotifications));
}

void NotificationTriggerScheduler::TriggerNotificationsForStoragePartition(
    content::StoragePartition* partition) {
  partition->GetPlatformNotificationContext()->TriggerNotifications();
}

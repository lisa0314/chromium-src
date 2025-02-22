// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package org.chromium.chrome.browser.webapps;

import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.support.v4.app.NotificationCompat;

import org.chromium.base.ContextUtils;
import org.chromium.base.metrics.RecordUserAction;
import org.chromium.blink_public.platform.WebDisplayMode;
import org.chromium.chrome.R;
import org.chromium.chrome.browser.ChromeFeatureList;
import org.chromium.chrome.browser.IntentHandler;
import org.chromium.chrome.browser.notifications.ChromeNotification;
import org.chromium.chrome.browser.notifications.NotificationBuilderFactory;
import org.chromium.chrome.browser.notifications.NotificationConstants;
import org.chromium.chrome.browser.notifications.NotificationManagerProxy;
import org.chromium.chrome.browser.notifications.NotificationManagerProxyImpl;
import org.chromium.chrome.browser.notifications.NotificationMetadata;
import org.chromium.chrome.browser.notifications.NotificationUmaTracker;
import org.chromium.chrome.browser.notifications.PendingIntentProvider;
import org.chromium.chrome.browser.notifications.channels.ChannelDefinitions;
import org.chromium.chrome.browser.tab.Tab;
import org.chromium.chrome.browser.util.IntentUtils;
import org.chromium.ui.base.Clipboard;

import java.lang.ref.WeakReference;

/**
 * Manages the notification shown by Chrome when running standalone Web Apps. It accomplishes
 * number of goals:
 * - Presents the current URL.
 * - Exposes 'Share' and 'Open in Chrome' actions.
 * - Messages that Web App runs in Chrome.
 */
class WebappActionsNotificationManager {
    private static final String ACTION_SHARE =
            "org.chromium.chrome.browser.webapps.NOTIFICATION_ACTION_SHARE";
    private static final String ACTION_OPEN_IN_CHROME =
            "org.chromium.chrome.browser.webapps.NOTIFICATION_ACTION_OPEN_IN_CHROME";
    private static final String ACTION_FOCUS =
            "org.chromium.chrome.browser.webapps.NOTIFICATION_ACTION_FOCUS";

    static boolean isEnabled() {
        return ChromeFeatureList.isEnabled(ChromeFeatureList.PWA_PERSISTENT_NOTIFICATION);
    }

    public static void maybeShowNotification(Tab tab, WebappInfo webappInfo) {
        if (!isEnabled() || tab == null) return;

        // All features provided by the notification are also available in the minimal-ui toolbar.
        if (webappInfo.displayMode() == WebDisplayMode.MINIMAL_UI) {
            return;
        }

        Context appContext = ContextUtils.getApplicationContext();
        ChromeNotification notification = createNotification(appContext, tab, webappInfo);
        NotificationManagerProxy nm = new NotificationManagerProxyImpl(appContext);
        nm.notify(notification);

        NotificationUmaTracker.getInstance().onNotificationShown(
                NotificationUmaTracker.SystemNotificationType.WEBAPP_ACTIONS,
                notification.getNotification());
    }

    private static ChromeNotification createNotification(
            Context appContext, Tab tab, WebappInfo webappInfo) {
        // The pending intents target an activity (instead of a service or a broadcast receiver) so
        // that the notification tray closes when a user taps the one of the notification action
        // links.
        PendingIntentProvider focusIntent =
                createPendingIntentWithAction(appContext, tab, ACTION_FOCUS);
        PendingIntentProvider openInChromeIntent =
                createPendingIntentWithAction(appContext, tab, ACTION_OPEN_IN_CHROME);
        PendingIntentProvider shareIntent =
                createPendingIntentWithAction(appContext, tab, ACTION_SHARE);

        NotificationMetadata metadata = new NotificationMetadata(
                NotificationUmaTracker.SystemNotificationType.WEBAPP_ACTIONS,
                null /* notificationTag */, NotificationConstants.NOTIFICATION_ID_WEBAPP_ACTIONS);
        return NotificationBuilderFactory
                .createChromeNotificationBuilder(true /* prefer compat */,
                        ChannelDefinitions.ChannelId.WEBAPP_ACTIONS,
                        null /* remoteAppPackageName */, metadata)
                .setSmallIcon(R.drawable.ic_chrome)
                .setContentTitle(webappInfo.shortName())
                .setContentText(appContext.getString(R.string.webapp_tap_to_copy_url))
                .setShowWhen(false)
                .setAutoCancel(false)
                .setOngoing(true)
                .setPriorityBeforeO(NotificationCompat.PRIORITY_MIN)
                .setContentIntent(focusIntent)
                .addAction(R.drawable.ic_share_white_24dp,
                        appContext.getResources().getString(R.string.share), shareIntent,
                        NotificationUmaTracker.ActionType.WEB_APP_ACTION_SHARE)
                .addAction(R.drawable.ic_exit_to_app_white_24dp,
                        appContext.getResources().getString(R.string.menu_open_in_chrome),
                        openInChromeIntent,
                        NotificationUmaTracker.ActionType.WEB_APP_ACTION_OPEN_IN_CHROME)
                .buildChromeNotification();
    }

    /** Creates an intent which targets {@link WebappLauncherActivity} with the passed-in action. */
    private static PendingIntentProvider createPendingIntentWithAction(
            Context context, Tab tab, String action) {
        Intent intent = new Intent(action);
        intent.setClass(context, WebappLauncherActivity.class);
        intent.putExtra(IntentHandler.EXTRA_TAB_ID, tab.getId());
        IntentHandler.addTrustedIntentExtras(intent);
        return PendingIntentProvider.getActivity(context, 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_ONE_SHOT);
    }

    public static void cancelNotification() {
        if (!isEnabled()) return;
        NotificationManager nm =
                (NotificationManager) ContextUtils.getApplicationContext().getSystemService(
                        Context.NOTIFICATION_SERVICE);
        nm.cancel(NotificationConstants.NOTIFICATION_ID_WEBAPP_ACTIONS);
    }

    public static boolean handleNotificationAction(Intent intent) {
        if (!IntentHandler.wasIntentSenderChrome(intent)) return false;

        int tabId =
                IntentUtils.safeGetIntExtra(intent, IntentHandler.EXTRA_TAB_ID, Tab.INVALID_TAB_ID);
        WeakReference<WebappActivity> webappActivityRef =
                WebappActivity.findWebappActivityWithTabId(tabId);
        if (webappActivityRef == null) return false;

        WebappActivity webappActivity = webappActivityRef.get();
        if (webappActivity == null) return false;

        if (ACTION_SHARE.equals(intent.getAction())) {
            // Not routing through onMenuOrKeyboardAction to control UMA String.
            webappActivity.onShareMenuItemSelected(
                    false /* share directly */, webappActivity.getCurrentTabModel().isIncognito());
            RecordUserAction.record("Webapp.NotificationShare");
            return true;
        } else if (ACTION_OPEN_IN_CHROME.equals(intent.getAction())) {
            webappActivity.onMenuOrKeyboardAction(R.id.open_in_browser_id, false /* fromMenu */);
            return true;
        } else if (ACTION_FOCUS.equals(intent.getAction())) {
            Tab tab = webappActivity.getActivityTab();
            if (tab != null) Clipboard.getInstance().copyUrlToClipboard(tab.getOriginalUrl());
            RecordUserAction.record("Webapp.NotificationFocused");
            return true;
        }
        return false;
    }
}

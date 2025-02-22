// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/chromeos/set_time_dialog.h"

#include "base/metrics/user_metrics.h"
#include "base/strings/string16.h"
#include "chrome/common/url_constants.h"
#include "chromeos/constants/chromeos_features.h"
#include "chromeos/login/login_state/login_state.h"
#include "ui/gfx/geometry/size.h"

namespace chromeos {

namespace {

const int kDefaultWidth = 490;
const int kDefaultHeight = 235;

// Material design dialog width and height in DIPs.
const int kDefaultWidthMd = 530;
const int kDefaultHeightWithTimezone = 255;
const int kDefaultHeightWithoutTimezone = 215;

}  // namespace

// static
void SetTimeDialog::ShowDialog(gfx::NativeWindow parent) {
  base::RecordAction(base::UserMetricsAction("Options_SetTimeDialog_Show"));
  auto* dialog = new SetTimeDialog();
  dialog->ShowSystemDialog(parent);
}

// static
bool SetTimeDialog::ShouldShowTimezone() {
  // After login the user should set the timezone via Settings, which applies
  // additional restrictions.
  return !LoginState::Get()->IsUserLoggedIn();
}

SetTimeDialog::SetTimeDialog()
    : SystemWebDialogDelegate(GURL(chrome::kChromeUISetTimeURL),
                              base::string16() /* title */) {}

SetTimeDialog::~SetTimeDialog() = default;

void SetTimeDialog::GetDialogSize(gfx::Size* size) const {
  if (features::IsSetTimeDialogMd()) {
    size->SetSize(kDefaultWidthMd, ShouldShowTimezone()
                                       ? kDefaultHeightWithTimezone
                                       : kDefaultHeightWithoutTimezone);
  } else {
    size->SetSize(kDefaultWidth, kDefaultHeight);
  }
}

}  // namespace chromeos

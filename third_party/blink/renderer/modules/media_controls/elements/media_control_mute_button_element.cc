// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/media_controls/elements/media_control_mute_button_element.h"

#include "third_party/blink/public/platform/platform.h"
#include "third_party/blink/renderer/core/dom/events/event.h"
#include "third_party/blink/renderer/core/html/media/html_media_element.h"
#include "third_party/blink/renderer/core/input_type_names.h"
#include "third_party/blink/renderer/modules/media_controls/media_controls_impl.h"
#include "third_party/blink/renderer/platform/text/platform_locale.h"

namespace blink {

MediaControlMuteButtonElement::MediaControlMuteButtonElement(
    MediaControlsImpl& media_controls)
    : MediaControlInputElement(media_controls, kMediaIgnore) {
  setType(input_type_names::kButton);
  SetShadowPseudoId(AtomicString("-webkit-media-controls-mute-button"));
}

bool MediaControlMuteButtonElement::WillRespondToMouseClickEvents() {
  return true;
}

void MediaControlMuteButtonElement::UpdateDisplayType() {
  // TODO(mlamouri): checking for volume == 0 because the mute button will look
  // 'muted' when the volume is 0 even if the element is not muted. This allows
  // the painting and the display type to actually match.
  bool muted = MediaElement().muted() || MediaElement().volume() == 0;
  setAttribute(html_names::kAriaLabelAttr,
               WTF::AtomicString(GetLocale().QueryString(
                   muted ? WebLocalizedString::kAXMediaUnMuteButton
                         : WebLocalizedString::kAXMediaMuteButton)));
  SetClass("muted", muted);
  UpdateOverflowString();

  MediaControlInputElement::UpdateDisplayType();
}

WebLocalizedString::Name MediaControlMuteButtonElement::GetOverflowStringName()
    const {
  if (MediaElement().muted())
    return WebLocalizedString::kOverflowMenuUnmute;
  return WebLocalizedString::kOverflowMenuMute;
}

bool MediaControlMuteButtonElement::HasOverflowButton() const {
  return true;
}

bool MediaControlMuteButtonElement::IsControlPanelButton() const {
  return true;
}

const char* MediaControlMuteButtonElement::GetNameForHistograms() const {
  return IsOverflowElement() ? "MuteOverflowButton" : "MuteButton";
}

void MediaControlMuteButtonElement::DefaultEventHandler(Event& event) {
  if (!IsDisabled() && (event.type() == event_type_names::kClick ||
                        event.type() == event_type_names::kGesturetap)) {
    if (MediaElement().muted()) {
      Platform::Current()->RecordAction(
          UserMetricsAction("Media.Controls.Unmute"));
    } else {
      Platform::Current()->RecordAction(
          UserMetricsAction("Media.Controls.Mute"));
    }

    MediaElement().setMuted(!MediaElement().muted());
    event.SetDefaultHandled();
  }

  if (!IsOverflowElement()) {
    if (event.type() == event_type_names::kFocus)
      GetMediaControls().OpenVolumeSliderIfNecessary();

    if (event.type() == event_type_names::kBlur)
      GetMediaControls().CloseVolumeSliderIfNecessary();
  }

  MediaControlInputElement::DefaultEventHandler(event);
}

}  // namespace blink

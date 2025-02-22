// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_MEDIA_HARDWARE_KEY_MEDIA_CONTROLLER_H_
#define CONTENT_BROWSER_MEDIA_HARDWARE_KEY_MEDIA_CONTROLLER_H_

#include <utility>
#include <vector>

#include "base/containers/flat_set.h"
#include "base/optional.h"
#include "content/common/content_export.h"
#include "mojo/public/cpp/bindings/binding.h"
#include "services/media_session/public/mojom/media_controller.mojom.h"
#include "ui/base/accelerators/media_keys_listener.h"
#include "ui/events/keycodes/keyboard_codes.h"

namespace service_manager {
class Connector;
}  // namespace service_manager

namespace content {

// HardwareKeyMediaController controls media sessions via hardware media keys.
class CONTENT_EXPORT HardwareKeyMediaController
    : public media_session::mojom::MediaControllerObserver,
      public ui::MediaKeysListener::Delegate {
 public:
  explicit HardwareKeyMediaController(service_manager::Connector* connector);
  ~HardwareKeyMediaController() override;

  // media_session::mojom::MediaControllerObserver:
  void MediaSessionInfoChanged(
      media_session::mojom::MediaSessionInfoPtr session_info) override;
  void MediaSessionMetadataChanged(
      const base::Optional<media_session::MediaMetadata>& metadata) override {}
  void MediaSessionActionsChanged(
      const std::vector<media_session::mojom::MediaSessionAction>& actions)
      override;
  void MediaSessionChanged(
      const base::Optional<base::UnguessableToken>& request_id) override {}

  // ui::MediaKeysListener::Delegate:
  void OnMediaKeysAccelerator(const ui::Accelerator& accelerator) override;

  void FlushForTesting();
  void SetMediaControllerForTesting(
      media_session::mojom::MediaControllerPtr controller) {
    media_controller_ptr_ = std::move(controller);
  }

 private:
  // These values are persisted to logs. Entries should not be renumbered and
  // numeric values should never be reused.
  enum class MediaHardwareKeyAction {
    kActionPlay = 0,
    kActionPause,
    kActionStop,
    kActionNextTrack,
    kActionPreviousTrack,
    kMaxValue = kActionPreviousTrack
  };

  // Used for converting between MediaSessionAction and KeyboardCode.
  media_session::mojom::MediaSessionAction KeyCodeToMediaSessionAction(
      ui::KeyboardCode key_code) const;

  // Returns nullopt if the action is not supported via hardware keys (e.g.
  // SeekBackward).
  base::Optional<ui::KeyboardCode> MediaSessionActionToKeyCode(
      media_session::mojom::MediaSessionAction action) const;

  bool SupportsAction(media_session::mojom::MediaSessionAction action) const;
  void PerformAction(media_session::mojom::MediaSessionAction action);
  void RecordAction(MediaHardwareKeyAction action);

  // Used to control the active session.
  media_session::mojom::MediaControllerPtr media_controller_ptr_;

  // Used to check whether a play/pause key should play or pause (based on
  // current playback state).
  media_session::mojom::MediaSessionInfoPtr session_info_;

  // Used to check which actions are currently supported.
  base::flat_set<media_session::mojom::MediaSessionAction> actions_;

  // Used to receive updates to the active media controller.
  mojo::Binding<media_session::mojom::MediaControllerObserver>
      media_controller_observer_binding_{this};

  DISALLOW_COPY_AND_ASSIGN(HardwareKeyMediaController);
};

}  // namespace content

#endif  // CONTENT_BROWSER_MEDIA_HARDWARE_KEY_MEDIA_CONTROLLER_H_

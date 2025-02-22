// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/exo/wayland/wayland_keyboard_delegate.h"

#include <wayland-server-core.h>
#include <wayland-server-protocol-core.h>

#include "base/containers/flat_map.h"

namespace exo {
namespace wayland {

#if BUILDFLAG(USE_XKBCOMMON)

WaylandKeyboardDelegate::WaylandKeyboardDelegate(wl_resource* keyboard_resource)
    : keyboard_resource_(keyboard_resource),
      xkb_context_(xkb_context_new(XKB_CONTEXT_NO_FLAGS)) {
#if defined(OS_CHROMEOS)
  ash::ImeController* ime_controller = ash::Shell::Get()->ime_controller();
  ime_controller->AddObserver(this);
  SendNamedLayout(ime_controller->keyboard_layout_name());
#else
  SendLayout(nullptr);
#endif
}

#if defined(OS_CHROMEOS)
WaylandKeyboardDelegate::~WaylandKeyboardDelegate() {
  ash::Shell::Get()->ime_controller()->RemoveObserver(this);
}
#endif

void WaylandKeyboardDelegate::OnKeyboardDestroying(Keyboard* keyboard) {
  delete this;
}

bool WaylandKeyboardDelegate::CanAcceptKeyboardEventsForSurface(
    Surface* surface) const {
  wl_resource* surface_resource = GetSurfaceResource(surface);
  // We can accept events for this surface if the client is the same as the
  // keyboard.
  return surface_resource &&
         wl_resource_get_client(surface_resource) == client();
}

void WaylandKeyboardDelegate::OnKeyboardEnter(
    Surface* surface,
    const base::flat_map<ui::DomCode, ui::DomCode>& pressed_keys) {
  wl_resource* surface_resource = GetSurfaceResource(surface);
  DCHECK(surface_resource);
  wl_array keys;
  wl_array_init(&keys);
  for (const auto& entry : pressed_keys) {
    uint32_t* value =
        static_cast<uint32_t*>(wl_array_add(&keys, sizeof(uint32_t)));
    DCHECK(value);
    *value = DomCodeToKey(entry.second);
  }
  wl_keyboard_send_enter(keyboard_resource_, next_serial(), surface_resource,
                         &keys);
  wl_array_release(&keys);
  wl_client_flush(client());
}

void WaylandKeyboardDelegate::OnKeyboardLeave(Surface* surface) {
  wl_resource* surface_resource = GetSurfaceResource(surface);
  DCHECK(surface_resource);
  wl_keyboard_send_leave(keyboard_resource_, next_serial(), surface_resource);
  wl_client_flush(client());
}

uint32_t WaylandKeyboardDelegate::OnKeyboardKey(base::TimeTicks time_stamp,
                                                ui::DomCode key,
                                                bool pressed) {
  uint32_t serial = next_serial();
  SendTimestamp(time_stamp);
  wl_keyboard_send_key(
      keyboard_resource_, serial, TimeTicksToMilliseconds(time_stamp),
      DomCodeToKey(key),
      pressed ? WL_KEYBOARD_KEY_STATE_PRESSED : WL_KEYBOARD_KEY_STATE_RELEASED);
  wl_client_flush(client());
  return serial;
}

void WaylandKeyboardDelegate::OnKeyboardModifiers(int modifier_flags) {
  // CrOS treats numlock as always on, but its event flags actually have that
  // key disabled, (i.e. chromeos apps specially handle numpad key events as
  // though numlock is on). In order to get the same result from the linux apps,
  // we need to ensure they always treat numlock as on.
  modifier_flags |= ui::EF_NUM_LOCK_ON;
  xkb_state_update_mask(xkb_state_.get(),
                        ModifierFlagsToXkbModifiers(modifier_flags), 0, 0, 0, 0,
                        0);
  wl_keyboard_send_modifiers(
      keyboard_resource_, next_serial(),
      xkb_state_serialize_mods(xkb_state_.get(), XKB_STATE_MODS_DEPRESSED),
      xkb_state_serialize_mods(xkb_state_.get(), XKB_STATE_MODS_LOCKED),
      xkb_state_serialize_mods(xkb_state_.get(), XKB_STATE_MODS_LATCHED),
      xkb_state_serialize_layout(xkb_state_.get(), XKB_STATE_LAYOUT_EFFECTIVE));
  wl_client_flush(client());
}

#if defined(OS_CHROMEOS)
void WaylandKeyboardDelegate::OnCapsLockChanged(bool enabled) {}

void WaylandKeyboardDelegate::OnKeyboardLayoutNameChanged(
    const std::string& layout_name) {
  SendNamedLayout(layout_name);
}
#endif

uint32_t WaylandKeyboardDelegate::DomCodeToKey(ui::DomCode code) const {
  // This assumes KeycodeConverter has been built with evdev/xkb codes.
  xkb_keycode_t xkb_keycode = static_cast<xkb_keycode_t>(
      ui::KeycodeConverter::DomCodeToNativeKeycode(code));

  // Keycodes are offset by 8 in Xkb.
  DCHECK_GE(xkb_keycode, 8u);
  return xkb_keycode - 8;
}

uint32_t WaylandKeyboardDelegate::ModifierFlagsToXkbModifiers(
    int modifier_flags) {
  struct {
    ui::EventFlags flag;
    const char* xkb_name;
  } modifiers[] = {
      {ui::EF_SHIFT_DOWN, XKB_MOD_NAME_SHIFT},
      {ui::EF_CONTROL_DOWN, XKB_MOD_NAME_CTRL},
      {ui::EF_ALT_DOWN, XKB_MOD_NAME_ALT},
      {ui::EF_COMMAND_DOWN, XKB_MOD_NAME_LOGO},
      {ui::EF_ALTGR_DOWN, "Mod5"},
      {ui::EF_MOD3_DOWN, "Mod3"},
      {ui::EF_NUM_LOCK_ON, XKB_MOD_NAME_NUM},
      {ui::EF_CAPS_LOCK_ON, XKB_MOD_NAME_CAPS},
  };
  uint32_t xkb_modifiers = 0;
  for (auto modifier : modifiers) {
    if (modifier_flags & modifier.flag) {
      xkb_modifiers |=
          1 << xkb_keymap_mod_get_index(xkb_keymap_.get(), modifier.xkb_name);
    }
  }
  return xkb_modifiers;
}

#if defined(OS_CHROMEOS)
void WaylandKeyboardDelegate::SendNamedLayout(const std::string& layout_name) {
  std::string layout_id, layout_variant;
  ui::XkbKeyboardLayoutEngine::ParseLayoutName(layout_name, &layout_id,
                                               &layout_variant);
  xkb_rule_names names = {.rules = nullptr,
                          .model = "pc101",
                          .layout = layout_id.c_str(),
                          .variant = layout_variant.c_str(),
                          .options = ""};
  SendLayout(&names);
}
#endif

void WaylandKeyboardDelegate::SendLayout(const xkb_rule_names* names) {
  xkb_keymap_.reset(xkb_keymap_new_from_names(xkb_context_.get(), names,
                                              XKB_KEYMAP_COMPILE_NO_FLAGS));
  xkb_state_.reset(xkb_state_new(xkb_keymap_.get()));
  std::unique_ptr<char, base::FreeDeleter> keymap_string(
      xkb_keymap_get_as_string(xkb_keymap_.get(), XKB_KEYMAP_FORMAT_TEXT_V1));
  DCHECK(keymap_string.get());
  size_t keymap_size = strlen(keymap_string.get()) + 1;
  base::SharedMemory shared_keymap;
  bool rv = shared_keymap.CreateAndMapAnonymous(keymap_size);
  DCHECK(rv);
  memcpy(shared_keymap.memory(), keymap_string.get(), keymap_size);
  wl_keyboard_send_keymap(keyboard_resource_, WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1,
                          shared_keymap.handle().GetHandle(), keymap_size);
  wl_client_flush(client());
}

wl_client* WaylandKeyboardDelegate::client() const {
  return wl_resource_get_client(keyboard_resource_);
}

uint32_t WaylandKeyboardDelegate::next_serial() const {
  return wl_display_next_serial(wl_client_get_display(client()));
}

#endif

}  // namespace wayland
}  // namespace exo

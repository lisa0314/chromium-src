// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "device/fido/make_credential_task.h"

#include <utility>

#include "base/bind.h"
#include "device/base/features.h"
#include "device/fido/ctap2_device_operation.h"
#include "device/fido/u2f_command_constructor.h"
#include "device/fido/u2f_register_operation.h"

namespace device {

namespace {

// Checks whether the incoming MakeCredential request has ClientPin option that
// is compatible with the Chrome's CTAP2 implementation. According to the CTAP
// spec, CTAP2 authenticators that have client pin set will always error out on
// MakeCredential request when "pinAuth" parameter in the request is not set. As
// ClientPin is not supported on Chrome yet, this check allows Chrome to avoid
// such failures if possible by defaulting to U2F request when user verification
// is not required and the device supports U2F protocol.
// TODO(hongjunchoi): Remove this once ClientPin command is implemented.
// See: https://crbug.com/870892
bool ShouldUseU2fBecauseCtapRequiresClientPin(
    const FidoDevice* device,
    const CtapMakeCredentialRequest& request) {
  if (request.user_verification() == UserVerificationRequirement::kRequired)
    return false;

  DCHECK(device && device->device_info());
  bool client_pin_set =
      device->device_info()->options().client_pin_availability ==
      AuthenticatorSupportedOptions::ClientPinAvailability::kSupportedAndPinSet;
  bool supports_u2f = base::ContainsKey(device->device_info()->versions(),
                                        ProtocolVersion::kU2f);
  return client_pin_set && supports_u2f;
}

}  // namespace

MakeCredentialTask::MakeCredentialTask(
    FidoDevice* device,
    CtapMakeCredentialRequest request_parameter,
    MakeCredentialTaskCallback callback)
    : FidoTask(device),
      request_parameter_(std::move(request_parameter)),
      callback_(std::move(callback)),
      weak_factory_(this) {
  // The UV parameter should have been made binary by this point because CTAP2
  // only takes a binary value.
  DCHECK_NE(request_parameter_.user_verification(),
            UserVerificationRequirement::kPreferred);
}

MakeCredentialTask::~MakeCredentialTask() = default;

void MakeCredentialTask::StartTask() {
  if (device()->supported_protocol() == ProtocolVersion::kCtap &&
      !request_parameter_.is_u2f_only() &&
      !ShouldUseU2fBecauseCtapRequiresClientPin(device(), request_parameter_)) {
    MakeCredential();
  } else {
    // |device_info| should be present iff the device is CTAP2. This will be
    // used in |MaybeRevertU2fFallback| to restore the protocol of CTAP2 devices
    // once this task is complete.
    DCHECK((device()->supported_protocol() == ProtocolVersion::kCtap) ==
           static_cast<bool>(device()->device_info()));
    device()->set_supported_protocol(ProtocolVersion::kU2f);
    U2fRegister();
  }
}

void MakeCredentialTask::MakeCredential() {
  register_operation_ = std::make_unique<Ctap2DeviceOperation<
      CtapMakeCredentialRequest, AuthenticatorMakeCredentialResponse>>(
      device(), std::move(request_parameter_), std::move(callback_),
      base::BindOnce(&ReadCTAPMakeCredentialResponse,
                     device()->DeviceTransport()));
  register_operation_->Start();
}

void MakeCredentialTask::U2fRegister() {
  if (!IsConvertibleToU2fRegisterCommand(request_parameter_)) {
    std::move(callback_).Run(CtapDeviceResponseCode::kCtap2ErrOther,
                             base::nullopt);
    return;
  }

  DCHECK_EQ(ProtocolVersion::kU2f, device()->supported_protocol());
  register_operation_ = std::make_unique<U2fRegisterOperation>(
      device(), std::move(request_parameter_),
      base::BindOnce(&MakeCredentialTask::MaybeRevertU2fFallback,
                     weak_factory_.GetWeakPtr()));
  register_operation_->Start();
}

void MakeCredentialTask::MaybeRevertU2fFallback(
    CtapDeviceResponseCode status,
    base::Optional<AuthenticatorMakeCredentialResponse> response) {
  DCHECK_EQ(ProtocolVersion::kU2f, device()->supported_protocol());
  if (device()->device_info()) {
    // This was actually a CTAP2 device, but the protocol version was set to U2F
    // because it had a PIN set and so, in order to make a credential, the U2F
    // interface was used.
    device()->set_supported_protocol(ProtocolVersion::kCtap);
  }

  std::move(callback_).Run(status, std::move(response));
}

}  // namespace device

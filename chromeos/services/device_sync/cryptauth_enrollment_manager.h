// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROMEOS_SERVICES_DEVICE_SYNC_CRYPTAUTH_ENROLLMENT_MANAGER_H_
#define CHROMEOS_SERVICES_DEVICE_SYNC_CRYPTAUTH_ENROLLMENT_MANAGER_H_

#include <memory>

#include "base/macros.h"
#include "base/observer_list.h"
#include "base/time/time.h"
#include "chromeos/services/device_sync/proto/cryptauth_api.pb.h"

namespace chromeos {

namespace device_sync {

// This class manages the device's enrollment with CryptAuth, periodically
// re-enrolling to keep the state on the server fresh. If an enrollment fails,
// the manager will schedule the next enrollment more aggressively to recover
// from the failure.
class CryptAuthEnrollmentManager {
 public:
  class Observer {
   public:
    // Called when an enrollment attempt is started.
    virtual void OnEnrollmentStarted() {}

    // Called when an enrollment attempt finishes with the |success| of the
    // attempt.
    virtual void OnEnrollmentFinished(bool success) {}

    virtual ~Observer() = default;
  };

  CryptAuthEnrollmentManager();
  virtual ~CryptAuthEnrollmentManager();

  void AddObserver(Observer* observer);
  void RemoveObserver(Observer* observer);

  // Begins scheduling periodic enrollment attempts.
  virtual void Start() = 0;

  // Skips the waiting period and forces an enrollment immediately. If an
  // enrollment is already in progress, this function does nothing.
  // |invocation_reason| specifies the reason that the enrollment was triggered,
  // which is upload to the server.
  virtual void ForceEnrollmentNow(
      cryptauth::InvocationReason invocation_reason) = 0;

  // Returns true if a successful enrollment has been recorded and this
  // enrollment has not expired.
  virtual bool IsEnrollmentValid() const = 0;

  // Returns the timestamp of the last successful enrollment. If no enrollment
  // has ever been made, then a null base::Time object will be returned.
  virtual base::Time GetLastEnrollmentTime() const = 0;

  // Returns the time to the next enrollment attempt.
  virtual base::TimeDelta GetTimeToNextAttempt() const = 0;

  // Returns true if an enrollment attempt is currently in progress.
  virtual bool IsEnrollmentInProgress() const = 0;

  // Returns true if the last enrollment failed and the manager is now
  // scheduling enrollments more aggressively to recover. If no enrollment has
  // ever been recorded, then this function will also return true.
  virtual bool IsRecoveringFromFailure() const = 0;

  // Returns the keypair used to enroll with CryptAuth. If no enrollment has
  // been completed, then an empty string will be returned.
  // Note: These keys are really serialized protocol buffer messages, and should
  // only be used by passing to SecureMessageDelegate.
  virtual std::string GetUserPublicKey() const = 0;
  virtual std::string GetUserPrivateKey() const = 0;

 protected:
  void NotifyEnrollmentStarted();
  void NotifyEnrollmentFinished(bool success);

 private:
  base::ObserverList<Observer>::Unchecked observers_;

  DISALLOW_COPY_AND_ASSIGN(CryptAuthEnrollmentManager);
};

}  // namespace device_sync

}  // namespace chromeos

#endif  // CHROMEOS_SERVICES_DEVICE_SYNC_CRYPTAUTH_ENROLLMENT_MANAGER_H_

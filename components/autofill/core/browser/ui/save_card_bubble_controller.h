// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef COMPONENTS_AUTOFILL_CORE_BROWSER_UI_SAVE_CARD_BUBBLE_CONTROLLER_H_
#define COMPONENTS_AUTOFILL_CORE_BROWSER_UI_SAVE_CARD_BUBBLE_CONTROLLER_H_

#include <memory>
#include <vector>

#include "base/macros.h"
#include "base/strings/string16.h"
#include "components/autofill/core/browser/autofill_client.h"
#include "components/autofill/core/browser/payments/legal_message_line.h"
#include "components/autofill/core/browser/sync_utils.h"
#include "components/signin/core/browser/account_info.h"
#include "url/gurl.h"

class Profile;

namespace signin_metrics {
enum class AccessPoint;
}

namespace autofill {

class CreditCard;
class SaveCardBubbleView;
enum class BubbleType;

// Interface that exposes controller functionality to SaveCardBubbleView.
class SaveCardBubbleController {
 public:
  SaveCardBubbleController() {}
  virtual ~SaveCardBubbleController() {}

  // Returns the title that should be displayed in the bubble.
  virtual base::string16 GetWindowTitle() const = 0;

  // Returns the explanatory text that should be displayed in the bubble.
  // Returns an empty string if no message should be displayed.
  virtual base::string16 GetExplanatoryMessage() const = 0;

  // Returns the account info of the signed-in user.
  virtual const AccountInfo& GetAccountInfo() const = 0;

  // Returns the profile.
  virtual Profile* GetProfile() const = 0;

  // Returns the card that will be uploaded if the user accepts.
  virtual const CreditCard& GetCard() const = 0;

  // Returns whether the dialog should include a textfield requesting the user
  // to confirm/provide cardholder name.
  virtual bool ShouldRequestNameFromUser() const = 0;

  // Returns whether the dialog should include a pair of dropdown lists
  // allowing the user to provide expiration date.
  virtual bool ShouldRequestExpirationDateFromUser() const = 0;

  // Returns whether or not a sign in / sync promo needs to be shown.
  virtual bool ShouldShowSignInPromo() const = 0;

  // Returns true iff the card saved animation can be shown.
  virtual bool CanAnimate() const = 0;

  // Interaction.
  // OnSyncPromoAccepted is called when the Dice Sign-in promo is clicked.
  virtual void OnSyncPromoAccepted(const AccountInfo& account,
                                   signin_metrics::AccessPoint access_point,
                                   bool is_default_promo_account) = 0;
  // OnSaveButton takes in a struct representing the cardholder name,
  // expiration date month and expiration date year confirmed/entered by the
  // user if they were requested, or struct with empty strings otherwise.
  virtual void OnSaveButton(const AutofillClient::UserProvidedCardDetails&
                                user_provided_card_details) = 0;
  virtual void OnCancelButton() = 0;
  virtual void OnLegalMessageLinkClicked(const GURL& url) = 0;
  virtual void OnManageCardsClicked() = 0;
  virtual void OnBubbleClosed() = 0;
  // Once the animation ends, it shows a new bubble if needed.
  virtual void OnAnimationEnded() = 0;

  // State.

  // Returns empty vector if no legal message should be shown.
  virtual const LegalMessageLines& GetLegalMessageLines() const = 0;
  // Returns true iff is showing or has showed bubble for upload save.
  virtual bool IsUploadSave() const = 0;
  // Returns the current state of the bubble.
  virtual BubbleType GetBubbleType() const = 0;
  // Returns the current sync state.
  virtual AutofillSyncSigninState GetSyncState() const = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(SaveCardBubbleController);
};

}  // namespace autofill

#endif  // COMPONENTS_AUTOFILL_CORE_BROWSER_UI_SAVE_CARD_BUBBLE_CONTROLLER_H_

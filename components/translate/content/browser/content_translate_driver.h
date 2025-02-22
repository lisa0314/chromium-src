// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef COMPONENTS_TRANSLATE_CONTENT_BROWSER_CONTENT_TRANSLATE_DRIVER_H_
#define COMPONENTS_TRANSLATE_CONTENT_BROWSER_CONTENT_TRANSLATE_DRIVER_H_

#include <map>
#include <string>

#include "base/macros.h"
#include "base/memory/weak_ptr.h"
#include "base/observer_list.h"
#include "components/translate/content/common/translate.mojom.h"
#include "components/translate/core/browser/translate_driver.h"
#include "components/translate/core/common/translate_errors.h"
#include "content/public/browser/web_contents_observer.h"
#include "mojo/public/cpp/bindings/binding_set.h"
#include "services/network/public/mojom/url_loader_factory.mojom.h"

namespace content {
class NavigationController;
class WebContents;
}

namespace language {
class UrlLanguageHistogram;
}  // namespace language

class TemplateURLService;

namespace translate {

struct LanguageDetectionDetails;
class TranslateManager;

// Content implementation of TranslateDriver.
class ContentTranslateDriver : public TranslateDriver,
                               public translate::mojom::ContentTranslateDriver,
                               public content::WebContentsObserver {
 public:
  // The observer for the ContentTranslateDriver.
  class Observer {
   public:
    // Handles when the value of IsPageTranslated is changed.
    virtual void OnIsPageTranslatedChanged(content::WebContents* source) {}

    // Handles when the value of translate_enabled is changed.
    virtual void OnTranslateEnabledChanged(content::WebContents* source) {}

    // Called when the page language has been determined.
    virtual void OnLanguageDetermined(
        const translate::LanguageDetectionDetails& details) {}

    // Called when the page has been translated.
    virtual void OnPageTranslated(const std::string& original_lang,
                                  const std::string& translated_lang,
                                  translate::TranslateErrors::Type error_type) {
    }

   protected:
    virtual ~Observer() {}
  };

  ContentTranslateDriver(
      content::NavigationController* nav_controller,
      const TemplateURLService* template_url_service,
      language::UrlLanguageHistogram* url_language_histogram);
  ~ContentTranslateDriver() override;

  // Adds or Removes observers.
  void AddObserver(Observer* observer);
  void RemoveObserver(Observer* observer);

  // Number of attempts before waiting for a page to be fully reloaded.
  void set_translate_max_reload_attempts(int attempts) {
    max_reload_check_attempts_ = attempts;
  }

  // Sets the TranslateManager associated with this driver.
  void set_translate_manager(TranslateManager* manager) {
    translate_manager_ = manager;
  }

  // Initiates translation once the page is finished loading.
  void InitiateTranslation(const std::string& page_lang, int attempt);

  // TranslateDriver methods.
  void OnIsPageTranslatedChanged() override;
  void OnTranslateEnabledChanged() override;
  bool IsLinkNavigation() override;
  void TranslatePage(int page_seq_no,
                     const std::string& translate_script,
                     const std::string& source_lang,
                     const std::string& target_lang) override;
  void RevertTranslation(int page_seq_no) override;
  bool IsIncognito() override;
  const std::string& GetContentsMimeType() override;
  const GURL& GetLastCommittedURL() override;
  const GURL& GetVisibleURL() override;
  bool HasCurrentPage() override;
  void OpenUrlInNewTab(const GURL& url) override;

  // content::WebContentsObserver implementation.
  void NavigationEntryCommitted(
      const content::LoadCommittedDetails& load_details) override;
  void DidFinishNavigation(
      content::NavigationHandle* navigation_handle) override;

  void OnPageTranslated(bool cancelled,
                        const std::string& original_lang,
                        const std::string& translated_lang,
                        TranslateErrors::Type error_type);

  // Adds a binding in |bindings_| for the passed |request|.
  void AddBinding(translate::mojom::ContentTranslateDriverRequest request);
  // Called when a page has been loaded and can be potentially translated.
  void RegisterPage(translate::mojom::PagePtr page,
                    const translate::LanguageDetectionDetails& details,
                    bool page_needs_translation) override;

 private:
  void OnPageAway(int page_seq_no);

  bool IsDefaultSearchEngineOriginator(
      const url::Origin& originating_origin) const;

  // Creates a URLLoaderFactory that may be used by the translate scripts that
  // get injected into isolated worlds within the page to be translated.  Such
  // scripts (or rather, their isolated worlds) are associated with a
  // translate-specific origin like https://translate.googleapis.com and use
  // this origin as |request_initiator| of http requests.
  network::mojom::URLLoaderFactoryPtr CreateURLLoaderFactory();

  // The navigation controller of the tab we are associated with.
  content::NavigationController* navigation_controller_;

  TranslateManager* translate_manager_;

  base::ObserverList<Observer, true>::Unchecked observer_list_;

  // Max number of attempts before checking if a page has been reloaded.
  int max_reload_check_attempts_;

  const TemplateURLService* template_url_service_;

  // Records mojo connections with all current alive pages.
  int next_page_seq_no_;
  // PagePtr is the connection between this driver and a TranslateHelper (which
  // are per RenderFrame). Each TranslateHelper has a |binding_| member,
  // representing the other end of this pipe.
  std::map<int, mojom::PagePtr> pages_;

  // Histogram to be notified about detected language of every page visited. Not
  // owned here.
  language::UrlLanguageHistogram* const language_histogram_;

  // ContentTranslateDriver is a singleton per web contents but multiple render
  // frames may be contained in a single web contents. TranslateHelpers get the
  // other end of this binding in the form of a ContentTranslateDriverPtr.
  mojo::BindingSet<translate::mojom::ContentTranslateDriver> bindings_;

  base::WeakPtrFactory<ContentTranslateDriver> weak_pointer_factory_;

  DISALLOW_COPY_AND_ASSIGN(ContentTranslateDriver);
};

}  // namespace translate

#endif  // COMPONENTS_TRANSLATE_CONTENT_BROWSER_CONTENT_TRANSLATE_DRIVER_H_

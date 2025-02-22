// Copyright 2013 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Decodes a UTF16 string that is encoded as base64.
function decodeUTF16Base64ToString(encoded_text) {
  var data = atob(encoded_text);
  var result = '';
  for (var i = 0; i < data.length; i += 2) {
    result +=
        String.fromCharCode(data.charCodeAt(i) * 256 + data.charCodeAt(i + 1));
  }
  return result;
}

function toggleHelpBox() {
  var helpBoxOuter = document.getElementById('details');
  helpBoxOuter.classList.toggle(HIDDEN_CLASS);
  var detailsButton = document.getElementById('details-button');
  if (helpBoxOuter.classList.contains(HIDDEN_CLASS))
    detailsButton.innerText = detailsButton.detailsText;
  else
    detailsButton.innerText = detailsButton.hideDetailsText;

  // Details appears over the main content on small screens.
  if (mobileNav) {
    document.getElementById('main-content').classList.toggle(HIDDEN_CLASS);
    var runnerContainer = document.querySelector('.runner-container');
    if (runnerContainer) {
      runnerContainer.classList.toggle(HIDDEN_CLASS);
    }
  }
}

function diagnoseErrors() {
// <if expr="not chromeos">
    if (window.errorPageController)
      errorPageController.diagnoseErrorsButtonClick();
// </if>
// <if expr="chromeos">
  var extensionId = 'idddmepepmjcgiedknnmlbadcokidhoa';
  var diagnoseFrame = document.getElementById('diagnose-frame');
  diagnoseFrame.innerHTML =
      '<iframe src="chrome-extension://' + extensionId +
      '/index.html"></iframe>';
// </if>
}

// Subframes use a different layout but the same html file.  This is to make it
// easier to support platforms that load the error page via different
// mechanisms (Currently just iOS).
if (window.top.location != window.location)
  document.documentElement.setAttribute('subframe', '');

// Re-renders the error page using |strings| as the dictionary of values.
// Used by NetErrorTabHelper to update DNS error pages with probe results.
function updateForDnsProbe(strings) {
  var context = new JsEvalContext(strings);
  jstProcess(context, document.getElementById('t'));
  onDocumentLoadOrUpdate();
}

// Given the classList property of an element, adds an icon class to the list
// and removes the previously-
function updateIconClass(classList, newClass) {
  var oldClass;

  if (classList.hasOwnProperty('last_icon_class')) {
    oldClass = classList['last_icon_class'];
    if (oldClass == newClass)
      return;
  }

  classList.add(newClass);
  if (oldClass !== undefined)
    classList.remove(oldClass);

  classList['last_icon_class'] = newClass;

  if (newClass == 'icon-offline') {
    document.body.classList.add('offline');
    new Runner('.interstitial-wrapper');
  } else {
    document.body.classList.add('neterror');
  }
}

// Does a search using |baseSearchUrl| and the text in the search box.
function search(baseSearchUrl) {
  var searchTextNode = document.getElementById('search-box');
  document.location = baseSearchUrl + searchTextNode.value;
  return false;
}

// Use to track clicks on elements generated by the navigation correction
// service.  If |trackingId| is negative, the element does not come from the
// correction service.
function trackClick(trackingId) {
  // This can't be done with XHRs because XHRs are cancelled on navigation
  // start, and because these are cross-site requests.
  if (trackingId >= 0 && errorPageController)
    errorPageController.trackClick(trackingId);
}

// Called when an <a> tag generated by the navigation correction service is
// clicked.  Separate function from trackClick so the resources don't have to
// be updated if new data is added to jstdata.
function linkClicked(jstdata) {
  trackClick(jstdata.trackingId);
}

// Implements button clicks.  This function is needed during the transition
// between implementing these in trunk chromium and implementing them in
// iOS.
function reloadButtonClick(url) {
  if (window.errorPageController) {
    errorPageController.reloadButtonClick();
  } else {
    location = url;
  }
}

function downloadButtonClick() {
  if (window.errorPageController) {
    errorPageController.downloadButtonClick();
    var downloadButton = document.getElementById('download-button');
    downloadButton.disabled = true;
    downloadButton.textContent = downloadButton.disabledText;

    document.getElementById('download-link-wrapper')
        .classList.add(HIDDEN_CLASS);
    document.getElementById('download-link-clicked-wrapper')
        .classList.remove(HIDDEN_CLASS);
  }
}

function detailsButtonClick() {
  if (window.errorPageController)
    errorPageController.detailsButtonClick();
}

/**
 * Replace the reload button with the Google cached copy suggestion.
 */
function setUpCachedButton(buttonStrings) {
  var reloadButton = document.getElementById('reload-button');

  reloadButton.textContent = buttonStrings.msg;
  var url = buttonStrings.cacheUrl;
  var trackingId = buttonStrings.trackingId;
  reloadButton.onclick = function(e) {
    e.preventDefault();
    trackClick(trackingId);
    if (window.errorPageController) {
      errorPageController.trackCachedCopyButtonClick();
    }
    location = url;
  };
  reloadButton.style.display = '';
}

var primaryControlOnLeft = true;
// <if expr="is_macosx or is_ios or is_linux or is_android">
primaryControlOnLeft = false;
// </if>

function setAutoFetchState(scheduled, can_schedule) {
  document.getElementById('cancel-save-page-button')
      .classList.toggle(HIDDEN_CLASS, !scheduled);
  document.getElementById('save-page-for-later-button')
      .classList.toggle(HIDDEN_CLASS, scheduled || !can_schedule);
}

function savePageLaterClick() {
  errorPageController.savePageForLater();
  // savePageForLater will eventually trigger a call to setAutoFetchState() when
  // it completes.
}

function cancelSavePageClick() {
  errorPageController.cancelSavePage();
  // setAutoFetchState is not called in response to cancelSavePage(), so do it
  // now.
  setAutoFetchState(false, true);
}

function toggleErrorInformationPopup() {
  document.getElementById('error-information-popup-container')
      .classList.toggle(HIDDEN_CLASS);
}

function launchOfflineItem(itemID, name_space) {
  errorPageController.launchOfflineItem(itemID, name_space);
}

function launchDownloadsPage() {
  errorPageController.launchDownloadsPage();
}

// Populates a summary of suggested offline content.
function offlineContentSummaryAvailable(summary) {
  // Note: See AvailableContentSummaryToValue in
  // available_offline_content_helper.cc for the data contained in |summary|.
  if (!summary || summary.total_items == 0 ||
      !loadTimeData.valueExists('offlineContentSummary')) {
    return;
  }
  // TODO(https://crbug.com/852872): Customize presented icons based on the
  // types of available offline content.
  document.getElementById('offline-content-summary').hidden = false;
}

function getIconForSuggestedItem(item) {
  // Note: |item.content_type| contains the enum values from
  // chrome::mojom::AvailableContentType.
  switch (item.content_type) {
    case 1:  // kVideo
      return 'image-video';
    case 2:  // kAudio
      return 'image-music-note';
    case 0:  // kPrefetchedPage
    case 3:  // kOtherPage
      return 'image-earth';
  }
  return 'image-file';
}

function getSuggestedContentDiv(item, index) {
  // Note: See AvailableContentToValue in available_offline_content_helper.cc
  // for the data contained in an |item|.
  // TODO(carlosk): Present |snippet_base64| when that content becomes
  // available.
  var visual = '';
  var extraContainerClasses = [];
  // html_inline.py will try to replace src attributes with data URIs using a
  // simple regex. The following is obfuscated slightly to avoid that.
  var src = 'src';
  if (item.thumbnail_data_uri) {
    extraContainerClasses.push('suggestion-with-image');
    visual = `<img ${src}="${item.thumbnail_data_uri}">`;
  } else {
    extraContainerClasses.push('suggestion-with-icon');
    iconClass = getIconForSuggestedItem(item);
    visual = `<div><img class="${iconClass}"></div>`;
  }

  if (!item.attribution_base64)
    extraContainerClasses.push('no-attribution');

  return `
  <div class="offline-content-suggestion ${extraContainerClasses.join(' ')}"
    onclick="launchOfflineItem('${item.ID}', '${item.name_space}')">
      <div class="offline-content-suggestion-texts">
        <div id="offline-content-suggestion-title-${index}"
             class="offline-content-suggestion-title">
        </div>
        <div class="offline-content-suggestion-attribution-freshness">
          <div id="offline-content-suggestion-attribution-${index}"
               class="offline-content-suggestion-attribution">
          </div>
          <div class="offline-content-suggestion-freshness">
            ${item.date_modified}
          </div>
          <div class="offline-content-suggestion-pin-spacer"></div>
          <div class="offline-content-suggestion-pin"></div>
        </div>
      </div>
      <div class="offline-content-suggestion-visual">
        ${visual}
      </div>
  </div>`;
}

// Populates a list of suggested offline content.
// Note: For security reasons all content downloaded from the web is considered
// unsafe and must be securely handled to be presented on the dino page. Images
// have already been safely re-encoded but textual content -- like title and
// attribution -- must be properly handled here.
function offlineContentAvailable(isShown, suggestions) {
  if (!suggestions || !loadTimeData.valueExists('offlineContentList'))
    return;

  var suggestionsHTML = [];
  for (var index = 0; index < suggestions.length; index++)
    suggestionsHTML.push(getSuggestedContentDiv(suggestions[index], index));

  document.getElementById('offline-content-suggestions').innerHTML =
      suggestionsHTML.join('\n');

  // Sets textual web content using |textContent| to make sure it's handled as
  // plain text.
  for (var index = 0; index < suggestions.length; index++) {
    document.getElementById(`offline-content-suggestion-title-${index}`)
        .textContent =
        decodeUTF16Base64ToString(suggestions[index].title_base64);
    document.getElementById(`offline-content-suggestion-attribution-${index}`)
        .textContent =
        decodeUTF16Base64ToString(suggestions[index].attribution_base64);
  }

  var contentListElement = document.getElementById('offline-content-list');
  if (document.dir == 'rtl')
    contentListElement.classList.add('is-rtl');
  contentListElement.hidden = false;
  // The list is configured as hidden by default. Show it if needed.
  if (isShown)
    toggleOfflineContentListVisibility(false);
}

function toggleOfflineContentListVisibility(updatePref) {
  if (!loadTimeData.valueExists('offlineContentList'))
    return;

  var contentListElement = document.getElementById('offline-content-list');
  var isVisible = !contentListElement.classList.toggle('list-hidden');

  if (updatePref && window.errorPageController) {
    errorPageController.listVisibilityChanged(isVisible);
  }
}

// Called on document load, and from updateForDnsProbe().
function onDocumentLoadOrUpdate() {
  var downloadButtonVisible = loadTimeData.valueExists('downloadButton') &&
      loadTimeData.getValue('downloadButton').msg;
  var detailsButton = document.getElementById('details-button');

  // If offline content suggestions will be visible, the usual buttons will not
  // be presented.
  var offlineContentVisible =
      loadTimeData.valueExists('suggestedOfflineContentPresentationMode');
  if (offlineContentVisible) {
    document.querySelector('.nav-wrapper').classList.add(HIDDEN_CLASS);
    detailsButton.classList.add(HIDDEN_CLASS);

    document.getElementById('download-link').hidden = !downloadButtonVisible;
    document.getElementById('download-links-wrapper')
        .classList.remove(HIDDEN_CLASS);
    document.getElementById('error-information-popup-container')
        .classList.add('use-popup-container', HIDDEN_CLASS)
    document.getElementById('error-information-button')
        .classList.remove(HIDDEN_CLASS);
  }

  var attemptAutoFetch = loadTimeData.valueExists('attemptAutoFetch') &&
      loadTimeData.getValue('attemptAutoFetch');

  var reloadButtonVisible = loadTimeData.valueExists('reloadButton') &&
      loadTimeData.getValue('reloadButton').msg;

  // Check for Google cached copy suggestion.
  var cacheButtonVisible = false;
  if (loadTimeData.valueExists('cacheButton')) {
    setUpCachedButton(loadTimeData.getValue('cacheButton'));
    cacheButtonVisible = true;
  }

  var reloadButton = document.getElementById('reload-button');
  var downloadButton = document.getElementById('download-button');
  if (reloadButton.style.display == 'none' &&
      downloadButton.style.display == 'none') {
    detailsButton.classList.add('singular');
  }

  // Show or hide control buttons.
  var controlButtonDiv = document.getElementById('control-buttons');
  controlButtonDiv.hidden = offlineContentVisible ||
      !(reloadButtonVisible || downloadButtonVisible || attemptAutoFetch ||
        cacheButtonVisible);
}

function onDocumentLoad() {
  // Sets up the proper button layout for the current platform.
  if (primaryControlOnLeft) {
    buttons.classList.add('suggested-left');
  } else {
    buttons.classList.add('suggested-right');
  }

  onDocumentLoadOrUpdate();
}

document.addEventListener('DOMContentLoaded', onDocumentLoad);

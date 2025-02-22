// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/offline_pages/offline_page_mhtml_archiver.h"

#include <utility>

#include "base/bind.h"
#include "base/bind_helpers.h"
#include "base/files/file_path.h"
#include "base/files/file_util.h"
#include "base/guid.h"
#include "base/location.h"
#include "base/logging.h"
#include "base/metrics/histogram_functions.h"
#include "base/strings/string16.h"
#include "base/task/post_task.h"
#include "base/threading/thread_task_runner_handle.h"
#include "chrome/browser/offline_pages/offline_page_utils.h"
#include "chrome/browser/ssl/security_state_tab_helper.h"
#include "components/offline_pages/core/archive_validator.h"
#include "components/offline_pages/core/model/offline_page_model_utils.h"
#include "components/offline_pages/core/offline_clock.h"
#include "components/offline_pages/core/offline_page_feature.h"
#include "components/security_state/core/security_state.h"
#include "content/public/browser/browser_thread.h"
#include "content/public/browser/navigation_entry.h"
#include "content/public/browser/web_contents.h"
#include "content/public/common/mhtml_generation_params.h"
#include "net/base/filename_util.h"

namespace offline_pages {
namespace {
void DeleteFileOnFileThread(const base::FilePath& file_path,
                            const base::Closure& callback) {
  base::PostTaskWithTraitsAndReply(
      FROM_HERE, {base::MayBlock(), base::TaskPriority::BEST_EFFORT},
      base::BindOnce(base::IgnoreResult(&base::DeleteFile), file_path,
                     false /* recursive */),
      callback);
}

// Compute a SHA256 digest using a background thread. The computed digest will
// be returned in the callback parameter. If it is empty, the digest calculation
// fails.
void ComputeDigestOnFileThread(
    const base::FilePath& file_path,
    base::OnceCallback<void(const std::string&)> callback) {
  base::PostTaskWithTraitsAndReplyWithResult(
      FROM_HERE, {base::MayBlock(), base::TaskPriority::BEST_EFFORT},
      base::BindOnce(&ArchiveValidator::ComputeDigest, file_path),
      std::move(callback));
}
}  // namespace

// static
OfflinePageMHTMLArchiver::OfflinePageMHTMLArchiver()
    : weak_ptr_factory_(this) {}

OfflinePageMHTMLArchiver::~OfflinePageMHTMLArchiver() {
}

void OfflinePageMHTMLArchiver::CreateArchive(
    const base::FilePath& archives_dir,
    const CreateArchiveParams& create_archive_params,
    content::WebContents* web_contents,
    CreateArchiveCallback callback) {
  DCHECK(callback_.is_null());
  DCHECK(!callback.is_null());
  callback_ = std::move(callback);

  // TODO(chili): crbug/710248 These checks should probably be done inside
  // the offliner.
  if (HasConnectionSecurityError(web_contents)) {
    ReportFailure(ArchiverResult::ERROR_SECURITY_CERTIFICATE);
    return;
  }

  // Don't save chrome error pages.
  if (GetPageType(web_contents) == content::PageType::PAGE_TYPE_ERROR) {
    ReportFailure(ArchiverResult::ERROR_ERROR_PAGE);
    return;
  }

  // Don't save chrome-injected interstitial info pages
  // i.e. "This site may be dangerous. Are you sure you want to continue?"
  if (GetPageType(web_contents) == content::PageType::PAGE_TYPE_INTERSTITIAL) {
    ReportFailure(ArchiverResult::ERROR_INTERSTITIAL_PAGE);
    return;
  }

  GenerateMHTML(archives_dir, web_contents, create_archive_params);
}

void OfflinePageMHTMLArchiver::GenerateMHTML(
    const base::FilePath& archives_dir,
    content::WebContents* web_contents,
    const CreateArchiveParams& create_archive_params) {
  if (archives_dir.empty()) {
    DVLOG(1) << "Archive path was empty. Can't create archive.";
    ReportFailure(ArchiverResult::ERROR_ARCHIVE_CREATION_FAILED);
    return;
  }

  if (!web_contents) {
    DVLOG(1) << "WebContents is missing. Can't create archive.";
    ReportFailure(ArchiverResult::ERROR_CONTENT_UNAVAILABLE);
    return;
  }

  if (!web_contents->GetRenderViewHost()) {
    DVLOG(1) << "RenderViewHost is not created yet. Can't create archive.";
    ReportFailure(ArchiverResult::ERROR_CONTENT_UNAVAILABLE);
    return;
  }

  GURL url(web_contents->GetLastCommittedURL());
  base::string16 title(web_contents->GetTitle());
  base::FilePath file_path(
      archives_dir.Append(base::GenerateGUID())
          .AddExtension(OfflinePageUtils::kMHTMLExtension));
  content::MHTMLGenerationParams params(file_path);
  params.use_binary_encoding = true;
  params.remove_popup_overlay = create_archive_params.remove_popup_overlay;
  params.use_page_problem_detectors =
      create_archive_params.use_page_problem_detectors;
  params.use_mojo_for_mhtml_serialization =
      IsOnTheFlyMhtmlHashComputationEnabled();

  web_contents->GenerateMHTML(
      params,
      base::BindOnce(&OfflinePageMHTMLArchiver::OnGenerateMHTMLDone,
                     weak_ptr_factory_.GetWeakPtr(), url, file_path, title,
                     create_archive_params.name_space, OfflineTimeNow()));
}

void OfflinePageMHTMLArchiver::OnGenerateMHTMLDone(
    const GURL& url,
    const base::FilePath& file_path,
    const base::string16& title,
    const std::string& name_space,
    base::Time mhtml_start_time,
    int64_t file_size) {
  if (file_size < 0) {
    DeleteFileAndReportFailure(file_path,
                               ArchiverResult::ERROR_ARCHIVE_CREATION_FAILED);
    return;
  }

  const base::Time digest_start_time = OfflineTimeNow();
  base::UmaHistogramTimes(
      model_utils::AddHistogramSuffix(
          name_space, "OfflinePages.SavePage.CreateArchiveTime"),
      digest_start_time - mhtml_start_time);
  ComputeDigestOnFileThread(
      file_path,
      base::BindOnce(&OfflinePageMHTMLArchiver::OnComputeDigestDone,
                     weak_ptr_factory_.GetWeakPtr(), url, file_path, title,
                     name_space, digest_start_time, file_size));
}

void OfflinePageMHTMLArchiver::OnComputeDigestDone(
    const GURL& url,
    const base::FilePath& file_path,
    const base::string16& title,
    const std::string& name_space,
    base::Time digest_start_time,
    int64_t file_size,
    const std::string& digest) {
  if (digest.empty()) {
    DeleteFileAndReportFailure(file_path,
                               ArchiverResult::ERROR_DIGEST_CALCULATION_FAILED);
    return;
  }

  base::UmaHistogramTimes(
      model_utils::AddHistogramSuffix(
          name_space, "OfflinePages.SavePage.ComputeDigestTime"),
      OfflineTimeNow() - digest_start_time);

  base::ThreadTaskRunnerHandle::Get()->PostTask(
      FROM_HERE,
      base::BindOnce(std::move(callback_), ArchiverResult::SUCCESSFULLY_CREATED,
                     url, file_path, title, file_size, digest));
}

bool OfflinePageMHTMLArchiver::HasConnectionSecurityError(
    content::WebContents* web_contents) {
  SecurityStateTabHelper::CreateForWebContents(web_contents);
  SecurityStateTabHelper* helper =
      SecurityStateTabHelper::FromWebContents(web_contents);
  DCHECK(helper);
  return security_state::SecurityLevel::DANGEROUS == helper->GetSecurityLevel();
}

content::PageType OfflinePageMHTMLArchiver::GetPageType(
    content::WebContents* web_contents) {
  return web_contents->GetController().GetVisibleEntry()->GetPageType();
}

void OfflinePageMHTMLArchiver::DeleteFileAndReportFailure(
    const base::FilePath& file_path,
    ArchiverResult result) {
  DeleteFileOnFileThread(file_path,
                         base::Bind(&OfflinePageMHTMLArchiver::ReportFailure,
                                    weak_ptr_factory_.GetWeakPtr(), result));
}

void OfflinePageMHTMLArchiver::ReportFailure(ArchiverResult result) {
  DCHECK(result != ArchiverResult::SUCCESSFULLY_CREATED);
  base::ThreadTaskRunnerHandle::Get()->PostTask(
      FROM_HERE,
      base::BindOnce(std::move(callback_), result, GURL(), base::FilePath(),
                     base::string16(), 0, std::string()));
}

}  // namespace offline_pages

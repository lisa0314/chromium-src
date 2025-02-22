// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CONTENT_BROWSER_BACKGROUND_FETCH_STORAGE_MATCH_REQUESTS_TASK_H_
#define CONTENT_BROWSER_BACKGROUND_FETCH_STORAGE_MATCH_REQUESTS_TASK_H_

#include <memory>

#include "base/callback_forward.h"
#include "base/memory/scoped_refptr.h"
#include "content/browser/background_fetch/background_fetch.pb.h"
#include "content/browser/background_fetch/background_fetch_request_match_params.h"
#include "content/browser/background_fetch/storage/database_task.h"
#include "content/browser/cache_storage/cache_storage_cache_handle.h"
#include "storage/browser/blob/blob_data_handle.h"
#include "third_party/blink/public/common/service_worker/service_worker_status_code.h"

namespace content {

class BackgroundFetchRequestMatchParams;

namespace background_fetch {

class MatchRequestsTask : public DatabaseTask {
 public:
  using SettledFetchesCallback = base::OnceCallback<void(
      blink::mojom::BackgroundFetchError,
      std::vector<blink::mojom::BackgroundFetchSettledFetchPtr>)>;

  // Gets settled fetches from cache storage, filtered according to
  // |match_params|.
  MatchRequestsTask(
      DatabaseTaskHost* host,
      const BackgroundFetchRegistrationId& registration_id,
      std::unique_ptr<BackgroundFetchRequestMatchParams> match_params,
      SettledFetchesCallback callback);

  ~MatchRequestsTask() override;

  // DatabaseTask implementation:
  void Start() override;

 private:
  void DidOpenCache(int64_t trace_id,
                    CacheStorageCacheHandle handle,
                    blink::mojom::CacheStorageError error);

  void DidGetAllMatchedEntries(
      int64_t trace_id,
      blink::mojom::CacheStorageError error,
      std::vector<CacheStorageCache::CacheEntry> entries);

  // Checks whether |request| shuld be matched given the provided query params.
  bool ShouldMatchRequest(const blink::mojom::FetchAPIRequestPtr& request);

  void FinishWithError(blink::mojom::BackgroundFetchError error) override;

  std::string HistogramName() const override;

  BackgroundFetchRegistrationId registration_id_;
  std::unique_ptr<BackgroundFetchRequestMatchParams> match_params_;
  SettledFetchesCallback callback_;

  CacheStorageCacheHandle handle_;
  std::vector<blink::mojom::BackgroundFetchSettledFetchPtr> settled_fetches_;

  base::WeakPtrFactory<MatchRequestsTask> weak_factory_;  // Keep as last.

  DISALLOW_COPY_AND_ASSIGN(MatchRequestsTask);
};

}  // namespace background_fetch

}  // namespace content

#endif  // CONTENT_BROWSER_BACKGROUND_FETCH_STORAGE_MATCH_REQUESTS_TASK_H_

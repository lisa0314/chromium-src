// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/page_load_metrics/page_load_metrics_observer.h"

#include <utility>

namespace page_load_metrics {

PageLoadExtraInfo::PageLoadExtraInfo(
    base::TimeTicks navigation_start,
    const base::Optional<base::TimeDelta>& first_background_time,
    const base::Optional<base::TimeDelta>& first_foreground_time,
    bool started_in_foreground,
    UserInitiatedInfo user_initiated_info,
    const GURL& url,
    const GURL& start_url,
    bool did_commit,
    PageEndReason page_end_reason,
    UserInitiatedInfo page_end_user_initiated_info,
    const base::Optional<base::TimeDelta>& page_end_time,
    const mojom::PageLoadMetadata& main_frame_metadata,
    const mojom::PageLoadMetadata& subframe_metadata,
    const mojom::PageRenderData& main_frame_render_data,
    ukm::SourceId source_id)
    : navigation_start(navigation_start),
      first_background_time(first_background_time),
      first_foreground_time(first_foreground_time),
      started_in_foreground(started_in_foreground),
      user_initiated_info(user_initiated_info),
      url(url),
      start_url(start_url),
      did_commit(did_commit),
      page_end_reason(page_end_reason),
      page_end_user_initiated_info(page_end_user_initiated_info),
      page_end_time(page_end_time),
      main_frame_metadata(main_frame_metadata),
      subframe_metadata(subframe_metadata),
      main_frame_render_data(main_frame_render_data),
      source_id(source_id) {}

PageLoadExtraInfo::PageLoadExtraInfo(const PageLoadExtraInfo& other) = default;

PageLoadExtraInfo::~PageLoadExtraInfo() {}

// static
PageLoadExtraInfo PageLoadExtraInfo::CreateForTesting(
    const GURL& url,
    bool started_in_foreground) {
  return PageLoadExtraInfo(
      base::TimeTicks::Now() /* navigation_start */,
      base::Optional<base::TimeDelta>() /* first_background_time */,
      base::Optional<base::TimeDelta>() /* first_foreground_time */,
      started_in_foreground /* started_in_foreground */,
      UserInitiatedInfo::BrowserInitiated(), url, url, true /* did_commit */,
      page_load_metrics::END_NONE,
      page_load_metrics::UserInitiatedInfo::NotUserInitiated(),
      base::TimeDelta(), page_load_metrics::mojom::PageLoadMetadata(),
      page_load_metrics::mojom::PageLoadMetadata(),
      page_load_metrics::mojom::PageRenderData(), 0 /* source_id */);
}

ExtraRequestCompleteInfo::ExtraRequestCompleteInfo(
    const GURL& url,
    const net::IPEndPoint& remote_endpoint,
    int frame_tree_node_id,
    bool was_cached,
    int64_t raw_body_bytes,
    int64_t original_network_content_length,
    std::unique_ptr<data_reduction_proxy::DataReductionProxyData>
        data_reduction_proxy_data,
    content::ResourceType detected_resource_type,
    int net_error,
    std::unique_ptr<net::LoadTimingInfo> load_timing_info)
    : url(url),
      remote_endpoint(remote_endpoint),
      frame_tree_node_id(frame_tree_node_id),
      was_cached(was_cached),
      raw_body_bytes(raw_body_bytes),
      original_network_content_length(original_network_content_length),
      data_reduction_proxy_data(std::move(data_reduction_proxy_data)),
      resource_type(detected_resource_type),
      net_error(net_error),
      load_timing_info(std::move(load_timing_info)) {}

ExtraRequestCompleteInfo::ExtraRequestCompleteInfo(
    const ExtraRequestCompleteInfo& other)
    : url(other.url),
      remote_endpoint(other.remote_endpoint),
      frame_tree_node_id(other.frame_tree_node_id),
      was_cached(other.was_cached),
      raw_body_bytes(other.raw_body_bytes),
      original_network_content_length(other.original_network_content_length),
      data_reduction_proxy_data(
          other.data_reduction_proxy_data == nullptr
              ? nullptr
              : other.data_reduction_proxy_data->DeepCopy()),
      resource_type(other.resource_type),
      net_error(other.net_error),
      load_timing_info(other.load_timing_info == nullptr
                           ? nullptr
                           : std::make_unique<net::LoadTimingInfo>(
                                 *other.load_timing_info)) {}

ExtraRequestCompleteInfo::~ExtraRequestCompleteInfo() {}

FailedProvisionalLoadInfo::FailedProvisionalLoadInfo(base::TimeDelta interval,
                                                     net::Error error)
    : time_to_failed_provisional_load(interval), error(error) {}

FailedProvisionalLoadInfo::~FailedProvisionalLoadInfo() {}

PageLoadMetricsObserver::ObservePolicy PageLoadMetricsObserver::OnStart(
    content::NavigationHandle* navigation_handle,
    const GURL& currently_committed_url,
    bool started_in_foreground) {
  return CONTINUE_OBSERVING;
}

PageLoadMetricsObserver::ObservePolicy PageLoadMetricsObserver::OnRedirect(
    content::NavigationHandle* navigation_handle) {
  return CONTINUE_OBSERVING;
}

PageLoadMetricsObserver::ObservePolicy PageLoadMetricsObserver::OnCommit(
    content::NavigationHandle* navigation_handle,
    ukm::SourceId source_id) {
  return CONTINUE_OBSERVING;
}

PageLoadMetricsObserver::ObservePolicy PageLoadMetricsObserver::OnHidden(
    const mojom::PageLoadTiming& timing,
    const PageLoadExtraInfo& extra_info) {
  return CONTINUE_OBSERVING;
}

PageLoadMetricsObserver::ObservePolicy PageLoadMetricsObserver::OnShown() {
  return CONTINUE_OBSERVING;
}

PageLoadMetricsObserver::ObservePolicy
PageLoadMetricsObserver::FlushMetricsOnAppEnterBackground(
    const mojom::PageLoadTiming& timing,
    const PageLoadExtraInfo& extra_info) {
  return CONTINUE_OBSERVING;
}

PageLoadMetricsObserver::ObservePolicy
PageLoadMetricsObserver::ShouldObserveMimeType(
    const std::string& mime_type) const {
  return IsStandardWebPageMimeType(mime_type) ? CONTINUE_OBSERVING
                                              : STOP_OBSERVING;
}

// static
bool PageLoadMetricsObserver::IsStandardWebPageMimeType(
    const std::string& mime_type) {
  return mime_type == "text/html" || mime_type == "application/xhtml+xml";
}

// static
bool PageLoadMetricsObserver::AssignTimeAndSizeForLargestContentfulPaint(
    const page_load_metrics::mojom::PaintTimingPtr& paint_timing,
    base::Optional<base::TimeDelta>* largest_content_paint_time,
    uint64_t* largest_content_paint_size,
    LargestContentType* largest_content_type) {
  base::Optional<base::TimeDelta>& text_time = paint_timing->largest_text_paint;
  base::Optional<base::TimeDelta>& image_time =
      paint_timing->largest_image_paint;
  uint64_t text_size = paint_timing->largest_text_paint_size;
  uint64_t image_size = paint_timing->largest_image_paint_size;

  // Size being 0 means the paint time is not recorded.
  if (!text_size && !image_size)
    return false;

  if ((text_size > image_size) ||
      (text_size == image_size && text_time < image_time)) {
    *largest_content_paint_time = text_time;
    *largest_content_paint_size = text_size;
    *largest_content_type = LargestContentType::kText;
  } else {
    *largest_content_paint_time = image_time;
    *largest_content_paint_size = image_size;
    *largest_content_type = LargestContentType::kImage;
  }
  return true;
}

PageLoadMetricsObserverDelegate* PageLoadMetricsObserver::GetDelegate() const {
  return delegate_;
}

void PageLoadMetricsObserver::SetDelegate(
    PageLoadMetricsObserverDelegate* delegate) {
  DCHECK(!delegate_);
  delegate_ = delegate;
}

}  // namespace page_load_metrics

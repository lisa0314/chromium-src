// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/sync/test/fake_server/fake_server_http_post_provider.h"

#include <utility>

#include "base/bind.h"
#include "base/location.h"
#include "base/time/time.h"
#include "components/sync/test/fake_server/fake_server.h"
#include "net/base/net_errors.h"

using syncer::HttpPostProviderInterface;

namespace fake_server {

// static
bool FakeServerHttpPostProvider::network_enabled_ = true;

FakeServerHttpPostProviderFactory::FakeServerHttpPostProviderFactory(
    const base::WeakPtr<FakeServer>& fake_server,
    scoped_refptr<base::SequencedTaskRunner> fake_server_task_runner)
    : fake_server_(fake_server),
      fake_server_task_runner_(fake_server_task_runner) {}

FakeServerHttpPostProviderFactory::~FakeServerHttpPostProviderFactory() {}

void FakeServerHttpPostProviderFactory::Init(
    const std::string& user_agent,
    const syncer::BindToTrackerCallback& bind_to_tracker_callback) {}

HttpPostProviderInterface* FakeServerHttpPostProviderFactory::Create() {
  FakeServerHttpPostProvider* http =
      new FakeServerHttpPostProvider(fake_server_, fake_server_task_runner_);
  http->AddRef();
  return http;
}

void FakeServerHttpPostProviderFactory::Destroy(
    HttpPostProviderInterface* http) {
  static_cast<FakeServerHttpPostProvider*>(http)->Release();
}

FakeServerHttpPostProvider::FakeServerHttpPostProvider(
    const base::WeakPtr<FakeServer>& fake_server,
    scoped_refptr<base::SequencedTaskRunner> fake_server_task_runner)
    : fake_server_(fake_server),
      fake_server_task_runner_(fake_server_task_runner),
      synchronous_post_completion_(
          base::WaitableEvent::ResetPolicy::AUTOMATIC,
          base::WaitableEvent::InitialState::NOT_SIGNALED),
      aborted_(false) {}

FakeServerHttpPostProvider::~FakeServerHttpPostProvider() {}

void FakeServerHttpPostProvider::SetExtraRequestHeaders(const char* headers) {
  // TODO(pvalenzuela): Add assertions on this value.
  extra_request_headers_.assign(headers);
}

void FakeServerHttpPostProvider::SetURL(const char* url, int port) {
  // TODO(pvalenzuela): Add assertions on these values.
  request_url_.assign(url);
  request_port_ = port;
}

void FakeServerHttpPostProvider::SetPostPayload(const char* content_type,
                                                int content_length,
                                                const char* content) {
  request_content_type_.assign(content_type);
  request_content_.assign(content, content_length);
}

bool FakeServerHttpPostProvider::MakeSynchronousPost(int* net_error_code,
                                                     int* http_status_code) {
  if (!network_enabled_) {
    response_.clear();
    *net_error_code = net::ERR_INTERNET_DISCONNECTED;
    *http_status_code = 0;
    return false;
  }

  synchronous_post_completion_.Reset();
  aborted_ = false;

  // It is assumed that a POST is being made to /command.
  int post_status_code = -1;
  std::string post_response;

  bool result = fake_server_task_runner_->PostTask(
      FROM_HERE,
      base::BindOnce(
          &FakeServerHttpPostProvider::HandleCommandOnFakeServerThread,
          base::RetainedRef(this), base::Unretained(&post_status_code),
          base::Unretained(&post_response)));

  if (!result) {
    response_.clear();
    *net_error_code = net::ERR_UNEXPECTED;
    *http_status_code = 0;
    return false;
  }

  synchronous_post_completion_.Wait();

  if (aborted_) {
    *net_error_code = net::ERR_ABORTED;
    return false;
  }

  // Zero means success.
  *net_error_code = 0;
  *http_status_code = post_status_code;
  response_ = post_response;

  return true;
}

int FakeServerHttpPostProvider::GetResponseContentLength() const {
  return response_.length();
}

const char* FakeServerHttpPostProvider::GetResponseContent() const {
  return response_.c_str();
}

const std::string FakeServerHttpPostProvider::GetResponseHeaderValue(
    const std::string& name) const {
  return std::string();
}

void FakeServerHttpPostProvider::Abort() {
  // The sync thread could be blocked in MakeSynchronousPost(), waiting
  // for HandleCommandOnFakeServerThread() to be processed and completed.
  // This causes an immediate unblocking which will be returned as
  // net::ERR_ABORTED.
  aborted_ = true;
  synchronous_post_completion_.Signal();
}

void FakeServerHttpPostProvider::DisableNetwork() {
  network_enabled_ = false;
}

void FakeServerHttpPostProvider::EnableNetwork() {
  network_enabled_ = true;
}

void FakeServerHttpPostProvider::HandleCommandOnFakeServerThread(
    int* http_status_code,
    std::string* response) {
  DCHECK(fake_server_task_runner_->RunsTasksInCurrentSequence());

  if (!fake_server_ || aborted_) {
    // Command explicitly aborted or server destroyed.
    return;
  }

  *http_status_code = fake_server_->HandleCommand(request_content_, response);
  synchronous_post_completion_.Signal();
}

}  // namespace fake_server

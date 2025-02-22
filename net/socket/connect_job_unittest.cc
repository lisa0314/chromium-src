// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "net/socket/connect_job.h"

#include "base/bind.h"
#include "base/callback.h"
#include "base/logging.h"
#include "base/macros.h"
#include "base/run_loop.h"
#include "base/test/scoped_task_environment.h"
#include "net/base/address_list.h"
#include "net/base/net_errors.h"
#include "net/base/request_priority.h"
#include "net/log/test_net_log.h"
#include "net/log/test_net_log_util.h"
#include "net/socket/connect_job_test_util.h"
#include "net/socket/socket_tag.h"
#include "net/socket/socket_test_util.h"
#include "net/test/gtest_util.h"
#include "testing/gtest/include/gtest/gtest.h"

namespace net {
namespace {

class TestConnectJob : public ConnectJob {
 public:
  enum class JobType {
    kSyncSuccess,
    kAsyncSuccess,
    kHung,
  };

  TestConnectJob(JobType job_type,
                 base::TimeDelta timeout_duration,
                 const CommonConnectJobParams* common_connect_job_params,
                 ConnectJob::Delegate* delegate)
      : ConnectJob(DEFAULT_PRIORITY,
                   SocketTag(),
                   timeout_duration,
                   common_connect_job_params,
                   delegate,
                   nullptr /* net_log */,
                   NetLogSourceType::TRANSPORT_CONNECT_JOB,
                   NetLogEventType::TRANSPORT_CONNECT_JOB_CONNECT),
        job_type_(job_type),
        last_seen_priority_(DEFAULT_PRIORITY) {
    switch (job_type_) {
      case JobType::kSyncSuccess:
        socket_data_provider_.set_connect_data(MockConnect(SYNCHRONOUS, OK));
        return;
      case JobType::kAsyncSuccess:
        socket_data_provider_.set_connect_data(MockConnect(ASYNC, OK));
        return;
      case JobType::kHung:
        socket_data_provider_.set_connect_data(
            MockConnect(SYNCHRONOUS, ERR_IO_PENDING));
        return;
    }
  }

  // From ConnectJob:
  LoadState GetLoadState() const override { return LOAD_STATE_IDLE; }
  bool HasEstablishedConnection() const override { return false; }
  int ConnectInternal() override {
    SetSocket(std::unique_ptr<StreamSocket>(new MockTCPClientSocket(
        AddressList(), net_log().net_log(), &socket_data_provider_)));
    return socket()->Connect(base::BindOnce(
        &TestConnectJob::NotifyDelegateOfCompletion, base::Unretained(this)));
  }
  void ChangePriorityInternal(RequestPriority priority) override {
    last_seen_priority_ = priority;
  }

  using ConnectJob::ResetTimer;

  // The priority seen during the most recent call to ChangePriorityInternal().
  RequestPriority last_seen_priority() const { return last_seen_priority_; }

 protected:
  const JobType job_type_;
  StaticSocketDataProvider socket_data_provider_;
  RequestPriority last_seen_priority_;

  DISALLOW_COPY_AND_ASSIGN(TestConnectJob);
};

class ConnectJobTest : public testing::Test {
 public:
  ConnectJobTest()
      : scoped_task_environment_(
            base::test::ScopedTaskEnvironment::MainThreadType::MOCK_TIME),
        common_connect_job_params_(
            nullptr /* client_socket_factory */,
            nullptr /* host_resolver */,
            nullptr /* proxy_delegate */,
            nullptr /* http_user_agent_settings */,
            SSLClientSocketContext(),
            SSLClientSocketContext(),
            nullptr /* socket_performance_watcher_factory */,
            nullptr /* network_quality_estimator */,
            &net_log_,
            nullptr /* websocket_endpoint_lock_manager */) {}
  ~ConnectJobTest() override = default;

 protected:
  base::test::ScopedTaskEnvironment scoped_task_environment_;
  TestNetLog net_log_;
  const CommonConnectJobParams common_connect_job_params_;
  TestConnectJobDelegate delegate_;
};

// Even though a timeout is specified, it doesn't time out on a synchronous
// completion.
TEST_F(ConnectJobTest, NoTimeoutOnSyncCompletion) {
  TestConnectJob job(TestConnectJob::JobType::kSyncSuccess,
                     base::TimeDelta::FromMicroseconds(1),
                     &common_connect_job_params_, &delegate_);
  EXPECT_THAT(job.Connect(), test::IsOk());
}

// Even though a timeout is specified, it doesn't time out on an asynchronous
// completion.
TEST_F(ConnectJobTest, NoTimeoutOnAsyncCompletion) {
  TestConnectJob job(TestConnectJob::JobType::kAsyncSuccess,
                     base::TimeDelta::FromMinutes(1),
                     &common_connect_job_params_, &delegate_);
  ASSERT_THAT(job.Connect(), test::IsError(ERR_IO_PENDING));
  EXPECT_THAT(delegate_.WaitForResult(), test::IsOk());
}

// Job shouldn't timeout when passed a TimeDelta of zero.
TEST_F(ConnectJobTest, NoTimeoutWithNoTimeDelta) {
  TestConnectJob job(TestConnectJob::JobType::kHung, base::TimeDelta(),
                     &common_connect_job_params_, &delegate_);
  ASSERT_THAT(job.Connect(), test::IsError(ERR_IO_PENDING));
  scoped_task_environment_.RunUntilIdle();
  EXPECT_FALSE(delegate_.has_result());
}

// Make sure that ChangePriority() works, and new priority is visible to
// subclasses during the SetPriorityInternal call.
TEST_F(ConnectJobTest, SetPriority) {
  TestConnectJob job(TestConnectJob::JobType::kAsyncSuccess,
                     base::TimeDelta::FromMicroseconds(1),
                     &common_connect_job_params_, &delegate_);
  ASSERT_THAT(job.Connect(), test::IsError(ERR_IO_PENDING));

  job.ChangePriority(HIGHEST);
  EXPECT_EQ(HIGHEST, job.priority());
  EXPECT_EQ(HIGHEST, job.last_seen_priority());

  job.ChangePriority(MEDIUM);
  EXPECT_EQ(MEDIUM, job.priority());
  EXPECT_EQ(MEDIUM, job.last_seen_priority());

  EXPECT_THAT(delegate_.WaitForResult(), test::IsOk());
}

TEST_F(ConnectJobTest, TimedOut) {
  const base::TimeDelta kTimeout = base::TimeDelta::FromHours(1);

  std::unique_ptr<TestConnectJob> job =
      std::make_unique<TestConnectJob>(TestConnectJob::JobType::kHung, kTimeout,
                                       &common_connect_job_params_, &delegate_);
  ASSERT_THAT(job->Connect(), test::IsError(ERR_IO_PENDING));

  // Nothing should happen before the specified time.
  scoped_task_environment_.FastForwardBy(kTimeout -
                                         base::TimeDelta::FromMilliseconds(1));
  base::RunLoop().RunUntilIdle();
  EXPECT_FALSE(delegate_.has_result());

  // At which point the job should time out.
  scoped_task_environment_.FastForwardBy(base::TimeDelta::FromMilliseconds(1));
  EXPECT_THAT(delegate_.WaitForResult(), test::IsError(ERR_TIMED_OUT));

  // Have to delete the job for it to log the end event.
  job.reset();

  TestNetLogEntry::List entries;
  net_log_.GetEntries(&entries);

  EXPECT_EQ(6u, entries.size());
  EXPECT_TRUE(LogContainsBeginEvent(entries, 0, NetLogEventType::CONNECT_JOB));
  EXPECT_TRUE(LogContainsBeginEvent(
      entries, 1, NetLogEventType::TRANSPORT_CONNECT_JOB_CONNECT));
  EXPECT_TRUE(LogContainsEvent(entries, 2,
                               NetLogEventType::CONNECT_JOB_SET_SOCKET,
                               NetLogEventPhase::NONE));
  EXPECT_TRUE(LogContainsEvent(entries, 3,
                               NetLogEventType::CONNECT_JOB_TIMED_OUT,
                               NetLogEventPhase::NONE));
  EXPECT_TRUE(LogContainsEndEvent(
      entries, 4, NetLogEventType::TRANSPORT_CONNECT_JOB_CONNECT));
  EXPECT_TRUE(LogContainsEndEvent(entries, 5, NetLogEventType::CONNECT_JOB));
}

TEST_F(ConnectJobTest, TimedOutWithRestartedTimer) {
  const base::TimeDelta kTimeout = base::TimeDelta::FromHours(1);

  TestConnectJob job(TestConnectJob::JobType::kHung, kTimeout,
                     &common_connect_job_params_, &delegate_);
  ASSERT_THAT(job.Connect(), test::IsError(ERR_IO_PENDING));

  // Nothing should happen before the specified time.
  scoped_task_environment_.FastForwardBy(kTimeout -
                                         base::TimeDelta::FromMilliseconds(1));
  base::RunLoop().RunUntilIdle();
  EXPECT_FALSE(delegate_.has_result());

  // Make sure restarting the timer is respected.
  job.ResetTimer(kTimeout);
  scoped_task_environment_.FastForwardBy(kTimeout -
                                         base::TimeDelta::FromMilliseconds(1));
  base::RunLoop().RunUntilIdle();
  EXPECT_FALSE(delegate_.has_result());

  scoped_task_environment_.FastForwardBy(base::TimeDelta::FromMilliseconds(1));
  EXPECT_THAT(delegate_.WaitForResult(), test::IsError(ERR_TIMED_OUT));
}

}  // namespace
}  // namespace net

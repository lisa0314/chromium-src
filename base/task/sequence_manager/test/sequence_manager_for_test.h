// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef BASE_TASK_SEQUENCE_MANAGER_TEST_SEQUENCE_MANAGER_FOR_TEST_H_
#define BASE_TASK_SEQUENCE_MANAGER_TEST_SEQUENCE_MANAGER_FOR_TEST_H_

#include <memory>

#include "base/single_thread_task_runner.h"
#include "base/task/sequence_manager/sequence_manager_impl.h"
#include "base/time/tick_clock.h"

namespace base {

namespace sequence_manager {

class SequenceManagerForTest : public internal::SequenceManagerImpl {
 public:
  ~SequenceManagerForTest() override = default;

  // Creates SequenceManagerForTest using ThreadControllerImpl constructed with
  // the given arguments. ThreadControllerImpl is slightly overridden to skip
  // nesting observers registration if message loop is absent.
  static std::unique_ptr<SequenceManagerForTest> Create(
      MessageLoopBase* message_loop_base,
      scoped_refptr<SingleThreadTaskRunner> task_runner,
      const TickClock* clock,
      // Since most test calls are in Blink, randomised sampling is enabled
      // by default in the test SequenceManager, as opposed to production code.
      SequenceManager::Settings settings = SequenceManager::Settings{
          .randomised_sampling_enabled = true});

  // Creates SequenceManagerForTest using the provided ThreadController.
  static std::unique_ptr<SequenceManagerForTest> Create(
      std::unique_ptr<internal::ThreadController> thread_controller,
      SequenceManager::Settings settings = SequenceManager::Settings{
          base::MessageLoop::TYPE_DEFAULT,
          /*randomised_sampling_enabled=*/true});

  size_t ActiveQueuesCount() const;
  bool HasImmediateWork() const;
  size_t PendingTasksCount() const;
  size_t QueuesToDeleteCount() const;
  size_t QueuesToShutdownCount();

  using internal::SequenceManagerImpl::GetNextSequenceNumber;
  using internal::SequenceManagerImpl::MoveReadyDelayedTasksToWorkQueues;
  using internal::SequenceManagerImpl::ReloadEmptyWorkQueues;

 private:
  explicit SequenceManagerForTest(
      std::unique_ptr<internal::ThreadController> thread_controller,
      SequenceManager::Settings settings);
};

}  // namespace sequence_manager
}  // namespace base

#endif  // BASE_TASK_SEQUENCE_MANAGER_TEST_SEQUENCE_MANAGER_FOR_TEST_H_

// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef COMPONENTS_LEVELDB_PROTO_TESTING_FAKE_DB_H_
#define COMPONENTS_LEVELDB_PROTO_TESTING_FAKE_DB_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/bind.h"
#include "base/callback.h"
#include "base/files/file_path.h"
#include "base/task/post_task.h"
#include "base/test/test_simple_task_runner.h"
#include "components/leveldb_proto/internal/proto_database_impl.h"
#include "components/leveldb_proto/public/proto_database.h"

namespace leveldb_proto {
namespace test {

template <typename P, typename T = P>
class FakeDB : public ProtoDatabaseImpl<P, T> {
  using Callback = base::OnceCallback<void(bool)>;

 public:
  using EntryMap = std::map<std::string, T>;

  explicit FakeDB(EntryMap* db);

  // ProtoDatabase implementation.
  void Init(const std::string& client_name,
            Callbacks::InitStatusCallback callback) override;
  void Init(Callbacks::InitStatusCallback callback) override;
  void Init(const leveldb_env::Options& unique_db_options,
            Callbacks::InitStatusCallback callback) override;
  void Init(const char* client_name,
            const base::FilePath& database_dir,
            const leveldb_env::Options& options,
            Callbacks::InitCallback callback) override;
  void UpdateEntries(std::unique_ptr<typename ProtoDatabase<T>::KeyEntryVector>
                         entries_to_save,
                     std::unique_ptr<std::vector<std::string>> keys_to_remove,
                     Callbacks::UpdateCallback callback) override;
  void UpdateEntriesWithRemoveFilter(
      std::unique_ptr<typename Util::Internal<T>::KeyEntryVector>
          entries_to_save,
      const KeyFilter& filter,
      Callbacks::UpdateCallback callback) override;
  void LoadEntries(
      typename Callbacks::Internal<T>::LoadCallback callback) override;
  void LoadEntriesWithFilter(
      const KeyFilter& key_filter,
      typename Callbacks::Internal<T>::LoadCallback callback) override;
  void LoadEntriesWithFilter(
      const KeyFilter& filter,
      const leveldb::ReadOptions& options,
      const std::string& target_prefix,
      typename Callbacks::Internal<T>::LoadCallback callback) override;
  void LoadKeysAndEntries(
      typename Callbacks::Internal<T>::LoadKeysAndEntriesCallback callback)
      override;
  void LoadKeysAndEntriesWithFilter(
      const KeyFilter& filter,
      typename Callbacks::Internal<T>::LoadKeysAndEntriesCallback callback)
      override;
  void LoadKeysAndEntriesWithFilter(
      const KeyFilter& filter,
      const leveldb::ReadOptions& options,
      const std::string& target_prefix,
      typename Callbacks::Internal<T>::LoadKeysAndEntriesCallback callback)
      override;
  void LoadKeysAndEntriesInRange(
      const std::string& start,
      const std::string& end,
      typename Callbacks::Internal<T>::LoadKeysAndEntriesCallback callback)
      override;
  void LoadKeys(Callbacks::LoadKeysCallback callback) override;
  void GetEntry(const std::string& key,
                typename Callbacks::Internal<T>::GetCallback callback) override;
  void Destroy(Callbacks::DestroyCallback callback) override;

  base::FilePath& GetDirectory();

  void InitCallback(bool success);

  void InitStatusCallback(Enums::InitStatus status);

  void LoadCallback(bool success);

  void LoadKeysCallback(bool success);

  void GetCallback(bool success);

  void UpdateCallback(bool success);

  void DestroyCallback(bool success);

  static base::FilePath DirectoryForTestDB();

 private:
  static void RunLoadCallback(
      typename Callbacks::Internal<T>::LoadCallback callback,
      std::unique_ptr<typename std::vector<T>> entries,
      bool success);
  static void RunLoadKeysAndEntriesCallback(
      typename Callbacks::Internal<T>::LoadKeysAndEntriesCallback callback,
      std::unique_ptr<typename std::map<std::string, T>> entries,
      bool success);

  static void RunLoadKeysCallback(
      typename Callbacks::LoadKeysCallback callback,
      std::unique_ptr<std::vector<std::string>> keys,
      bool success);

  static void RunGetCallback(
      typename Callbacks::Internal<T>::GetCallback callback,
      std::unique_ptr<T> entry,
      bool success);

  base::FilePath dir_;
  EntryMap* db_;

  Callback init_callback_;
  Callbacks::InitStatusCallback init_status_callback_;
  Callback load_callback_;
  Callback load_keys_callback_;
  Callback get_callback_;
  Callback update_callback_;
  Callback destroy_callback_;
};

template <typename P, typename T>
FakeDB<P, T>::FakeDB(EntryMap* db)
    : ProtoDatabaseImpl<P, T>(
          base::MakeRefCounted<base::TestSimpleTaskRunner>()) {
  db_ = db;
}

template <typename P, typename T>
void FakeDB<P, T>::Init(const std::string& client_name,
                        Callbacks::InitStatusCallback callback) {
  dir_ = base::FilePath(FILE_PATH_LITERAL("db_dir"));
  init_status_callback_ = std::move(callback);
}

template <typename P, typename T>
void FakeDB<P, T>::Init(const leveldb_env::Options& unique_db_options,
                        Callbacks::InitStatusCallback callback) {
  Init("fake_db", std::move(callback));
}

template <typename P, typename T>
void FakeDB<P, T>::Init(Callbacks::InitStatusCallback callback) {
  Init("fake_db", std::move(callback));
}

template <typename P, typename T>
void FakeDB<P, T>::Init(const char* client_name,
                        const base::FilePath& database_dir,
                        const leveldb_env::Options& options,
                        Callbacks::InitCallback callback) {
  dir_ = database_dir;
  init_callback_ = std::move(callback);
}

template <typename P, typename T>
void FakeDB<P, T>::UpdateEntries(
    std::unique_ptr<typename ProtoDatabase<T>::KeyEntryVector> entries_to_save,
    std::unique_ptr<std::vector<std::string>> keys_to_remove,
    Callbacks::UpdateCallback callback) {
  for (const auto& pair : *entries_to_save)
    (*db_)[pair.first] = pair.second;

  for (const auto& key : *keys_to_remove)
    db_->erase(key);

  update_callback_ = std::move(callback);
}

template <typename P, typename T>
void FakeDB<P, T>::UpdateEntriesWithRemoveFilter(
    std::unique_ptr<typename Util::Internal<T>::KeyEntryVector> entries_to_save,
    const KeyFilter& delete_key_filter,
    Callbacks::UpdateCallback callback) {
  for (const auto& pair : *entries_to_save)
    (*db_)[pair.first] = pair.second;

  auto it = db_->begin();
  while (it != db_->end()) {
    if (!delete_key_filter.is_null() && delete_key_filter.Run(it->first))
      db_->erase(it++);
    else
      ++it;
  }

  update_callback_ = std::move(callback);
}

template <typename P, typename T>
void FakeDB<P, T>::LoadEntries(
    typename Callbacks::Internal<T>::LoadCallback callback) {
  LoadEntriesWithFilter(KeyFilter(), std::move(callback));
}

template <typename P, typename T>
void FakeDB<P, T>::LoadEntriesWithFilter(
    const KeyFilter& key_filter,
    typename Callbacks::Internal<T>::LoadCallback callback) {
  LoadEntriesWithFilter(key_filter, leveldb::ReadOptions(), std::string(),
                        std::move(callback));
}

template <typename P, typename T>
void FakeDB<P, T>::LoadEntriesWithFilter(
    const KeyFilter& key_filter,
    const leveldb::ReadOptions& options,
    const std::string& target_prefix,
    typename Callbacks::Internal<T>::LoadCallback callback) {
  std::unique_ptr<std::vector<T>> entries(new std::vector<T>());
  for (const auto& pair : *db_) {
    if (key_filter.is_null() || key_filter.Run(pair.first)) {
      if (pair.first.compare(0, target_prefix.length(), target_prefix) == 0)
        entries->push_back(pair.second);
    }
  }

  load_callback_ =
      base::BindOnce(RunLoadCallback, std::move(callback), std::move(entries));
}

template <typename P, typename T>
void FakeDB<P, T>::LoadKeysAndEntries(
    typename Callbacks::Internal<T>::LoadKeysAndEntriesCallback callback) {
  LoadKeysAndEntriesWithFilter(KeyFilter(), std::move(callback));
}

template <typename P, typename T>
void FakeDB<P, T>::LoadKeysAndEntriesWithFilter(
    const KeyFilter& key_filter,
    typename Callbacks::Internal<T>::LoadKeysAndEntriesCallback callback) {
  LoadKeysAndEntriesWithFilter(key_filter, leveldb::ReadOptions(),
                               std::string(), std::move(callback));
}

template <typename P, typename T>
void FakeDB<P, T>::LoadKeysAndEntriesWithFilter(
    const KeyFilter& key_filter,
    const leveldb::ReadOptions& options,
    const std::string& target_prefix,
    typename Callbacks::Internal<T>::LoadKeysAndEntriesCallback callback) {
  auto keys_entries = std::make_unique<std::map<std::string, T>>();
  for (const auto& pair : *db_) {
    if (key_filter.is_null() || key_filter.Run(pair.first)) {
      if (pair.first.compare(0, target_prefix.length(), target_prefix) == 0)
        keys_entries->insert(pair);
    }
  }

  load_callback_ = base::BindOnce(RunLoadKeysAndEntriesCallback,
                                  std::move(callback), std::move(keys_entries));
}

template <typename P, typename T>
void FakeDB<P, T>::LoadKeysAndEntriesInRange(
    const std::string& start,
    const std::string& end,
    typename Callbacks::Internal<T>::LoadKeysAndEntriesCallback callback) {
  auto keys_entries = std::make_unique<std::map<std::string, T>>();
  for (const auto& pair : *db_) {
    std::string key = pair.first;
    if (start <= key && key <= end)
      keys_entries->insert(pair);
  }

  load_callback_ = base::BindOnce(RunLoadKeysAndEntriesCallback,
                                  std::move(callback), std::move(keys_entries));
}

template <typename P, typename T>
void FakeDB<P, T>::LoadKeys(Callbacks::LoadKeysCallback callback) {
  std::unique_ptr<std::vector<std::string>> keys(
      new std::vector<std::string>());
  for (const auto& pair : *db_)
    keys->push_back(pair.first);

  load_keys_callback_ =
      base::BindOnce(RunLoadKeysCallback, std::move(callback), std::move(keys));
}

template <typename P, typename T>
void FakeDB<P, T>::GetEntry(
    const std::string& key,
    typename Callbacks::Internal<T>::GetCallback callback) {
  std::unique_ptr<T> entry;
  auto it = db_->find(key);
  if (it != db_->end())
    entry.reset(new T(it->second));

  get_callback_ =
      base::BindOnce(RunGetCallback, std::move(callback), std::move(entry));
}

template <typename P, typename T>
void FakeDB<P, T>::Destroy(Callbacks::DestroyCallback callback) {
  db_->clear();
  destroy_callback_ = std::move(callback);
}

template <typename P, typename T>
base::FilePath& FakeDB<P, T>::GetDirectory() {
  return dir_;
}

template <typename P, typename T>
void FakeDB<P, T>::InitCallback(bool success) {
  std::move(init_callback_).Run(success);
}

template <typename P, typename T>
void FakeDB<P, T>::InitStatusCallback(Enums::InitStatus status) {
  std::move(init_status_callback_).Run(status);
}

template <typename P, typename T>
void FakeDB<P, T>::LoadCallback(bool success) {
  std::move(load_callback_).Run(success);
}

template <typename P, typename T>
void FakeDB<P, T>::LoadKeysCallback(bool success) {
  std::move(load_keys_callback_).Run(success);
}

template <typename P, typename T>
void FakeDB<P, T>::GetCallback(bool success) {
  std::move(get_callback_).Run(success);
}

template <typename P, typename T>
void FakeDB<P, T>::UpdateCallback(bool success) {
  std::move(update_callback_).Run(success);
}

template <typename P, typename T>
void FakeDB<P, T>::DestroyCallback(bool success) {
  std::move(destroy_callback_).Run(success);
}

// static
template <typename P, typename T>
void FakeDB<P, T>::RunLoadCallback(
    typename Callbacks::Internal<T>::LoadCallback callback,
    std::unique_ptr<typename std::vector<T>> entries,
    bool success) {
  std::move(callback).Run(success, std::move(entries));
}

// static
template <typename P, typename T>
void FakeDB<P, T>::RunLoadKeysAndEntriesCallback(
    typename Callbacks::Internal<T>::LoadKeysAndEntriesCallback callback,
    std::unique_ptr<typename std::map<std::string, T>> keys_entries,
    bool success) {
  std::move(callback).Run(success, std::move(keys_entries));
}

// static
template <typename P, typename T>
void FakeDB<P, T>::RunLoadKeysCallback(
    Callbacks::LoadKeysCallback callback,
    std::unique_ptr<std::vector<std::string>> keys,
    bool success) {
  std::move(callback).Run(success, std::move(keys));
}

// static
template <typename P, typename T>
void FakeDB<P, T>::RunGetCallback(
    typename Callbacks::Internal<T>::GetCallback callback,
    std::unique_ptr<T> entry,
    bool success) {
  std::move(callback).Run(success, std::move(entry));
}

// static
template <typename P, typename T>
base::FilePath FakeDB<P, T>::DirectoryForTestDB() {
  return base::FilePath(FILE_PATH_LITERAL("/fake/path"));
}

}  // namespace test
}  // namespace leveldb_proto

#endif  // COMPONENTS_LEVELDB_PROTO_TESTING_FAKE_DB_H_

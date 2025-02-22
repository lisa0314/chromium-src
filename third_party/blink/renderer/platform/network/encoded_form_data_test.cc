// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/platform/network/encoded_form_data_mojom_traits.h"

#include "base/sequenced_task_runner.h"
#include "base/test/scoped_task_environment.h"
#include "mojo/public/cpp/base/file_mojom_traits.h"
#include "mojo/public/cpp/base/file_path_mojom_traits.h"
#include "mojo/public/cpp/base/time_mojom_traits.h"
#include "mojo/public/cpp/bindings/string_traits_wtf.h"
#include "mojo/public/cpp/test_support/test_utils.h"
#include "services/network/public/mojom/url_loader.mojom-blink.h"
#include "third_party/blink/public/mojom/blob/blob.mojom-blink.h"
#include "third_party/blink/renderer/platform/network/encoded_form_data.h"

#include "testing/gtest/include/gtest/gtest.h"

namespace blink {

namespace {

class EncodedFormDataTest : public testing::Test {
 public:
  void CheckDeepCopied(const String& a, const String& b) {
    EXPECT_EQ(a, b);
    if (b.Impl())
      EXPECT_NE(a.Impl(), b.Impl());
  }

  void CheckDeepCopied(const KURL& a, const KURL& b) {
    EXPECT_EQ(a, b);
    CheckDeepCopied(a.GetString(), b.GetString());
    if (a.InnerURL() && b.InnerURL())
      CheckDeepCopied(*a.InnerURL(), *b.InnerURL());
  }

  void CheckDeepCopied(const FormDataElement& a, const FormDataElement& b) {
    EXPECT_EQ(a, b);
    CheckDeepCopied(a.filename_, b.filename_);
    CheckDeepCopied(a.blob_uuid_, b.blob_uuid_);
  }
};

TEST_F(EncodedFormDataTest, DeepCopy) {
  scoped_refptr<EncodedFormData> original(EncodedFormData::Create());
  original->AppendData("Foo", 3);
  original->AppendFileRange("example.txt", 12345, 56789, 9999.0);
  original->AppendBlob("originalUUID", nullptr);

  Vector<char> boundary_vector;
  boundary_vector.Append("----boundaryForTest", 19);
  original->SetIdentifier(45678);
  original->SetBoundary(boundary_vector);
  original->SetContainsPasswordData(true);

  scoped_refptr<EncodedFormData> copy = original->DeepCopy();

  // Check that contents are copied (compare the copy with expected values).
  const Vector<FormDataElement>& original_elements = original->Elements();
  const Vector<FormDataElement>& copy_elements = copy->Elements();
  ASSERT_EQ(3ul, copy_elements.size());

  Vector<char> foo_vector;
  foo_vector.Append("Foo", 3);

  EXPECT_EQ(FormDataElement::kData, copy_elements[0].type_);
  EXPECT_EQ(foo_vector, copy_elements[0].data_);

  EXPECT_EQ(FormDataElement::kEncodedFile, copy_elements[1].type_);
  EXPECT_EQ(String("example.txt"), copy_elements[1].filename_);
  EXPECT_EQ(12345ll, copy_elements[1].file_start_);
  EXPECT_EQ(56789ll, copy_elements[1].file_length_);
  EXPECT_EQ(9999.0, copy_elements[1].expected_file_modification_time_);

  EXPECT_EQ(FormDataElement::kEncodedBlob, copy_elements[2].type_);
  EXPECT_EQ(String("originalUUID"), copy_elements[2].blob_uuid_);

  EXPECT_EQ(45678, copy->Identifier());
  EXPECT_EQ(boundary_vector, copy->Boundary());
  EXPECT_EQ(true, copy->ContainsPasswordData());

  // Check that contents are copied (compare the copy with the original).
  EXPECT_EQ(*original, *copy);

  // Check pointers are different, i.e. deep-copied.
  ASSERT_NE(original.get(), copy.get());

  for (wtf_size_t i = 0; i < 3; ++i) {
    if (copy_elements[i].filename_.Impl()) {
      EXPECT_NE(original_elements[i].filename_.Impl(),
                copy_elements[i].filename_.Impl());
      EXPECT_TRUE(copy_elements[i].filename_.Impl()->HasOneRef());
    }

    if (copy_elements[i].blob_uuid_.Impl()) {
      EXPECT_NE(original_elements[i].blob_uuid_.Impl(),
                copy_elements[i].blob_uuid_.Impl());
      EXPECT_TRUE(copy_elements[i].blob_uuid_.Impl()->HasOneRef());
    }

    // m_optionalBlobDataHandle is not checked, because BlobDataHandle is
    // ThreadSafeRefCounted.
  }
}

TEST(EncodedFormDataMojomTraitsTest, Roundtrips_FormDataElement) {
  base::test::ScopedTaskEnvironment scoped_task_environment_;

  FormDataElement original1;
  original1.type_ = blink::FormDataElement::kData;
  original1.data_ = {'a', 'b', 'c', 'd'};
  FormDataElement copied1;
  EXPECT_TRUE(
      mojo::test::SerializeAndDeserialize<network::mojom::blink::DataElement>(
          &original1, &copied1));
  EXPECT_EQ(original1.type_, copied1.type_);
  EXPECT_EQ(original1.data_, copied1.data_);

  FormDataElement original2;
  original2.type_ = blink::FormDataElement::kEncodedFile;
  original2.file_start_ = 0;
  original2.file_length_ = 4;
  original2.filename_ = "file.name";
  original2.expected_file_modification_time_ = base::Time::Now().ToDoubleT();
  FormDataElement copied2;
  EXPECT_TRUE(
      mojo::test::SerializeAndDeserialize<network::mojom::blink::DataElement>(
          &original2, &copied2));
  EXPECT_EQ(original2.type_, copied2.type_);
  EXPECT_EQ(original2.file_start_, copied2.file_start_);
  EXPECT_EQ(original2.file_length_, copied2.file_length_);
  EXPECT_EQ(original2.filename_, copied2.filename_);
  EXPECT_EQ(original2.expected_file_modification_time_,
            copied2.expected_file_modification_time_);

  FormDataElement original3;
  original3.type_ = blink::FormDataElement::kEncodedBlob;
  original3.blob_uuid_ = "uuid-test";
  mojo::MessagePipe pipe;
  original3.optional_blob_data_handle_ = BlobDataHandle::Create(
      original3.blob_uuid_, "type-test", 100,
      mojom::blink::BlobPtrInfo(std::move(pipe.handle0), 0));
  FormDataElement copied3;
  EXPECT_TRUE(
      mojo::test::SerializeAndDeserialize<network::mojom::blink::DataElement>(
          &original3, &copied3));
  EXPECT_EQ(copied3.type_, blink::FormDataElement::kDataPipe);

  FormDataElement original4;
  original4.type_ = blink::FormDataElement::kDataPipe;
  network::mojom::blink::DataPipeGetterPtr data_pipe_getter;
  auto request = mojo::MakeRequest(&data_pipe_getter);
  original4.data_pipe_getter_ =
      base::MakeRefCounted<blink::WrappedDataPipeGetter>(
          std::move(data_pipe_getter));
  FormDataElement copied4;
  EXPECT_TRUE(
      mojo::test::SerializeAndDeserialize<network::mojom::blink::DataElement>(
          &original4, &copied4));
  EXPECT_TRUE(copied4.data_pipe_getter_);
}

}  // namespace
}  // namespace blink

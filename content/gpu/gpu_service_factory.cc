// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/gpu/gpu_service_factory.h"

#include <memory>

#include "base/task/post_task.h"
#include "base/threading/thread_task_runner_handle.h"
#include "build/build_config.h"
#include "services/ml/ml_service.h"
#include "services/ml/public/mojom/constants.mojom.h"
#include "services/shape_detection/public/mojom/constants.mojom.h"
#include "services/shape_detection/shape_detection_service.h"

#if BUILDFLAG(ENABLE_MOJO_MEDIA_IN_GPU_PROCESS)
#include "base/bind.h"
#include "media/mojo/interfaces/constants.mojom.h"      // nogncheck
#include "media/mojo/services/media_service_factory.h"  // nogncheck
#if BUILDFLAG(ENABLE_LIBRARY_CDMS)
#include "content/public/gpu/content_gpu_client.h"
#endif  // BUILDFLAG(ENABLE_LIBRARY_CDMS)
#endif  // BUILDFLAG(ENABLE_MOJO_MEDIA_IN_GPU_PROCESS)

namespace content {

GpuServiceFactory::GpuServiceFactory(
    const gpu::GpuPreferences& gpu_preferences,
    const gpu::GpuDriverBugWorkarounds& gpu_workarounds,
    const gpu::GpuFeatureInfo& gpu_feature_info,
    base::WeakPtr<media::MediaGpuChannelManager> media_gpu_channel_manager,
    media::AndroidOverlayMojoFactoryCB android_overlay_factory_cb) {
#if BUILDFLAG(ENABLE_MOJO_MEDIA_IN_GPU_PROCESS)
  gpu_preferences_ = gpu_preferences;
  gpu_workarounds_ = gpu_workarounds;
  gpu_feature_info_ = gpu_feature_info;
  task_runner_ = base::ThreadTaskRunnerHandle::Get();
  media_gpu_channel_manager_ = std::move(media_gpu_channel_manager);
  android_overlay_factory_cb_ = std::move(android_overlay_factory_cb);
#endif
}

GpuServiceFactory::~GpuServiceFactory() {}

void GpuServiceFactory::RunService(
    const std::string& service_name,
    mojo::PendingReceiver<service_manager::mojom::Service> receiver) {
  auto request = service_manager::mojom::ServiceRequest(std::move(receiver));
#if BUILDFLAG(ENABLE_MOJO_MEDIA_IN_GPU_PROCESS)
  if (service_name == media::mojom::kMediaServiceName) {
    media::CdmProxyFactoryCB cdm_proxy_factory_cb;
#if BUILDFLAG(ENABLE_LIBRARY_CDMS)
    cdm_proxy_factory_cb =
        base::BindRepeating(&ContentGpuClient::CreateCdmProxy,
                            base::Unretained(GetContentClient()->gpu()));
#endif  // BUILDFLAG(ENABLE_LIBRARY_CDMS)

    // This service will host audio/video decoders, and if these decoding
    // operations are blocked, user may hear audio glitch or see video freezing,
    // hence "user blocking".
    scoped_refptr<base::SingleThreadTaskRunner> task_runner;
#if defined(OS_WIN)
    // Run everything on the gpu main thread, since that's where the CDM runs.
    task_runner = task_runner_;
#else
    // TODO(crbug.com/786169): Check whether this needs to be single threaded.
    task_runner = base::CreateSingleThreadTaskRunnerWithTraits(
        {base::TaskPriority::USER_BLOCKING});
#endif  // defined(OS_WIN)

    using FactoryCallback =
        base::OnceCallback<std::unique_ptr<service_manager::Service>()>;
    FactoryCallback factory = base::BindOnce(
        &media::CreateGpuMediaService, std::move(request), gpu_preferences_,
        gpu_workarounds_, gpu_feature_info_, task_runner_,
        media_gpu_channel_manager_, android_overlay_factory_cb_,
        std::move(cdm_proxy_factory_cb));
    task_runner->PostTask(
        FROM_HERE, base::BindOnce(
                       [](FactoryCallback factory) {
                         service_manager::Service::RunAsyncUntilTermination(
                             std::move(factory).Run());
                       },
                       std::move(factory)));
    return;
  }
#endif  // BUILDFLAG(ENABLE_MOJO_MEDIA_IN_GPU_PROCESS)

  if (service_name == shape_detection::mojom::kServiceName) {
    service_manager::Service::RunAsyncUntilTermination(
        std::make_unique<shape_detection::ShapeDetectionService>(
            std::move(request)));
    return;
  }

  if (service_name == ml::mojom::kServiceName) {
    scoped_refptr<base::SingleThreadTaskRunner> task_runner =
        base::CreateSingleThreadTaskRunnerWithTraits(
            {base::TaskPriority::USER_BLOCKING});
    task_runner->PostTask(
        FROM_HERE,
        base::BindOnce(
            [](service_manager::mojom::ServiceRequest request) {
              service_manager::Service::RunAsyncUntilTermination(
                  std::make_unique<ml::MLService>(std::move(request)));
            },
            std::move(request)));
    return;
  }
}

}  // namespace content

/*
 * Copyright (C) 2008, 2009 Apple Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 2. Redistributions in binary form must reproduce the above copyright
 *
 * THIS SOFTWARE IS PROVIDED BY APPLE COMPUTER, INC. ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL APPLE COMPUTER, INC. OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef THIRD_PARTY_BLINK_RENDERER_CORE_WORKERS_WORKER_GLOBAL_SCOPE_H_
#define THIRD_PARTY_BLINK_RENDERER_CORE_WORKERS_WORKER_GLOBAL_SCOPE_H_

#include <memory>
#include "services/network/public/mojom/fetch_api.mojom-shared.h"
#include "services/service_manager/public/cpp/interface_provider.h"
#include "services/service_manager/public/mojom/interface_provider.mojom-blink.h"
#include "third_party/blink/public/mojom/script/script_type.mojom-blink.h"
#include "third_party/blink/renderer/bindings/core/v8/active_script_wrappable.h"
#include "third_party/blink/renderer/core/core_export.h"
#include "third_party/blink/renderer/core/dom/frame_request_callback_collection.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/core/frame/csp/content_security_policy.h"
#include "third_party/blink/renderer/core/frame/dom_timer_coordinator.h"
#include "third_party/blink/renderer/core/messaging/blink_transferable_message.h"
#include "third_party/blink/renderer/core/script/script.h"
#include "third_party/blink/renderer/core/workers/global_scope_creation_params.h"
#include "third_party/blink/renderer/core/workers/worker_animation_frame_provider.h"
#include "third_party/blink/renderer/core/workers/worker_or_worklet_global_scope.h"
#include "third_party/blink/renderer/core/workers/worker_settings.h"
#include "third_party/blink/renderer/platform/heap/handle.h"
#include "third_party/blink/renderer/platform/loader/fetch/cached_metadata_handler.h"
#include "third_party/blink/renderer/platform/wtf/casting.h"

namespace service_manager {
class InterfaceProvider;
}

namespace blink {

class ConsoleMessage;
class ExceptionState;
class FetchClientSettingsObjectSnapshot;
class FontFaceSet;
class OffscreenFontSelector;
class V8VoidFunction;
class StringOrTrustedScriptURL;
class TrustedTypePolicyFactory;
class WorkerLocation;
class WorkerNavigator;
class WorkerThread;

class CORE_EXPORT WorkerGlobalScope
    : public WorkerOrWorkletGlobalScope,
      public ActiveScriptWrappable<WorkerGlobalScope>,
      public Supplementable<WorkerGlobalScope> {
  DEFINE_WRAPPERTYPEINFO();
  USING_GARBAGE_COLLECTED_MIXIN(WorkerGlobalScope);

 public:
  ~WorkerGlobalScope() override;

  // Returns null if caching is not supported.
  virtual SingleCachedMetadataHandler* CreateWorkerScriptCachedMetadataHandler(
      const KURL& script_url,
      const Vector<uint8_t>* meta_data) {
    return nullptr;
  }

  // WorkerOrWorkletGlobalScope
  bool IsClosing() const final { return closing_; }
  void Dispose() override;
  WorkerThread* GetThread() const final { return thread_; }

  void ExceptionUnhandled(int exception_id);

  // WorkerGlobalScope
  WorkerGlobalScope* self() { return this; }
  WorkerLocation* location() const;
  WorkerNavigator* navigator() const override;
  void close();
  bool isSecureContextForBindings() const {
    return ExecutionContext::IsSecureContext();
  }

  String origin() const;

  DEFINE_ATTRIBUTE_EVENT_LISTENER(error, kError)
  DEFINE_ATTRIBUTE_EVENT_LISTENER(languagechange, kLanguagechange)
  DEFINE_ATTRIBUTE_EVENT_LISTENER(rejectionhandled, kRejectionhandled)
  DEFINE_ATTRIBUTE_EVENT_LISTENER(unhandledrejection, kUnhandledrejection)

  // WorkerUtils
  virtual void importScripts(const HeapVector<StringOrTrustedScriptURL>& urls,
                             ExceptionState&);

  // ExecutionContext
  const KURL& Url() const final;
  KURL CompleteURL(const String&) const final;
  bool IsWorkerGlobalScope() const final { return true; }
  bool IsContextThread() const final;
  const KURL& BaseURL() const final;
  String UserAgent() const final { return user_agent_; }
  HttpsState GetHttpsState() const override { return https_state_; }
  const base::UnguessableToken& GetAgentClusterID() const final {
    return agent_cluster_id_;
  }

  void InitializeURL(const KURL& url);

  DOMTimerCoordinator* Timers() final { return &timers_; }
  SecurityContext& GetSecurityContext() final { return *this; }
  void AddConsoleMessage(ConsoleMessage*) final;
  bool IsSecureContext(String& error_message) const override;
  service_manager::InterfaceProvider* GetInterfaceProvider() final;

  OffscreenFontSelector* GetFontSelector() { return font_selector_; }

  CoreProbeSink* GetProbeSink() final;
  const base::UnguessableToken& GetParentDevToolsToken() {
    return parent_devtools_token_;
  }

  // EventTarget
  ExecutionContext* GetExecutionContext() const final;
  bool IsWindowOrWorkerGlobalScope() const final { return true; }

  // These methods should be called in the scope of a pausable
  // task runner. ie. They should not be called when the context
  // is paused.
  void EvaluateClassicScript(const KURL& script_url,
                             String source_code,
                             std::unique_ptr<Vector<uint8_t>> cached_meta_data,
                             const v8_inspector::V8StackTraceId& stack_id);

  // Fetches and evaluates the top-level classic script.
  virtual void FetchAndRunClassicScript(
      const KURL& script_url,
      const FetchClientSettingsObjectSnapshot& outside_settings_object,
      const v8_inspector::V8StackTraceId& stack_id) = 0;

  // Fetches and evaluates the top-level module script.
  virtual void FetchAndRunModuleScript(
      const KURL& module_url_record,
      const FetchClientSettingsObjectSnapshot& outside_settings_object,
      network::mojom::FetchCredentialsMode) = 0;

  void ReceiveMessage(BlinkTransferableMessage);
  base::TimeTicks TimeOrigin() const { return time_origin_; }
  WorkerSettings* GetWorkerSettings() const { return worker_settings_.get(); }

  void Trace(blink::Visitor*) override;

  // TODO(fserb): This can be removed once we WorkerGlobalScope implements
  // FontFaceSource on the IDL.
  FontFaceSet* fonts();

  // https://html.spec.whatwg.org/C/#windoworworkerglobalscope-mixin
  void queueMicrotask(V8VoidFunction*);

  int requestAnimationFrame(V8FrameRequestCallback* callback, ExceptionState&);
  void cancelAnimationFrame(int id);

  WorkerAnimationFrameProvider* GetAnimationFrameProvider() {
    return animation_frame_provider_;
  }

  TrustedTypePolicyFactory* trustedTypes();

 protected:
  WorkerGlobalScope(std::unique_ptr<GlobalScopeCreationParams>,
                    WorkerThread*,
                    base::TimeTicks time_origin);

  // ExecutionContext
  void ExceptionThrown(ErrorEvent*) override;
  void RemoveURLFromMemoryCache(const KURL&) final;

  // Evaluates the given top-level classic script.
  virtual void EvaluateClassicScriptInternal(
      const KURL& script_url,
      String source_code,
      std::unique_ptr<Vector<uint8_t>> cached_meta_data);

  mojom::ScriptType GetScriptType() const { return script_type_; }

  GlobalScopeCSPApplyMode GetCSPApplyMode() const { return csp_apply_mode_; }

 private:
  void SetWorkerSettings(std::unique_ptr<WorkerSettings>);

  // Used for importScripts().
  void ImportScriptsInternal(const Vector<String>& urls, ExceptionState&);
  bool FetchClassicImportedScript(
      const KURL& script_url,
      KURL* out_response_url,
      String* out_source_code,
      std::unique_ptr<Vector<uint8_t>>* out_cached_meta_data);

  // ExecutionContext
  EventTarget* ErrorEventTarget() final { return this; }

  KURL url_;
  const mojom::ScriptType script_type_;
  const String user_agent_;
  const base::UnguessableToken parent_devtools_token_;
  std::unique_ptr<WorkerSettings> worker_settings_;

  mutable Member<WorkerLocation> location_;
  mutable TraceWrapperMember<WorkerNavigator> navigator_;
  Member<TrustedTypePolicyFactory> trusted_types_;

  WorkerThread* thread_;

  bool closing_ = false;

  DOMTimerCoordinator timers_;

  const base::TimeTicks time_origin_;

  HeapHashMap<int, Member<ErrorEvent>> pending_error_events_;
  int last_pending_error_event_id_ = 0;

  Member<OffscreenFontSelector> font_selector_;
  TraceWrapperMember<WorkerAnimationFrameProvider> animation_frame_provider_;

  service_manager::InterfaceProvider interface_provider_;

  const base::UnguessableToken agent_cluster_id_;

  HttpsState https_state_;

  GlobalScopeCSPApplyMode csp_apply_mode_;
};

template <>
struct DowncastTraits<WorkerGlobalScope> {
  static bool AllowFrom(const ExecutionContext& context) {
    return context.IsWorkerGlobalScope();
  }
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_CORE_WORKERS_WORKER_GLOBAL_SCOPE_H_

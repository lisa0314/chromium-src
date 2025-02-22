// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/core/html/portal/html_portal_element.h"

#include <utility>
#include "services/service_manager/public/cpp/interface_provider.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/bindings/core/v8/serialization/serialized_script_value.h"
#include "third_party/blink/renderer/core/dom/document.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/dom/node.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/core/frame/local_frame.h"
#include "third_party/blink/renderer/core/frame/local_frame_client.h"
#include "third_party/blink/renderer/core/frame/remote_frame.h"
#include "third_party/blink/renderer/core/html/html_unknown_element.h"
#include "third_party/blink/renderer/core/html/parser/html_parser_idioms.h"
#include "third_party/blink/renderer/core/html/portal/document_portals.h"
#include "third_party/blink/renderer/core/html/portal/portal_activate_options.h"
#include "third_party/blink/renderer/core/html_names.h"
#include "third_party/blink/renderer/core/inspector/console_message.h"
#include "third_party/blink/renderer/core/inspector/thread_debugger.h"
#include "third_party/blink/renderer/core/layout/layout_iframe.h"
#include "third_party/blink/renderer/core/messaging/message_port.h"
#include "third_party/blink/renderer/platform/bindings/script_state.h"
#include "third_party/blink/renderer/platform/heap/handle.h"
#include "third_party/blink/renderer/platform/runtime_enabled_features.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

namespace blink {

HTMLPortalElement::HTMLPortalElement(Document& document)
    : HTMLFrameOwnerElement(html_names::kPortalTag, document) {}

HTMLPortalElement::~HTMLPortalElement() {}

void HTMLPortalElement::Trace(Visitor* visitor) {
  HTMLFrameOwnerElement::Trace(visitor);
  visitor->Trace(portal_frame_);
}

HTMLElement* HTMLPortalElement::Create(Document& document) {
  if (RuntimeEnabledFeatures::PortalsEnabled())
    return MakeGarbageCollected<HTMLPortalElement>(document);
  return HTMLUnknownElement::Create(html_names::kPortalTag, document);
}

void HTMLPortalElement::Navigate() {
  KURL url = GetNonEmptyURLAttribute(html_names::kSrcAttr);
  if (!url.IsEmpty() && portal_ptr_) {
    portal_ptr_->Navigate(url);
  }
}

namespace {

BlinkTransferableMessage ActivateDataAsMessage(
    ScriptState* script_state,
    PortalActivateOptions* options,
    ExceptionState& exception_state) {
  v8::Isolate* isolate = script_state->GetIsolate();
  Transferables transferables;
  if (options->hasTransfer()) {
    if (!SerializedScriptValue::ExtractTransferables(
            script_state->GetIsolate(), options->transfer(), transferables,
            exception_state))
      return {};
  }

  SerializedScriptValue::SerializeOptions serialize_options;
  serialize_options.transferables = &transferables;
  v8::Local<v8::Value> data = options->hasData()
                                  ? options->data().V8Value()
                                  : v8::Null(isolate).As<v8::Value>();

  BlinkTransferableMessage msg;
  msg.message = SerializedScriptValue::Serialize(
      isolate, data, serialize_options, exception_state);
  if (!msg.message)
    return {};

  msg.message->UnregisterMemoryAllocatedWithCurrentScriptContext();

  auto* execution_context = ExecutionContext::From(script_state);
  msg.ports = MessagePort::DisentanglePorts(
      execution_context, transferables.message_ports, exception_state);
  if (exception_state.HadException())
    return {};

  // msg.user_activation is left out; we will probably handle user activation
  // explicitly for activate data.
  // TODO(crbug.com/936184): Answer this for good.

  if (ThreadDebugger* debugger = ThreadDebugger::From(isolate))
    msg.sender_stack_trace_id = debugger->StoreCurrentStackTrace("activate");

  if (msg.message->IsLockedToAgentCluster()) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kDataCloneError,
        "Cannot send agent cluster-locked data (e.g. SharedArrayBuffer) "
        "through portal activation.");
    return {};
  }

  return msg;
}

}  // namespace

ScriptPromise HTMLPortalElement::activate(ScriptState* script_state,
                                          PortalActivateOptions* options) {
  ScriptPromiseResolver* resolver = ScriptPromiseResolver::Create(script_state);
  ScriptPromise promise = resolver->Promise();

  ExceptionState exception_state(script_state->GetIsolate(),
                                 ExceptionState::kExecutionContext,
                                 "HTMLPortalElement", "activate");

  if (!portal_ptr_) {
    exception_state.ThrowDOMException(
        DOMExceptionCode::kInvalidStateError,
        "The HTMLPortalElement is not associated with a portal context.");
    resolver->Reject(exception_state);
    return promise;
  }

  BlinkTransferableMessage data =
      ActivateDataAsMessage(script_state, options, exception_state);
  if (exception_state.HadException()) {
    resolver->Reject(exception_state);
    return promise;
  }

  // The HTMLPortalElement is bound as a persistent so that it won't get
  // garbage collected while there is a pending callback. This is necessary
  // because the HTMLPortalElement owns the mojo interface, so if it were
  // garbage collected the callback would never be called and the promise
  // would never be resolved.
  portal_ptr_->Activate(
      std::move(data),
      WTF::Bind([](HTMLPortalElement* portal,
                   ScriptPromiseResolver* resolver) { resolver->Resolve(); },
                WrapPersistent(this), WrapPersistent(resolver)));
  return promise;
}

HTMLPortalElement::InsertionNotificationRequest HTMLPortalElement::InsertedInto(
    ContainerNode& node) {
  auto result = HTMLFrameOwnerElement::InsertedInto(node);

  if (!node.IsInDocumentTree() || !GetDocument().IsHTMLDocument() ||
      !GetDocument().GetFrame())
    return result;

  // We don't support embedding portals in nested browsing contexts.
  if (!GetDocument().GetFrame()->IsMainFrame()) {
    GetDocument().AddConsoleMessage(ConsoleMessage::Create(
        kRenderingMessageSource, mojom::ConsoleMessageLevel::kWarning,
        "Cannot use <portal> in a nested browsing context."));
    return result;
  }

  std::tie(portal_frame_, portal_token_) =
      GetDocument().GetFrame()->Client()->CreatePortal(
          this, mojo::MakeRequest(&portal_ptr_));
  DocumentPortals::From(GetDocument()).OnPortalInserted(this);
  Navigate();

  return result;
}

void HTMLPortalElement::RemovedFrom(ContainerNode& node) {
  HTMLFrameOwnerElement::RemovedFrom(node);

  Document& document = GetDocument();

  if (node.IsInDocumentTree() && document.IsHTMLDocument()) {
    // The portal creation is asynchronous, and the Document only gets notified
    // after the element receives a callback from the browser that assigns its
    // token, so we need to check whether that has been completed before
    // notifying the document about the portal's removal.
    if (!portal_token_.is_empty())
      DocumentPortals::From(GetDocument()).OnPortalRemoved(this);

    portal_token_ = base::UnguessableToken();
    portal_ptr_.reset();
  }
}

bool HTMLPortalElement::IsURLAttribute(const Attribute& attribute) const {
  return attribute.GetName() == html_names::kSrcAttr ||
         HTMLFrameOwnerElement::IsURLAttribute(attribute);
}

void HTMLPortalElement::ParseAttribute(
    const AttributeModificationParams& params) {
  HTMLFrameOwnerElement::ParseAttribute(params);

  if (params.name == html_names::kSrcAttr)
    Navigate();
}

LayoutObject* HTMLPortalElement::CreateLayoutObject(
    const ComputedStyle& style) {
  return new LayoutIFrame(this);
}

void HTMLPortalElement::AttachLayoutTree(AttachContext& context) {
  HTMLFrameOwnerElement::AttachLayoutTree(context);

  if (GetLayoutEmbeddedContent() && ContentFrame())
    SetEmbeddedContentView(ContentFrame()->View());
}

}  // namespace blink

# `thinwrap/llm` (Go) — contributor guide

This folder (`.ai/`) is for developers — and the coding agents working alongside
them — who are **changing this library**: adding a provider or improving the
package. It is not usage documentation.

> **Using the package in your app?** See [`../README.md`](../README.md) and the
> per-provider pages under [`../docs/providers/`](../docs/providers). `.ai/` is
> contributor-only.

## Map of this folder

- **guidelines.md** (this file) — entry point + the two "add a provider" recipes.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — facade → dispatch → connector, the two connector tiers, and the invariants.
- [`CONVENTIONS.md`](./CONVENTIONS.md) — file layout, naming, error mapping, streaming, testing.

## The shape in one sentence

A consumer constructs the `Chat` facade by provider id + config
(`llm.NewChat(llm.OpenAI, llm.OpenAICompatConfig{APIKey: k})`); the facade
dispatches to a connector that turns the normalized `ChatInput` into the vendor
wire and the vendor response back into the normalized `ChatResult` /
`ChatStreamDelta` / `*ConnectorError`.

## Two connector tiers

1. **First-class OpenAI-compatible (15 providers).** They share ONE connector —
   `openai_compat.go` — parameterized by a row in the `specs` registry
   (`spec.go`). Adding one is a spec row, not a new type.
2. **Native adapters (Anthropic, Bedrock, Gemini).** Structurally different wire
   (Messages / Converse / generateContent). Each is its own `<id>.go` with a real
   connector that emits the **identical** normalized surface, self-registering via
   `init()` → `registerNativeChat`.

## Setup & verify

```bash
go build ./... && go vet ./...
gofmt -l .            # must print nothing
go test -cover ./...
```

Go ≥ 1.18. **Zero third-party dependencies — do not add any.** Bedrock's SigV4 is
hand-rolled in `awssigv4.go`; SSE streaming is parsed in `sse.go` on `bufio`.
Avoid Go 1.21+ builtins (`min`/`max`) and the `slices`/`maps` packages (1.18 floor).

## Add a first-class OpenAI-compatible provider

1. **Register the id** — add the `ProviderID` const in [`provider.go`](../provider.go) (+ to `ProviderIDs`).
2. **Add the spec row** — in [`spec.go`](../spec.go): `id`, `defaultBaseURL`, `buildAuthHeaders`, and any quirk (`supportsTools`, `maxTokensField`, `streamUsagePath`, `strictParams`). That's the whole connector. Add to `embeddingsProviderIDs` too if it exposes an OpenAI-float `/embeddings`.
3. **Test** — add a case to `openai_compat_test.go`.
4. **Docs** — `docs/providers/<id>.md`.

No facade change is needed — `NewChat` dispatches every `specs` id to the shared connector.

## Add a native-adapter provider

Copy [`anthropic.go`](../anthropic.go). Touch-points:

1. **Register the id** — `provider.go`.
2. **Config** — add `<Name>Config` (implementing `isChatConfig()`) to [`config.go`](../config.go).
3. **Connector** — `<id>.go`: `func init() { registerNativeChat(<Const>, new<Name>Chat) }`; a `new<Name>Chat(cfg ChatConfig, o facadeOptions) (chatConnector, error)` that type-asserts the config; a `<id>Connector` embedding `baseConnector` with `complete()` + `stream()` + a private `<id>MapCode`. All wire translation is local.
4. **Test** — `<id>_test.go`.
5. **Docs** — `docs/providers/<id>.md`.

### Definition of done (the CI gates)

```bash
go build ./... && go vet ./... && gofmt -l . && go test -cover ./...   # ≥80%
```

## Invariants you must not break

- **Zero dependencies / no vendor SDKs.** SigV4 hand-rolled on the stdlib.
- **Stateless wrapper.** No caching, retries, idempotency keys, conversation
  state, or telemetry. Prompt caching is a consumer concern via `Passthrough`.
- **Canonical shape = OpenAI Chat Completions.** One `Chat` facade; connectors may
  target other wires under the hood.
- **≥90% baseline-coverage rule.** A field belongs on `ChatInput`/`ChatResult`
  only if ≥90% of providers support it normalizably; everything else is
  `Passthrough` (in) / `Raw` (out) — never emulated. The deliberate sub-90%
  carve-out is `Reasoning.Effort`.
- **Every connector emits the identical `ChatInput`/`ChatResult`/`ChatStreamDelta`/`*ConnectorError`.** That parity is the whole product.
- **Seven `ProviderCode` values**, surfaced via `*ConnectorError`; retry hints
  ride in `Cause["retryAfter"]` / `Cause["retryAfterSeconds"]` — never a top-level field.
- **Go idioms:** return `(*Result, error)` with a `*ConnectorError` reachable via
  `errors.As`; stream via `*ChatStream` (`Recv` → `io.EOF`); thread `context.Context`.

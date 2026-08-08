# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.0.1] — 2026-08-08

### Changed

- `HTTPClient` documents its contract: a non-2xx must be RETURNED, never reported as an error.

## [1.0.0] — 2026-07-24

### Added

- Initial Go port of `@thinwrap/llm` and `thinwrap/llm`: `Chat` (`Complete` + pull-based
  `Stream`) and `Embeddings` facades over 18 providers, with a typed `*ConnectorError`.
- Fifteen OpenAI-compatible providers share one connector driven by a spec registry; Anthropic,
  Bedrock and Gemini have bespoke wire adapters emitting identical shapes.
- Zero third-party dependencies: Bedrock SigV4 hand-rolled on the standard library, SSE parsed
  on `bufio`.

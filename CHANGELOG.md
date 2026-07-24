# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Initial Go port of `@thinwrap/llm` (TypeScript) and `thinwrap/llm` (PHP): a
  `Chat` facade (`Complete` + pull-based `Stream`) and an `Embeddings` facade over
  18 providers behind one normalized `ChatInput` / `ChatResult` /
  `ChatStreamDelta` surface, with a single typed `*ConnectorError`.
- Fifteen first-class OpenAI-compatible providers share one connector driven by a
  per-provider spec registry; three natives — Anthropic (Messages), Bedrock
  (Converse, SigV4), Gemini (generateContent) — have bespoke wire adapters that
  emit the identical normalized shapes.
- Zero third-party dependencies: Bedrock SigV4 is hand-rolled on the standard
  library; SSE streaming is parsed on `bufio`.

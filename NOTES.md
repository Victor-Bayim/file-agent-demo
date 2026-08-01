# Implementation Notes

## Loop and termination

The handwritten loop performs model completion, native Tool Call parsing, schema validation,
deterministic tool execution, immediate Trace emission, Tool Result feedback, and then either
continues or returns final text. It stops on a final assistant response, model-turn or tool-call
limit, Token budget, elapsed-time limit, repeated identical calls, explicit cancellation,
unsupported finish reason, invalid tool batch, or structured model/tool error. The Web and CLI
share this exact loop.

## Context selection

Context contains the trusted System message, the user's task, assistant Tool Calls, and bounded,
sanitized Tool Results needed for the next decision. It does not contain whole large files,
unbounded prior results, server paths, credentials, full write bodies in Web events, or hidden
chain-of-thought. Large evidence is searched or read in pages, and older tool history is compacted
under an explicit character ceiling.

## Key trade-off

Only one mutating tool call is accepted in a model response, and it cannot be mixed with reads.
This sacrifices parallel write throughput, but makes ordering, observation versions, mutation
accounting, post-write verification, and audit review substantially clearer. For a file assistant
operating on untrusted workspace data, that predictability is more valuable than speculative
concurrency.

## Known next step

A production multi-instance version should move Sessions, sliding-window rate limits, and durable
Run state to Redis and a database, with explicit trusted-proxy configuration. This submission
intentionally remains a single-worker demonstration: temporary Session copies and in-memory
coordination keep the security boundaries visible and testable without pretending to provide
cross-instance consistency or persistent history.

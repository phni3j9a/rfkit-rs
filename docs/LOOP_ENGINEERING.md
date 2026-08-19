# Loop Engineering

`rfkit-rs` may be grown through a semi-automated maintainer loop in which ChatGPT owns repository-level prioritization and review, GitHub Issues are the handoff contract, and Codex owns implementation planning and pull-request creation.

The goal is **safe continuous improvement**, not maximum feature throughput.

## Roles

### ChatGPT Maintainer

The maintainer owns:

- repository-health assessment;
- pull-request review and merge readiness;
- Issue triage;
- deciding what should be worked on next;
- defining what / why / done in implementation-ready Issues;
- dispatching at most one ready Issue by default;
- escalating decisions that require human RF/product/policy judgment.

The maintainer does not prescribe routine repository-specific implementation details. Codex Main owns implementation planning after inspecting the repository.

### Codex Worker

The scheduled Codex worker owns:

- finding a dispatched Issue;
- claiming it before implementation;
- reading `AGENTS.md` and the Issue contract;
- repository inspection and implementation planning;
- implementation through the configured IssueFlow `issue-to-pr` workflow;
- deterministic verification;
- independent review;
- Git operations and opening the PR.

### Human owner

Human attention should normally be required only for escalated decisions such as public-API policy, ambiguous RF semantics, unavailable paid standards/papers, provenance/licensing uncertainty, major architecture changes, or release/publishing actions.

## Dispatch state

Use these GitHub labels as the machine-queryable handoff state:

- `loop:ready` — the Issue contract is ready for Codex implementation;
- `loop:in-progress` — a Codex worker has claimed the Issue;
- `loop:blocked` — implementation cannot currently proceed.

The normal transition is:

```text
open Issue
   ↓ ChatGPT triage
loop:ready
   ↓ Codex claim
loop:in-progress
   ↓ implementation + review
Pull Request
   ↓ ChatGPT maintainer review
merge / request changes / escalate
```

An open Issue without `loop:ready` is **not** permission for scheduled Codex implementation.

If these labels are not present, scheduled automation must fail closed rather than silently inventing a different dispatch mechanism.

## Work-in-progress policy

Default to **WIP = 1**.

If a `loop:ready` or `loop:in-progress` Issue already exists, ChatGPT should not create another speculative implementation Issue.

This keeps roadmap generation adaptive: after each merged increment, the next task is selected from the new `main` state rather than from a long pre-generated AI backlog.

Existing user-reported Issues may remain open as backlog or discussion items without counting as dispatched WIP unless they carry the ready/in-progress state.

## Maintainer cycle

A scheduled maintainer run should use fresh GitHub state and process work in this order:

1. inspect repository policy and `main` health;
2. inspect open PRs before generating new work;
3. merge a clearly ready PR when authorized, or request bounded changes;
4. triage open Issues that may take precedence;
5. check for `loop:ready` / `loop:in-progress` work;
6. only when the queue is empty, select one next bounded increment;
7. create/update the Issue contract and apply `loop:ready`;
8. report mutations and any escalation concisely.

A run with no mutation is valid. Do not create an Issue merely to prove that the loop ran.

## rfkit-rs prioritization policy

Use repository evidence rather than blindly consuming the README scope top to bottom.

Prefer work in roughly this order:

1. **broken main / correctness regressions** — failing required CI, wrong numerical behavior, invalid invariants, serious compatibility regressions;
2. **conformance debt** — missing or weak mathematical definition, deterministic tests, RF invariant/property tests, scikit-rf differential coverage, unjustified tolerance handling;
3. **foundational capability** — primitives that unlock several downstream RF operations while preserving the core N-port/complex-Z0 model;
4. **high-value user-facing RF capability** — a bounded vertical increment from the project scope;
5. **ergonomics / optimization / cleanup** — after correctness and conformance are characterized.

Do not optimize for feature count.

## Definition of merge-ready for numerical work

For substantive numerical features, review against `docs/CONFORMANCE.md` and `AGENTS.md`.

A PR should not be treated as complete merely because it compiles or CI is green. Where applicable, verify evidence for:

- documented mathematical behavior;
- deterministic Rust unit tests;
- RF invariant/property tests;
- differential comparison against the pinned scikit-rf oracle;
- justified tolerances;
- N-port behavior;
- scalar/per-port/frequency-dependent and complex reference impedances where supported by the operation;
- near-singular or ill-conditioned behavior when relevant;
- provenance/licensing requirements.

## Choosing the next Issue

When there is no existing dispatched work, the maintainer may consider several candidate directions internally, but should create **one** Issue only.

A good next Issue is:

- aligned with the current project goals;
- small enough for independent review and merge;
- useful on its own or clearly unlocks subsequent work;
- objectively verifiable;
- product-complete enough that Codex does not need to guess externally visible behavior;
- free of unnecessary implementation prescriptions.

Avoid umbrella Issues such as "implement all Touchstone support" or "port scikit-rf" as scheduled coding tasks. Prefer a bounded vertical slice with explicit compatibility and acceptance criteria.

## Escalation boundary

ChatGPT should stop generation and ask the human owner when a material decision involves:

- breaking or freezing a public API;
- choosing between multiple plausible RF definitions or wave conventions without a repository rule selecting one;
- disagreement between a published standard, mathematical source, and observed scikit-rf behavior that changes externally visible semantics;
- a paid IEEE/industry specification, paper, dataset, fixture, or other unavailable source needed to decide correctness;
- uncertain copyright, license, attribution, or provenance obligations;
- a major crate-boundary or repository-wide architecture change;
- release, crates.io publishing, signing, or other irreversible external publication;
- security or safety policy decisions beyond a bounded defect fix.

The escalation should state the decision needed and the smallest useful set of options. Do not create implementation work that assumes an unresolved answer.

## Suggested schedule

A simple initial cadence is:

```text
09:00 JST  ChatGPT maintainer cycle
10:00 JST  Codex worker checks for one loop:ready Issue
next day  ChatGPT reviews the resulting PR before dispatching more work
```

The exact clock times may change. The important invariant is **maintainer review before new generation** and **explicit Issue dispatch before Codex implementation**.

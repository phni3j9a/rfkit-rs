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

Use repository evidence and **marginal value**, rather than blindly consuming either the README scope or the conformance matrix top to bottom.

Prefer work in roughly this order:

1. **broken main / correctness regressions** — failing required CI, wrong numerical behavior, invalid invariants, serious compatibility regressions;
2. **blocking conformance debt** — missing or weak evidence that creates a concrete correctness risk, blocks a safe capability/API decision, or is implicated by an observed disagreement or regression;
3. **foundational capability** — primitives that unlock several downstream RF operations while preserving the core N-port/complex-Z0 model;
4. **high-value user-facing RF capability** — a bounded vertical increment from the project scope;
5. **non-blocking characterization / conformance** — useful additional coverage that is not tied to a concrete blocker, defect, or imminent capability risk;
6. **ergonomics / optimization / cleanup** — after correctness and conformance are characterized enough for the current stage.

Conformance debt is **not** synonymous with every dimension listed in `docs/CONFORMANCE.md` lacking an explicit dedicated fixture or Issue. Do not optimize for feature count, but also do not optimize for conformance-checkbox count.

## Avoiding conformance tunnels and diminishing returns

`docs/CONFORMANCE.md` is a **test-design matrix, not an autonomous roadmap checklist**. Its dimensions describe evidence that should be considered where relevant; they are not instructions to create one Issue per unchecked dimension.

When selecting autonomous work:

- A missing coverage dimension by itself is insufficient reason for a standalone Issue. A conformance-only Issue should identify the concrete failure mode it could catch, why existing evidence is insufficient, and why that risk should be retired before the next capability increment.
- Prefer adding proportionate conformance coverage together with the next capability when doing so keeps the change independently reviewable. Split conformance into a separate Issue when it blocks safe implementation/review, responds to observed evidence, or materially reduces a realistic numerical risk.
- Reuse existing fixtures and proofs. Do not create a standalone Issue merely to add metadata, labels, or a second certificate for a property that existing tests already establish strongly enough for the current stage unless the new evidence can catch a materially different defect class.
- For composed operations, test the risks introduced by the composition. Do not mechanically duplicate every lower-level edge-case matrix when the constituent kernels are already verified; use direct differential evidence, meaningful composition invariants, and coverage for composition-specific validation/failure paths in proportion to the new risk.
- After **two consecutive merged conformance-only increments**, if `main` is healthy and there is no concrete correctness defect, blocker, or newly exposed numerical risk, the next autonomous Issue should normally advance a foundational or user-facing capability. Choosing another conformance-only increment requires a specific repository-evidence justification.
- Before dispatching work, consider a small set of plausible candidate directions internally and choose the one with the best marginal value across **risk reduction, capability unlocked, user value, and implementation/review cost**. Do not publish a speculative long roadmap merely to record that comparison.
- If all available next increments have low marginal value, doing nothing is preferable to manufacturing work.

These rules do not weaken the merge bar. They prevent high-quality verification from turning into self-perpetuating verification work that crowds out useful RF capability.

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

Apply this list **proportionately to the risk introduced by the change**. It is not necessary for every operation or composition layer to receive a bespoke fixture for every conformance dimension when lower-level evidence plus focused end-to-end verification already establishes the relevant behavior.

## Choosing the next Issue

When there is no existing dispatched work, the maintainer should consider several plausible candidate directions internally, but should create **one** Issue only.

Choose by marginal value rather than by whichever unchecked README/conformance item is easiest to name. In particular, distinguish blocking conformance work from useful-but-non-blocking characterization before deciding that conformance should precede capability growth.

A good next Issue is:

- aligned with the current project goals;
- small enough for independent review and merge;
- useful on its own or clearly unlocks subsequent work;
- objectively verifiable;
- product-complete enough that Codex does not need to guess externally visible behavior;
- justified by current repository evidence rather than checklist completion;
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

After the initial once-daily rollout has been validated, use this cadence:

```text
09:00 JST  ChatGPT maintainer cycle
10:00 JST  Codex worker checks for one loop:ready Issue
21:00 JST  ChatGPT maintainer cycle
22:00 JST  Codex worker checks for one loop:ready Issue
```

The exact clock times may change. The important invariant is **maintainer review before new generation** and **explicit Issue dispatch before Codex implementation**.

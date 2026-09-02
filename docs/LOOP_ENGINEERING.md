# Loop Engineering

`rfkit-rs` may be grown through a bounded autonomous loop in which GitHub is the shared state, a scheduled Codex Planner chooses the next increment, and a scheduled Codex Worker implements one dispatched Issue.

The goal is **safe meaningful continuous improvement**, not maximum activity or feature throughput.

Read `docs/DEVELOPMENT_DIRECTION.md` first for the repository north star and current development horizon.

## Roles

### Codex Planner

The scheduled planner owns repository-level progress decisions during normal autonomous operation:

- inspect fresh `main`, repository policy, open PRs, Issues, CI, and loop state;
- protect repository health before generating new work;
- assess whether existing PR/Issue work must finish first;
- choose the next bounded increment by marginal value;
- define what / why / done in one implementation-ready Issue;
- apply `loop:ready` only when the Issue is genuinely ready for implementation;
- merge a clearly merge-ready autonomous PR when policy permits;
- request bounded changes or mark work blocked when appropriate;
- escalate decisions that require human RF/product/policy judgment.

The planner decides **what should happen next**, but does not pre-prescribe routine implementation details. The worker owns repository-specific implementation planning after claiming the Issue.

### Codex Worker

The scheduled worker owns execution of one dispatched Issue:

- find exactly one `loop:ready` Issue;
- claim it before implementation;
- read `AGENTS.md`, repository policy, and the Issue contract;
- inspect the repository and create the implementation plan;
- implement through the configured IssueFlow `issue-to-pr` workflow;
- run deterministic verification;
- obtain independent review;
- perform Git operations and open the PR.

### ChatGPT governor / auditor

ChatGPT is outside the normal scheduled loop. It does not need to generate routine Issues.

When asked by the human owner, ChatGPT should inspect fresh GitHub state and recent development history to determine whether the autonomous system is making **meaningful RF progress**. The audit should look for capability growth, correctness improvement, downstream leverage, repeated low-value work, conformance tunnels, cleanup/refactor loops, over-engineering, weak Issue selection, and policy drift.

If the loop is optimizing the wrong thing, ChatGPT should recommend or apply bounded policy changes rather than manually micromanaging the next few Issues.

### Human owner

Human attention should normally be required only for escalated decisions such as public-API policy, ambiguous RF semantics, unavailable paid standards/papers, provenance/licensing uncertainty, major architecture changes, release/publishing actions, or a deliberate change to autonomous-development policy.

## Dispatch state

Use these GitHub labels as machine-queryable handoff state:

- `loop:ready` — the Issue contract is ready for worker implementation;
- `loop:in-progress` — a worker has claimed the Issue;
- `loop:blocked` — progress cannot safely continue without resolution.

The normal transition is:

```text
fresh repository state
   ↓ Codex Planner
open Issue + loop:ready
   ↓ Codex Worker claim
loop:in-progress
   ↓ implementation + independent review
Pull Request
   ↓ next Codex Planner cycle
merge / request changes / block / escalate
   ↓
select next increment only when WIP is clear
```

An open Issue without `loop:ready` is **not** permission for scheduled implementation.

If the required labels are missing, scheduled automation must fail closed rather than inventing another dispatch mechanism.

## Work-in-progress policy

Default to **WIP = 1** across dispatched autonomous implementation.

If a `loop:ready` or `loop:in-progress` Issue exists, or an autonomous PR is still awaiting resolution, the planner should normally finish that work before dispatching another implementation Issue.

Existing user-reported Issues may remain open as backlog/discussion items without counting as dispatched WIP unless they carry loop state.

Increasing schedule frequency does **not** mean creating more work. It shortens idle latency between state transitions.

## Planner cycle

Every scheduled planner run must use fresh GitHub state and process work in this order:

1. read `docs/DEVELOPMENT_DIRECTION.md`, this document, `AGENTS.md`, and referenced policy;
2. inspect `main` health and required CI;
3. inspect open PRs before generating new work;
4. if one autonomous PR is clearly merge-ready, merge it when authorized by this policy;
5. if changes are needed, keep the correction bounded to the existing work rather than generating a replacement task;
6. inspect blocked work and relevant open Issues;
7. check `loop:ready` / `loop:in-progress` state and unresolved autonomous PRs;
8. only when WIP is clear, compare a small set of plausible next directions internally;
9. create **one** implementation-ready Issue for the best bounded increment and apply `loop:ready`;
10. report mutations or escalation concisely.

A run with no mutation is valid. Never manufacture an Issue merely because the timer fired.

## Choosing the next Issue

Use `docs/DEVELOPMENT_DIRECTION.md`, repository evidence, and marginal value rather than blindly consuming README scope, scikit-rf surface area, or the conformance matrix.

Prefer work in roughly this order:

1. broken main / correctness regressions;
2. concrete blocking conformance or numerical risk;
3. foundational capability with substantial downstream leverage;
4. high-value bounded RF capability;
5. non-blocking characterization/conformance when justified;
6. ergonomics, optimization, cleanup, refactor, or documentation when they unlock or protect higher-value work.

A good autonomous Issue is:

- aligned with the current development horizon;
- small enough for independent review and merge;
- useful on its own or clearly unlocking subsequent work;
- objectively verifiable;
- product-complete enough that the worker need not invent externally visible behavior;
- justified by current repository evidence;
- free of unnecessary implementation prescriptions.

Before dispatch, consider several plausible directions internally and choose one. Do not publish a speculative long roadmap.

If all plausible next increments have low marginal value, do nothing and report that conclusion.

## Avoiding false progress

`docs/CONFORMANCE.md` is a test-design matrix, not an autonomous roadmap checklist.

A missing coverage dimension alone is insufficient reason for a standalone Issue. A conformance-only Issue must identify the materially different defect class it can catch and why that risk should be retired now.

Prefer proportionate conformance coverage inside capability work when independently reviewable. Reuse existing fixtures and lower-level proofs rather than creating duplicate evidence.

After two consecutive merged conformance-only increments, if `main` is healthy and no concrete blocker or newly exposed risk exists, the next autonomous Issue should normally advance capability.

Likewise, repeated cleanup/refactor/documentation-only increments require concrete justification that they unlock, simplify, or protect meaningful RF development. Repository motion is not itself progress.

## Definition of merge-ready for numerical work

For substantive numerical features, review against `docs/CONFORMANCE.md` and `AGENTS.md`.

Green CI is necessary but not sufficient. Where applicable, verify proportionate evidence for:

- documented mathematical behavior;
- deterministic Rust unit tests;
- RF invariant/property tests;
- differential comparison against the pinned scikit-rf oracle;
- justified tolerances;
- N-port behavior;
- scalar/per-port/frequency-dependent and complex reference impedances where supported;
- near-singular or ill-conditioned behavior when relevant;
- provenance/licensing requirements.

The planner should also ask whether the PR actually fulfills the Issue's intended capability or correctness outcome without unnecessary complexity.

## Planner merge authority

The planner may merge an autonomous PR without human confirmation when all of the following are true:

- the PR implements an Issue created/dispatched under this loop;
- required CI and deterministic verification are green;
- independent review has no unresolved substantive finding;
- the diff is bounded to the Issue contract;
- no escalation condition below is triggered;
- the change does not publish/release externally or freeze a material public-policy decision.

Otherwise the planner must request bounded changes, mark blocked, or escalate. It must not merge merely to keep the loop moving.

## Escalation boundary

Stop generation or merge and ask the human owner when a material decision involves:

- breaking or freezing a public API;
- choosing between plausible RF definitions or wave conventions without repository policy selecting one;
- disagreement between standards/mathematics/scikit-rf that changes externally visible semantics;
- unavailable paid specifications, papers, datasets, or fixtures needed for correctness;
- uncertain copyright, license, attribution, or provenance obligations;
- major crate-boundary or repository-wide architecture changes;
- release, crates.io publishing, signing, or other irreversible external publication;
- security or safety policy outside a bounded defect fix;
- deliberate relaxation of WIP, review, or autonomous merge policy.

The escalation must state the decision needed and the smallest useful set of options. Do not create implementation work that assumes an unresolved answer.

## Recommended cadence

Run planner and worker as separate systemd timers. A four-cycle daily cadence is reasonable once the loop is stable:

```text
03:00 JST  Codex Planner
03:30 JST  Codex Worker
09:00 JST  Codex Planner
09:30 JST  Codex Worker
15:00 JST  Codex Planner
15:30 JST  Codex Worker
21:00 JST  Codex Planner
21:30 JST  Codex Worker
```

The exact times may change. The invariants are more important than the clock:

- planner evaluates fresh state before new dispatch;
- explicit Issue dispatch precedes implementation;
- WIP remains bounded;
- no-op runs are normal;
- higher cadence reduces waiting, not quality bars.
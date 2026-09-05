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
- request bounded changes and re-dispatch the same linked Issue for a correction pass, or mark work blocked when appropriate;
- escalate decisions that require human RF/product/policy judgment.

The planner decides **what should happen next**, but does not pre-prescribe routine implementation details. The worker owns repository-specific implementation planning after claiming the Issue.

### Codex Worker

The scheduled worker owns execution of one dispatched Issue:

- find exactly one `loop:ready` Issue authorizing either initial implementation or a correction pass;
- claim it before implementation;
- update its existing autonomous PR rather than opening a duplicate when the dispatch is for bounded corrections;
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

- `loop:ready` — the Issue is authorized for the next Worker pass, either initial implementation or bounded correction of its existing autonomous PR;
- `loop:in-progress` — a Worker has claimed the currently authorized implementation or correction pass;
- `loop:blocked` — progress cannot safely continue without resolution.

These state labels are mutually exclusive on an open Issue. An Issue carrying more than one of them is ambiguous and must not authorize implementation.

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
   ├─ merge ──────────────────────────────→ WIP clears
   ├─ bounded change request
   │      ↓ re-dispatch the same Issue as loop:ready
   │   Worker correction pass
   │      ↓ update the same Pull Request
   │   next Codex Planner cycle
   └─ block / escalate
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
5. if implementation-ready changes are needed, keep the correction bounded to the existing work and re-dispatch its linked Issue under the correction-loop rules below rather than generating a replacement task;
6. inspect blocked work and relevant open Issues, including interrupted pre-PR claims under the recovery rules below;
7. check `loop:ready` / `loop:in-progress` state and unresolved autonomous PRs;
8. only when WIP is clear, compare a small set of plausible next directions internally;
9. create **one** implementation-ready Issue for the best bounded increment and apply `loop:ready`;
10. report mutations or escalation concisely.

A run with no mutation is valid. Never manufacture an Issue merely because the timer fired.

## Correction loop for an existing pull request

A bounded change request on an autonomous PR is unfinished work within the same WIP slot. It must be able to return to the Worker without creating another Issue or PR.

The Planner may re-dispatch a correction pass only when all of the following are true:

- exactly one open loop Issue is unambiguously linked to exactly one open autonomous PR through GitHub's closing-reference relationship;
- no closed/unmerged PR is associated with that Issue;
- that Issue carries only `loop:in-progress`, or already carries only `loop:ready` because this same re-dispatch completed earlier;
- the Issue and PR are the existing WIP, not unrelated backlog or a new increment;
- the required changes are bounded by the existing Issue contract and do not require a human escalation decision;
- the Planner has independently accepted a concrete unresolved change request for the head commit it inspected;
- the PR targets `main`, and its head is a branch in this repository that the Worker can update without force-pushing.

When those conditions hold, the Planner must:

1. create a concise change request if the current head does not already have an adequate one;
2. reuse an adequate unresolved request for an unchanged head instead of posting a duplicate comment or review;
3. replace `loop:in-progress` with `loop:ready` on the same Issue, preserving WIP=1;
4. verify that the post-transition Issue has exactly `loop:ready`, not `loop:in-progress` or `loop:blocked`.

If the Issue is already in that verified ready state, the Planner leaves it unchanged. A partially applied or ambiguous label transition must fail closed and be reported; it is not permission to create replacement work. Multiple linked PRs, a fork or otherwise unwritable head, an unclear change request, or product/RF/policy ambiguity must be blocked or escalated instead of re-dispatched.

After selecting exactly one `loop:ready` Issue, the Worker must verify that it carries no other loop state and that no other `loop:ready` or `loop:in-progress` Issue or unresolved autonomous PR violates WIP=1. It must then determine the pass type from fresh GitHub state before editing:

- with no current or historical PR associated with the Issue, perform the initial implementation and create one PR;
- with exactly one closing-linked open autonomous PR, perform a correction pass against that PR;
- with an ambiguous Issue-to-PR relationship, more than one linked open PR, or any closed/unmerged historical PR whether or not an open PR also exists, fail closed without choosing a target.

For a correction pass, the Worker must verify the current-head change request, claim the Issue by replacing `loop:ready` with `loop:in-progress`, and re-read the Issue, PR, head SHA, change request, base, and branch ownership before editing. It then fetches and checks out the exact existing head and applies only the requested bounded correction through the normal IssueFlow implementation, verification, and independent-review workflow. Main commits and pushes normally to the same branch so the same PR is updated. The Worker must not force-push, open a replacement PR, or broaden the Issue contract. If fresh revalidation differs from the claimed state or the remote head changes after inspection, the Worker must stop rather than overwrite concurrent work and let a later Planner cycle decide the next safe transition.

After the PR is updated, the Issue remains `loop:in-progress` until the next Planner cycle merges, re-dispatches another bounded correction, blocks, or escalates it.

## Recovery after interruption before a pull request

A Worker can stop after claiming an Issue but before publishing a PR. `loop:in-progress` is a dispatch label, not proof that a process is still running. A successful no-op service exit also does not prove the claimed implementation finished. Resolve this existing WIP before creating new work.

Automatic recovery is limited to the configured **single execution host**. All authorized Planner/Worker runs, including manual runs, must use its same lock-holding wrapper. If another host or an independently launched Worker may still be running, automatic recovery is unavailable until that executor is accounted for. Do not infer termination from claim age, missing comments, a missing PR, a GitHub search result, or a PID alone.

The Planner may re-dispatch the same Issue for an interrupted initial pass only after verifying all of these conditions:

1. The host execution context described in `docs/CODEX_AUTOMATION.md` identifies this run as a Planner holding the shared lock. The single-host execution contract is in effect, so no authorized Worker can still be running. Identify the prior claim/run from its Issue comment and host evidence. If the Worker stopped between the label transition and claim comment, a uniquely matching host Worker run interval and the Issue's direct label timeline may supply that linkage; ambiguous or clock-discontinuous intervals are insufficient. For a claim predating host run records, an explicit maintainer audit may supply the missing historical linkage.
2. Fresh direct GitHub API state shows exactly one dispatched Issue, carrying only `loop:in-progress`, with no other ready/in-progress work and no unresolved autonomous PR. Inspect its body, comments, full relationship history, and current remote heads. There must be no current or historical PR associated with it. Existing PR work must follow the correction-loop rules instead; closed/unmerged or ambiguous relationships must not be treated as a fresh start.
3. The Issue remains implementation-ready under current `main` and its original contract. There is no unresolved RF/product/API/provenance decision. A remote work branch without a PR, concurrent updates, unexplained local work, or uncertain ownership requires escalation rather than overwriting or creating replacement work.
4. Inspect the host's preserved checkout/run evidence when present. Record where uncommitted or unpushed work is preserved and whether it belongs to this claim. Saved work is unreviewed input, not a completed implementation. If a prior host discarded the checkout, state that explicitly and authorize rebuilding the same contract; do not pretend the work was recovered.
5. This Issue has not already used an automatic pre-PR recovery. Permit **one automatic recovery per Issue**. A further interrupted claim must be marked `loop:blocked` with the failure evidence and required intervention, rather than retried indefinitely. Ordinary corrections of an existing PR do not count as pre-PR recovery.

When these conditions hold, the Planner must post one concise recovery comment on the same Issue, marked `<!-- rfkit-pre-pr-recovery:v1 -->`, identifying the interrupted claim/run, current Planner run, termination/exclusivity evidence, preserved-work disposition, and unchanged implementation contract. Re-read the Issue labels, comments, relationships, remote branches, and WIP immediately before replacing `loop:in-progress` with `loop:ready`, preserving all non-loop labels. Verify the result through the Issue's direct API. Do not create a replacement Issue or PR, implement code, or dispatch additional work in this cycle.

If a run stops between the recovery comment and label update, a later Planner may finish that same transition only when there has been no intervening ready/claim transition or other material change. Reuse the comment rather than consuming a second attempt. If the Issue is already only `loop:ready`, leave it ready. If a Worker has claimed it again, the recovery has been consumed; do not mistake the new claim for an unfinished label update. Use the Issue timeline and run IDs to distinguish these cases.

When termination, provenance, ownership, or relationships cannot be established, leave potentially live work untouched and record a concise blocker/required evidence on the Issue. If termination is established but safe restart is not, replace its loop state with `loop:blocked`. Reuse an adequate comment for unchanged evidence instead of repeatedly posting. A journal-only no-op is insufficient when an orphaned claim requires intervention.

After recovery dispatch, the Worker uses the normal initial implementation workflow on the same Issue, with the normal claim, verification, fresh independent review, and one PR. Before product edits, it records a claim comment containing the host run ID (when supplied), base SHA and intended branch, and verifies the claim again. It may inspect clearly attributed preserved work and reapply useful changes into its fresh checkout through the usual implementation workflow; it must not alter the archive, treat an old review as current, or push over an unexplained remote branch. Missing or ambiguous recovery evidence must be surfaced, not guessed. The Worker never changes an unready Issue to ready itself.

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

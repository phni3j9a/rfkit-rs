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
- classify it as Green or Yellow and define what / why / done in one implementation-ready Issue;
- apply `loop:ready` only when the Issue is genuinely ready for implementation;
- merge a clearly merge-ready autonomous PR when policy permits;
- request bounded changes and re-dispatch the same linked Issue for a correction pass, or mark work blocked when appropriate;
- resolve Green and Yellow decisions within the repository envelope and escalate only Red decisions.

The planner decides **what should happen next**, but does not pre-prescribe routine implementation details. The worker owns repository-specific implementation planning after claiming the Issue.

The Planner remains repository-policy-driven and does not depend on Axiom skill availability or implementation routing.

### Codex Worker

The scheduled worker owns execution of one dispatched Issue:

- find exactly one `loop:ready` Issue authorizing either initial implementation or a correction pass;
- explicitly invoke `$axiom:axiom` from the canonical Worker prompt and verify that its full instructions loaded before claiming an Issue; do not infer unavailability merely because Codex abbreviates its initial skills list, and do not use a hardcoded plugin cache/version path as a fallback;
- if no `loop:ready` Issue exists, complete a normal no-op without checking delegation routing, delegating, or mutating GitHub; loading the explicitly invoked Axiom skill before this check is acceptable;
- before claiming it, verify that the directly exposed `collaboration.spawn_agent` tool definition includes explicit `model` and `reasoning_effort` parameters needed to request `gpt-5.6-luna` at MAX for bounded implementation and a fresh `gpt-5.6-sol` at XHIGH for independent review. If the skill or routing is unavailable, fail closed and report the blocker rather than reinstalling IssueFlow, creating a replacement orchestrator, or substituting models;
- claim it before implementation;
- update its existing autonomous PR rather than opening a duplicate when the dispatch is for bounded corrections;
- read `AGENTS.md`, repository policy, and the Issue contract;
- inspect the repository and create the implementation plan;
- use Axiom as execution guidance to delegate bounded product-code implementation to the named Luna MAX route and independent review to a fresh named Sol XHIGH route, while Main retains integration and acceptance;
- verify actual delegated turns from their requested spawn arguments, child `turn_context`, and `task_complete` evidence; a throwaway routing canary is not required;
- allow independent bounded subtasks to run in parallel within the selected Issue when useful, without increasing WIP or authorizing another Issue;
- have Main run deterministic verification, adjudicate review findings, perform Git operations, and create or update the PR;
- within an active Worker pass/review cycle, retain the same Sol XHIGH reviewer session for re-review while the review boundary remains stable; each scheduled correction pass still obtains a fresh independent review.

### ChatGPT governor / auditor

ChatGPT is outside the normal scheduled loop. It does not need to generate routine Issues.

When asked by the human owner, ChatGPT should inspect fresh GitHub state and recent development history to determine whether the autonomous system is making **meaningful RF progress**. The audit should look for capability growth, correctness improvement, downstream leverage, repeated low-value work, conformance tunnels, cleanup/refactor loops, over-engineering, weak Issue selection, and policy drift.

If the loop is optimizing the wrong thing, ChatGPT should recommend or apply bounded policy changes rather than manually micromanaging the next few Issues.

### Human owner

The human owner sets the development horizon, risk tolerance, and Red boundaries, and audits batches of completed work. Routine provisional API design is governed by the envelope below rather than per-method pre-approval.

Human attention is required for Red decisions such as stabilization or a compatibility promise, an authoritative RF conflict that explicit APIs cannot preserve, unavailable evidence required for correctness, unresolved provenance/licensing obligations, major difficult-to-reverse architecture changes, release/publishing actions, security-policy expansion, or a deliberate change to autonomous WIP/review/merge authority.

## Autonomous decision classes

Every newly dispatched Issue must state `Autonomy class: Green` or `Autonomy class: Yellow`. A Red Issue must not receive `loop:ready` until the human owner records the specific decision that makes a bounded Green or Yellow implementation possible.

### Green

Green work is routine within established repository policy:

- the behavior follows a single well-supported mathematical/specification interpretation and existing RF semantics;
- consequential wave, unit, grid, tolerance, and error behavior is explicit;
- the change is additive, corrective, or internal and does not make a new compatibility promise;
- it follows the current canonical model and architecture;
- verification and provenance requirements are clear;
- the action is reversible and does not publish, release, sign, or use new credentials.

The Planner may define, dispatch, and later merge Green work without human confirmation.

### Yellow

Yellow work contains a bounded design choice, but may still proceed autonomously when:

- the choice remains reversible during the provisional `0.x` phase;
- authoritative sources do not leave an externally visible RF conflict that the proposed explicit API cannot represent;
- the Issue records plausible alternatives, selection criteria, compatibility impact, and rollback or migration;
- the selected behavior is explicit rather than hidden behind a vague default;
- the independent review explicitly evaluates the decision, evidence, scope, and reversibility;
- no Red condition applies.

Examples include a justified supporting public type, an explicitly named additional convention, a bounded dependency or crate-boundary adjustment, a provisional API revision with migration notes, and consolidating a recorded public-surface overlap under `docs/PUBLIC_API.md`. The Planner may define, dispatch, and later merge Yellow work without human confirmation when every Yellow record and merge gate is satisfied.

### Red

Red work requires a human decision before dispatch or merge:

- declaring stability, making a compatibility promise, or breaking a promise already made;
- release, crates.io publication, signing, credential changes, or another irreversible external action;
- externally visible RF behavior where authoritative sources materially disagree and explicit side-by-side semantics cannot preserve the alternatives;
- unavailable paid or restricted evidence that is necessary to establish correctness;
- unresolved copyright, license, attribution, or provenance obligations;
- a major, difficult-to-reverse replacement of the canonical data model, storage representation, crate layout, or repository architecture;
- security or safety policy beyond a bounded defect fix;
- relaxation of WIP=1, independent review, autonomous merge gates, or these decision classes.

Uncertainty alone does not make work Red. The Planner should first narrow the decision, seek authoritative evidence, prefer explicit semantics over an implicit default, and determine whether a reversible Yellow choice is available. It must not relabel a genuinely Red decision merely to keep the loop active.

The classification and its evidence belong in the Issue and PR, not in additional GitHub labels. This keeps `loop:*` labels reserved for dispatch state.

## Decision records and post-merge audit

A Green Issue records the classification, explicit semantics, authoritative basis, verification plan, and why the work is reversible. A Yellow Issue additionally records the plausible alternatives, the selected option and criteria, compatibility impact, and rollback or migration path. The PR must preserve or update that record to describe the implementation that was actually reviewed.

The human owner or ChatGPT governor may audit a batch of merged decisions against these records and open or request ordinary corrective work when classifications, evidence, or outcomes are weak. That audit is a feedback mechanism for future work, not a standing pre-merge approval gate. It becomes a gate only when the owner explicitly intervenes or the work reaches a Red boundary.

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

A `loop:blocked` Issue without an unresolved autonomous PR does not consume the WIP slot. After recording the blocker, the Planner may dispatch an independent Green or Yellow increment that will not conflict with preserved or potentially live work. One blocked decision must not become a repository-wide stop.

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
8. only when WIP is clear, run the public-surface audit if it is due (see "Public surface growth and consolidation"), then compare a small set of plausible next directions internally;
9. classify the best bounded increment as Green or Yellow, create **one** implementation-ready Issue containing the required decision record, and apply `loop:ready`;
10. report mutations or escalation concisely.

A run with no mutation is valid. Never manufacture an Issue merely because the timer fired.

## Correction loop for an existing pull request

A bounded change request on an autonomous PR is unfinished work within the same WIP slot. It must be able to return to the Worker without creating another Issue or PR.

The Planner may re-dispatch a correction pass only when all of the following are true:

- exactly one open loop Issue is unambiguously linked to exactly one open autonomous PR through GitHub's closing-reference relationship;
- no closed/unmerged PR is associated with that Issue;
- that Issue carries only `loop:in-progress`, or already carries only `loop:ready` because this same re-dispatch completed earlier;
- the Issue and PR are the existing WIP, not unrelated backlog or a new increment;
- the required changes are bounded by the existing Issue contract and do not cross a Red boundary;
- the Planner has independently accepted a concrete unresolved change request for the head commit it inspected;
- the PR targets `main`, and its head is a branch in this repository that the Worker can update without force-pushing.

When those conditions hold, the Planner must:

1. create a concise change request if the current head does not already have an adequate one;
2. reuse an adequate unresolved request for an unchanged head instead of posting a duplicate comment or review;
3. replace `loop:in-progress` with `loop:ready` on the same Issue, preserving WIP=1;
4. verify that the post-transition Issue has exactly `loop:ready`, not `loop:in-progress` or `loop:blocked`.

If the Issue is already in that verified ready state, the Planner leaves it unchanged. A partially applied or ambiguous label transition must fail closed and be reported; it is not permission to create replacement work. Multiple linked PRs, a fork or otherwise unwritable head, an unclear change request, an unclear autonomy classification, or a Red decision must be blocked or escalated instead of re-dispatched.

After selecting exactly one `loop:ready` Issue, the Worker must verify that it carries no other loop state and that no other `loop:ready` or `loop:in-progress` Issue or unresolved autonomous PR violates WIP=1. It must then determine the pass type from fresh GitHub state before editing:

- with no current or historical PR associated with the Issue, perform the initial implementation and create one PR;
- with exactly one closing-linked open autonomous PR, perform a correction pass against that PR;
- with an ambiguous Issue-to-PR relationship, more than one linked open PR, or any closed/unmerged historical PR whether or not an open PR also exists, fail closed without choosing a target.

For a correction pass, the Worker must verify the current-head change request, claim the Issue by replacing `loop:ready` with `loop:in-progress`, and re-read the Issue, PR, head SHA, change request, base, and branch ownership before editing. It then fetches and checks out the exact existing head and applies only the requested bounded correction through the Axiom-guided implementation and independent-review execution described above. Main runs deterministic verification, adjudicates review findings, and commits and pushes normally to the same branch so the same PR is updated. Within this active Worker pass/review cycle, the same Sol XHIGH reviewer session is reused for re-review while the review boundary remains stable; each scheduled correction pass obtains a fresh independent review. The Worker must not force-push, open a replacement PR, or broaden the Issue contract. If fresh revalidation differs from the claimed state or the remote head changes after inspection, the Worker must stop rather than overwrite concurrent work and let a later Planner cycle decide the next safe transition.

After the PR is updated, the Issue remains `loop:in-progress` until the next Planner cycle merges, re-dispatches another bounded correction, blocks, or escalates it.

## Recovery after interruption before a pull request

A Worker can stop after claiming an Issue but before publishing a PR. `loop:in-progress` is a dispatch label, not proof that a process is still running. A successful no-op service exit also does not prove the claimed implementation finished. Resolve this existing WIP before creating new work.

Automatic recovery is limited to the configured **single execution host**. All authorized Planner/Worker runs, including manual runs, must use its same lock-holding wrapper. If another host or an independently launched Worker may still be running, automatic recovery is unavailable until that executor is accounted for. Do not infer termination from claim age, missing comments, a missing PR, a GitHub search result, or a PID alone.

The Planner may re-dispatch the same Issue for an interrupted initial pass only after verifying all of these conditions:

1. The host execution context described in `docs/CODEX_AUTOMATION.md` identifies this run as a Planner holding the shared lock. The single-host execution contract is in effect, so no authorized Worker can still be running. Identify the prior claim/run from its Issue comment and host evidence. If the Worker stopped between the label transition and claim comment, a uniquely matching host Worker run interval and the Issue's direct label timeline may supply that linkage; ambiguous or clock-discontinuous intervals are insufficient. For a claim predating host run records, an explicit maintainer audit may supply the missing historical linkage.
2. Fresh direct GitHub API state shows exactly one dispatched Issue, carrying only `loop:in-progress`, with no other ready/in-progress work and no unresolved autonomous PR. Inspect its body, comments, full relationship history, and current remote heads. There must be no current or historical PR associated with it. Existing PR work must follow the correction-loop rules instead; closed/unmerged or ambiguous relationships must not be treated as a fresh start.
3. The Issue remains implementation-ready under current `main` and its original Green or Yellow contract. There is no unresolved Red decision. A remote work branch without a PR, concurrent updates, unexplained local work, or uncertain ownership requires escalation rather than overwriting or creating replacement work.
4. Inspect the host's preserved checkout/run evidence when present. Record where uncommitted or unpushed work is preserved and whether it belongs to this claim. Saved work is unreviewed input, not a completed implementation. If a prior host discarded the checkout, state that explicitly and authorize rebuilding the same contract; do not pretend the work was recovered.
5. This Issue has not already used an automatic pre-PR recovery. Permit **one automatic recovery per Issue**. A further interrupted claim must be marked `loop:blocked` with the failure evidence and required intervention, rather than retried indefinitely. Ordinary corrections of an existing PR do not count as pre-PR recovery.

When these conditions hold, the Planner must post one concise recovery comment on the same Issue, marked `<!-- rfkit-pre-pr-recovery:v1 -->`, identifying the interrupted claim/run, current Planner run, termination/exclusivity evidence, preserved-work disposition, and unchanged implementation contract. Re-read the Issue labels, comments, relationships, remote branches, and WIP immediately before replacing `loop:in-progress` with `loop:ready`, preserving all non-loop labels. Verify the result through the Issue's direct API. Do not create a replacement Issue or PR, implement code, or dispatch additional work in this cycle.

If a run stops between the recovery comment and label update, a later Planner may finish that same transition only when there has been no intervening ready/claim transition or other material change. Reuse the comment rather than consuming a second attempt. If the Issue is already only `loop:ready`, leave it ready. If a Worker has claimed it again, the recovery has been consumed; do not mistake the new claim for an unfinished label update. Use the Issue timeline and run IDs to distinguish these cases.

When termination, provenance, ownership, or relationships cannot be established, leave potentially live work untouched and record a concise blocker/required evidence on the Issue. If termination is established but safe restart is not, replace its loop state with `loop:blocked`. Reuse an adequate comment for unchanged evidence instead of repeatedly posting. A journal-only no-op is insufficient when an orphaned claim requires intervention.

After recovery dispatch, the Worker uses the normal Axiom-guided initial implementation execution on the same Issue, with the normal claim, verification, fresh independent review, and one PR. Before product edits, it records a claim comment containing the host run ID (when supplied), base SHA and intended branch, and verifies the claim again. It may inspect clearly attributed preserved work and reapply useful changes into its fresh checkout through the usual bounded delegation; it must not alter the archive, treat an old review as current, or push over an unexplained remote branch. Missing or ambiguous recovery evidence must be surfaced, not guessed. The Worker never changes an unready Issue to ready itself.

## Choosing the next Issue

Use `docs/DEVELOPMENT_DIRECTION.md`, repository evidence, and marginal value rather than blindly consuming README scope, scikit-rf surface area, or the conformance matrix.

Prefer work in roughly this order:

1. broken main / correctness regressions;
2. concrete blocking conformance or numerical risk;
3. foundational capability with substantial downstream leverage;
4. high-value bounded RF capability, or consolidation of a public-surface overlap recorded in `docs/PUBLIC_API.md` that reduces how many entry points callers must choose among for one RF operation;
5. non-blocking characterization/conformance when justified;
6. ergonomics, optimization, cleanup, refactor, or documentation when they unlock or protect higher-value work.

A good autonomous Issue is:

- aligned with the current development horizon;
- small enough for independent review and merge;
- useful on its own or clearly unlocking subsequent work;
- objectively verifiable;
- explicit enough that the worker can resolve exposed choices within its recorded Green or Yellow authority without broadening the contract;
- justified by current repository evidence;
- free of unnecessary implementation prescriptions.

Before dispatch, consider several plausible directions internally and choose one. Do not publish a speculative long roadmap.

If all plausible next increments have low marginal value, do nothing and report that conclusion.

If one plausible direction is Red or blocked, consider independent Green and Yellow directions before concluding that useful work is unavailable. Do not evade the blocked decision by implementing a hidden prerequisite whose primary purpose is to prejudge it.

## Avoiding false progress

`docs/CONFORMANCE.md` is a test-design matrix, not an autonomous roadmap checklist.

A missing coverage dimension alone is insufficient reason for a standalone Issue. A conformance-only Issue must identify the materially different defect class it can catch and why that risk should be retired now.

Prefer proportionate conformance coverage inside capability work when independently reviewable. Reuse existing fixtures and lower-level proofs rather than creating duplicate evidence.

After two consecutive merged conformance-only increments, if `main` is healthy and no concrete blocker or newly exposed risk exists, the next autonomous Issue should normally advance capability.

Likewise, repeated cleanup/refactor/documentation-only increments require concrete justification that they unlock, simplify, or protect meaningful RF development. Repository motion is not itself progress.

Consolidating an entry in the public-surface overlap inventory is not cleanup-only work. It changes what callers must understand to perform an RF operation, and it is ranked with bounded RF capability. Internal refactors that leave the public surface unchanged remain under the cleanup rule.

## Public surface growth and consolidation

Adding a named public variant must not be cheaper by policy than generalizing an existing operation. Otherwise the provisional surface grows until a human intervenes.

### Overlap rule for new work

When a proposed public operation would compute the same RF quantity as an existing public operation on a common domain, the Issue must choose one of these and record the choice:

1. generalize or replace the existing operation under the consolidation rules in `docs/PUBLIC_API.md` (normally Yellow); or
2. add the variant and add an overlap-inventory entry stating why callers, not the implementation, need both entry points and what condition will retire the overlap.

Explicit coexistence is not a default tie-breaker. Choose it only when the recorded caller-facing reason holds. The independent review must check that the overlap choice is recorded and justified.

### Planner public-surface audit

The Planner owns a periodic audit of the public surface so that correcting drift does not depend on an ad hoc human request.

- The audit log is the long-lived GitHub Issue titled `Public surface audit log` (#107). It never carries `loop:*` labels and does not consume WIP.
- An autonomous product PR is a merged PR for a loop-dispatched Issue that changes code under `crates/`.
- An audit is due when at least 10 autonomous product PRs have merged after the last PR covered by the latest audit comment, or when no audit comment exists.
- When an audit is due and WIP is clear, the Planner performs it in the same run, before comparing next directions. It compares the current public entry points of each crate with the overlap inventory in `docs/PUBLIC_API.md`, looks for unrecorded overlaps and strategy-only qualifiers, and appends one audit comment in the format described on the log Issue.
- Findings feed ordinary selection: an unrecorded overlap, or an inventory entry whose consolidation now has the highest marginal value, may become the next dispatched Issue. The audit does not by itself authorize more than one dispatch or relax WIP.
- When WIP is not clear, the audit waits until the next run in which WIP is clear.

The ChatGPT governor/auditor and the human owner may still audit at any time. The Planner audit is the default feedback path, not a replacement for them.

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
- the Issue and PR carry the required Green or Yellow classification and, for Yellow, the alternatives, compatibility impact, rollback/migration, and decision-focused review evidence;
- no Red condition in this policy is triggered;
- the change does not publish or release externally.

Otherwise the planner must request bounded changes, mark blocked, or escalate. It must not merge merely to keep the loop moving.

## Escalation boundary

Stop dispatch or merge and ask the human owner only when the decision is Red under this document or a safe Green/Yellow classification cannot be established after narrowing it.

The escalation must state the decision needed, authoritative evidence or uncertainty, why explicit or reversible alternatives are insufficient, and the smallest useful set of options. Mark only the affected Issue `loop:blocked` when safe to do so. Do not create implementation work that assumes an unresolved Red answer, but continue considering independent eligible work under WIP=1.

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

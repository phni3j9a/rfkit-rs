# Codex Automation

This document defines the recommended scheduled Codex worker for the `rfkit-rs` loop-engineering workflow.

It is intentionally thin: GitHub Issue dispatch decides **what** work is authorized, `AGENTS.md` defines repository engineering rules, and the IssueFlow `issue-to-pr` skill owns the implementation workflow.

## Prerequisites

Before enabling the worker:

1. the repository checkout used by Codex can fetch/push and create pull requests;
2. Codex can read `AGENTS.md` and `docs/LOOP_ENGINEERING.md`;
3. the IssueFlow `issue-to-pr` skill is installed and its required Luna MAX / Sol XHIGH child-model routing is available;
4. these GitHub labels exist:
   - `loop:ready`
   - `loop:in-progress`
   - `loop:blocked`

One-time label setup with GitHub CLI can be done from an authenticated checkout:

```bash
gh label create 'loop:ready' --description 'Ready for scheduled Codex implementation' --color '0E8A16'
gh label create 'loop:in-progress' --description 'Claimed by the scheduled Codex worker' --color '1D76DB'
gh label create 'loop:blocked' --description 'Loop work is blocked and requires attention' --color 'D73A4A'
```

If a label already exists, do not recreate or rename it during a normal worker run.

## Recommended schedule

Start with one worker run per day:

```text
10:00 JST daily
```

The ChatGPT maintainer cycle is expected to run earlier, for example at 09:00 JST, so it can review pull requests and prepare at most one `loop:ready` Issue before the worker checks the queue.

## Automation instruction

Use the following instruction as the Codex Automation task body:

```text
Operate as the scheduled implementation worker for phni3j9a/rfkit-rs.

Read AGENTS.md and docs/LOOP_ENGINEERING.md before taking action. Use fresh GitHub state.

Query open GitHub Issues carrying label loop:ready.

- If none exist, report that there is no dispatched work and exit without implementing any other open Issue.
- If more than one exists, treat this as a WIP=1 invariant violation. Report the conflicting Issue numbers and exit without choosing silently.
- If exactly one exists, claim it before implementation by removing loop:ready and adding loop:in-progress. If the claim write fails, do not implement.

After a successful claim, implement that Issue end-to-end using the installed IssueFlow issue-to-pr skill.

The GitHub Issue is the product contract. Codex Main owns repository inspection, implementation planning, task decomposition, integration, deterministic verification, review-finding adjudication, Git, and PR creation. Delegate bounded product-code implementation to Luna MAX and independently review the integrated candidate with a fresh Sol XHIGH reviewer as defined by the installed skill. Do not silently substitute unspecified child models if required routing is unavailable.

Follow all rfkit-rs numerical, conformance, provenance, and architecture requirements in AGENTS.md. Do not guess a material RF/product/licensing decision that the Issue and repository policy leave unresolved. If such a blocker appears, add loop:blocked, explain the blocker on the Issue, and stop without opening a misleading completion PR.

When implementation and review succeed, open a PR that links/closes the claimed Issue and includes concise verification evidence. Leave loop:in-progress in place while the PR is awaiting maintainer review; the next ChatGPT maintainer cycle owns merge/readiness judgment. Do not claim a second Issue in the same run.
```

## Failure behavior

The worker must fail closed when:

- GitHub dispatch labels are missing;
- it cannot atomically establish a clear claim state;
- the working tree cannot be safely isolated from unrelated user work;
- required Luna MAX / Sol XHIGH routing is unavailable;
- the Issue requires unresolved product or RF semantics;
- required standards/papers/fixtures are unavailable;
- provenance or licensing obligations are unclear.

A failed or no-work run should not mutate unrelated Issues or generate speculative backlog.

## Review ownership

The Codex worker produces a reviewed PR, but it does not replace the next maintainer cycle.

The ChatGPT maintainer remains responsible for deciding whether the PR should merge under the repository's product intent and conformance policy. If changes are requested, the existing PR/Issue should be completed before new work is dispatched under WIP=1.

# Codex Automation

This document defines the recommended scheduled Codex Planner and Worker for the `rfkit-rs` autonomous engineering loop.

The timer/task instructions should stay thin. GitHub documents are the source of truth:

- `docs/DEVELOPMENT_DIRECTION.md` — what meaningful project progress means;
- `docs/LOOP_ENGINEERING.md` — autonomous roles, state machine, WIP, merge and escalation policy;
- `AGENTS.md` — repository engineering contract;
- the dispatched GitHub Issue — the implementation contract for a Worker run.

Do not copy long-lived policy into systemd units or shell prompts.

## Prerequisites

Before enabling automation:

1. the repository checkout used by Codex can fetch/push and use `gh` for Issues/PRs;
2. Planner and Worker can read the policy files above from fresh `main`;
3. the Worker has the configured IssueFlow `issue-to-pr` skill and required Luna MAX / Sol XHIGH routing;
4. these GitHub labels exist:
   - `loop:ready`
   - `loop:in-progress`
   - `loop:blocked`
5. each run starts from a safely isolated/clean checkout and fails closed on ambiguous local state.

## Recommended schedule

Once stable, run four planner/worker cycles per day:

```text
03:00 JST  planner
03:30 JST  worker
09:00 JST  planner
09:30 JST  worker
15:00 JST  planner
15:30 JST  worker
21:00 JST  planner
21:30 JST  worker
```

The 30-minute spacing is not a correctness requirement; it simply avoids overlapping the normal planner mutation window. Do not start a second run of the same role while the prior run is still active.

## Planner instruction

Use this as the complete scheduled `codex exec` instruction for the Planner:

```text
Operate as the autonomous Codex Planner for phni3j9a/rfkit-rs.

Use fresh GitHub and main state. Read docs/DEVELOPMENT_DIRECTION.md, docs/LOOP_ENGINEERING.md, AGENTS.md, and every repository document they require.

Execute exactly one Planner cycle as defined by docs/LOOP_ENGINEERING.md. Inspect main health, open PRs, CI, Issues, loop:* state, and recent relevant history before acting. Finish or resolve existing autonomous WIP before dispatching new work. When WIP is clear, choose at most one next bounded increment by marginal value and create one implementation-ready Issue with loop:ready only if worthwhile.

Apply the repository's merge and escalation policy exactly. A no-op run is valid. Do not manufacture work, publish a speculative roadmap, or rely on policy copied into this prompt when repository policy differs.
```

The Planner owns task selection and autonomous merge/readiness decisions. It must not implement the newly created Issue in the same run.

## Worker instruction

Use this as the complete scheduled `codex exec` instruction for the Worker:

```text
Operate as the autonomous Codex Worker for phni3j9a/rfkit-rs.

Use fresh GitHub and main state. Read AGENTS.md, docs/DEVELOPMENT_DIRECTION.md, docs/LOOP_ENGINEERING.md, and every repository document they require.

Query open GitHub Issues carrying loop:ready and execute exactly one Worker cycle under docs/LOOP_ENGINEERING.md.

If none exist, exit without implementing other work. If more than one exists, report the WIP invariant violation and exit. If exactly one exists, claim it by moving loop:ready to loop:in-progress before implementation; if the claim mutation fails, do not implement.

Treat the Issue as the product contract. Inspect the repository, own the implementation plan, and implement end-to-end through the installed IssueFlow issue-to-pr workflow. Follow all numerical, conformance, provenance, architecture, verification, and escalation requirements. Use the configured Luna MAX implementation routing and fresh Sol XHIGH independent review; fail closed rather than silently substituting required roles.

When implementation and review succeed, open a PR that closes/links the Issue and records concise verification evidence. Leave loop:in-progress while the PR awaits the next Planner cycle. Do not claim or implement a second Issue in this run.
```

## Planner failure behavior

The Planner must fail closed or no-op when:

- required policy cannot be read;
- GitHub state is ambiguous or unavailable;
- required loop labels are missing;
- WIP state is inconsistent;
- CI/review evidence needed for a merge decision is unavailable;
- the next useful step crosses an escalation boundary;
- all plausible new increments have low marginal value.

It must never create work merely to make a scheduled run look productive.

## Worker failure behavior

The Worker must fail closed when:

- GitHub dispatch labels are missing;
- it cannot establish an unambiguous claim;
- the working tree cannot be safely isolated from unrelated work;
- required model/skill routing is unavailable;
- the Issue requires unresolved product/RF/licensing policy;
- required authoritative sources or fixtures are unavailable;
- provenance/licensing obligations are unclear.

On a material blocker after claim, add `loop:blocked`, explain the blocker on the Issue, and stop rather than opening a misleading completion PR.

## systemd operational guidance

Keep service units operational rather than strategic. They should primarily:

1. lock against overlap;
2. enter/update the repository checkout safely;
3. invoke `codex exec` with the thin Planner or Worker instruction;
4. capture logs and a meaningful exit status.

Do not embed a roadmap, RF policy, acceptance criteria templates, or copied loop rules in systemd configuration. Updating GitHub policy should be sufficient to change future autonomous behavior.

## Observability

Retain per-run logs with role, start/end time, exit status, and Codex output so the human owner or ChatGPT auditor can distinguish:

- no-op due to WIP;
- successful merge/dispatch;
- successful Issue-to-PR implementation;
- invariant violation;
- blocked/escalated work;
- infrastructure/model/tool failure.

The GitHub repository remains the authoritative development state; logs are diagnostic evidence, not a second state machine.
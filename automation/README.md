# Scheduled Codex entrypoints

This directory contains the **version-controlled runtime prompts** for the autonomous `rfkit-rs` Planner and Worker.

The prompts are intentionally thin. Repository policy lives in `AGENTS.md`, `docs/DEVELOPMENT_DIRECTION.md`, `docs/LOOP_ENGINEERING.md`, and `docs/CODEX_AUTOMATION.md`. Systemd is only the scheduler/execution layer and must not duplicate or override that policy.

## Runtime prompts

- `planner-prompt.txt` — run exactly one autonomous Planner cycle.
- `worker-prompt.txt` — run exactly one autonomous Worker cycle.

A host integration should pass the appropriate file contents to `codex exec` without maintaining a separate copy of the prompt text.

Conceptually:

```sh
codex exec "$(cat automation/planner-prompt.txt)"
codex exec "$(cat automation/worker-prompt.txt)"
```

The concrete invocation may include the host's normal Codex flags, working-directory handling, logging, and non-interactive environment setup.

## Recommended cadence

Use four Planner/Worker opportunities per day in JST:

```text
03:00  Planner
03:30  Worker
09:00  Planner
09:30  Worker
15:00  Planner
15:30  Worker
21:00  Planner
21:30  Worker
```

The cadence reduces idle latency; it does **not** authorize four Issues or four implementations per day. WIP and no-op behavior remain defined by repository policy.

## Host integration requirements

The systemd integration should remain simple and operational rather than becoming a second orchestration system.

- Run Planner and Worker as separate services/timers.
- Use a shared host-level lock so Planner and Worker cannot mutate the same checkout concurrently.
- For interrupted-claim recovery, use the single-host execution context contract in `docs/CODEX_AUTOMATION.md`; all manual cycles must use the same wrapper and lock too.
- Run from the intended `rfkit-rs` checkout and ensure each cycle starts from fresh, safe repository/GitHub state.
- Do not discard unrelated local user work. Fail closed if the checkout cannot be made safe for automation.
- Preserve interrupted, dirty, or unpublished checkouts before creating a fresh one, and log the preserved path. Bound service runtime and terminate its child processes as required by `docs/CODEX_AUTOMATION.md`.
- Ensure non-interactive `HOME`, `PATH`, Git/GitHub authentication, and Codex authentication are available to the service.
- Keep stdout/stderr visible through the systemd journal.
- Prefer `Persistent=true` for timers so a powered-off host can recover a missed opportunity without inventing additional work.
- Avoid duplicate legacy timers/workers after migration.
- Do not embed Issue-selection, merge, RF, conformance, or implementation policy in unit files or shell wrappers.
- A no-work or blocked run is a successful operational outcome when repository policy says no mutation should occur.

## Recovery acceptance checks

| Observed state | Required outcome |
| --- | --- |
| Another cycle holds the shared lock | Skip without GitHub mutation or checkout cleanup. |
| One interrupted pre-PR claim; current exclusive-host context; clear ownership; no earlier recovery | Planner records evidence and returns the same Issue to ready. |
| Claim is old, but execution context or ownership is unknown | Do not infer termination; record the missing evidence. |
| Related open PR exists | Use the existing correction/merge policy, not pre-PR recovery. |
| Closed/unmerged PR history or an unexplained remote branch exists | Fail closed and escalate; never create replacement work. |
| Recovery comment exists but its label update was interrupted | Revalidate timeline/run IDs and finish only that same transition. |
| Worker claims again and stops before a PR after its one recovery | Block with evidence; no unbounded retry. |
| Host stops mid-run or the prior runtime survives a reboot | Preserve work before starting fresh; retain evidence of the interruption. |

## Updating behavior

Change autonomous-development behavior through a reviewed repository change:

- change direction/decision policy in `docs/` or `AGENTS.md`;
- change only the role-entry wording in `automation/*-prompt.txt`;
- change host scheduling/environment details in systemd configuration.

This separation keeps autonomous-development behavior inspectable from GitHub and lets ChatGPT or a human audit the same policy that scheduled Codex actually receives.

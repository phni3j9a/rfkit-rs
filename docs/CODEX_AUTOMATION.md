# Codex Automation

This document defines the host-independent execution contract for the autonomous `rfkit-rs` loop.

Repository policy decides **what** work is valid. Version-controlled prompt files in `automation/` provide the runtime role entrypoints. Systemd or another host scheduler should decide only **when and where** those prompts run.

## Source of truth

Scheduled Codex must read and follow the current repository state, especially:

- `AGENTS.md` — engineering and numerical correctness contract;
- `docs/DEVELOPMENT_DIRECTION.md` — project direction and meaningful-progress criteria;
- `docs/LOOP_ENGINEERING.md` — Planner/Worker state machine, WIP, merge, prioritization, and escalation policy;
- this document — host/execution contract;
- `automation/planner-prompt.txt` and `automation/worker-prompt.txt` — canonical runtime role-entry prompts.

Do not maintain divergent copies of the Planner or Worker prompt in systemd units, shell scripts, cron entries, personal notes, or external automation configuration. The files under `automation/` are the canonical prompt text.

## Roles

### Planner

The scheduled Planner performs one repository-maintenance/planning cycle. It uses fresh GitHub state, prioritizes existing PRs and active work, may merge only when all autonomous merge gates are satisfied, may re-dispatch the Issue linked to an existing autonomous PR for a bounded correction pass, and may dispatch at most one new implementation-ready Issue when WIP is empty and the work has meaningful marginal value.

The Planner does not implement product code and does not claim `loop:ready` work as a Worker.

### Worker

The scheduled Worker performs one implementation cycle. It may implement only an explicitly dispatched `loop:ready` Issue, claims exactly one Issue before work, follows the repository implementation/review/verification contract, and either creates its initial PR or updates its one unambiguously linked existing PR for a dispatched correction pass.

The Worker does not choose speculative roadmap work and does not claim a second Issue in the same run.

## Dispatch labels

The repository uses these machine-queryable labels:

- `loop:ready` — authorized for the next scheduled Worker pass, including bounded correction of an existing autonomous PR;
- `loop:in-progress` — the currently authorized initial or correction pass has been claimed by the Worker;
- `loop:blocked` — cannot proceed without resolution.

The three loop state labels are mutually exclusive on an open Issue. Conflicting state labels are ambiguous and do not authorize implementation.

If required labels are missing, ambiguous state exists, or more than one ready Issue violates WIP=1, automation must fail closed rather than inventing a replacement mechanism.

## Recommended cadence

After host integration is validated, run four Planner/Worker opportunities per day:

```text
03:00 JST  Planner
03:30 JST  Worker
09:00 JST  Planner
09:30 JST  Worker
15:00 JST  Planner
15:30 JST  Worker
21:00 JST  Planner
21:30 JST  Worker
```

This cadence reduces idle latency. It does not imply four Issues, implementations, PRs, or merges per day. No-op cycles are valid and expected.

## Runtime prompts

Use the prompt files directly from the checked-out repository:

```text
automation/planner-prompt.txt
automation/worker-prompt.txt
```

A host wrapper should conceptually do only enough to establish a safe checkout/non-interactive environment and then invoke the appropriate role, for example:

```sh
codex exec "$(cat automation/planner-prompt.txt)"
codex exec "$(cat automation/worker-prompt.txt)"
```

Do not copy the prompt body into the unit file. Do not add host-specific strategic instructions that compete with repository policy.

See `automation/README.md` for host-integration requirements.

## Host requirements

Before enabling scheduled execution, verify that:

1. the checkout can fetch, push, create/update Issues and PRs, and merge when authorized;
2. Codex can read the repository policy and runtime prompt files;
3. required IssueFlow/model routing used by the repository workflow is available;
4. GitHub and Codex authentication work non-interactively under the systemd service account;
5. `HOME`, `PATH`, working directory, and other required environment are explicit enough for non-interactive execution;
6. Planner and Worker share a host-level lock so they cannot mutate the same checkout concurrently;
7. unrelated local user work cannot be silently reset or overwritten;
8. stdout/stderr are retained in the systemd journal or equivalent host logs;
9. legacy scheduled workers/maintainers are disabled so the new loop is not duplicated.

### Execution evidence and interrupted work

For automatic pre-PR recovery, configure one execution host and require every scheduled or manual Planner/Worker invocation to use the same lock-holding wrapper. A second independent executor invalidates this recovery assumption. Hold the lock for the entire process tree; after a timeout or stop, terminate remaining child processes before another cycle can proceed.

After acquiring the lock, the host exposes `RFKIT_AUTOMATION_CONTEXT`, the path to a host-written JSON file outside the writable checkout. It records factual fields: `run_id`, `role`, `boot_id`, `lock_path`, `lock_holder_pid`, `started_at`, and `checkout`. The file identifies the current invocation and is replaced only under the shared lock. It must be readable by Codex and not writable by the run. Missing context disables automatic recovery; an Issue comment alone cannot substitute for this host evidence. The file is operational evidence, not a prompt or an Issue-selection instruction.

Keep run metadata and journal start/end results. Preserve a failed/interrupted checkout, including uncommitted and unpushed work, before starting a fresh checkout. Also preserve work from an apparently successful run unless the host can establish that the checkout is clean and all local commits are published. Report archive locations in the journal and retain them for inspection; do not silently delete a prior residual runtime on startup. Archive retention and eventual removal belong to host maintenance, not autonomous product work.

Set a finite service runtime limit shorter than the interval between Planner opportunities (for example, two hours for the six-hour cadence) and terminate the whole service process group on timeout. This bounds an unresponsive run without introducing another daemon or scheduler. Timeout is an execution failure, not permission for the wrapper to change a GitHub label. The next Planner decides recovery under `docs/LOOP_ENGINEERING.md` using host evidence and fresh GitHub state.

Prefer a small shell wrapper plus systemd service/timer units over introducing an additional daemon, queue, scheduler, or orchestration framework unless a demonstrated requirement cannot be met by systemd.

## Failure behavior

Planner and Worker must fail closed when safe automation cannot be established, including cases such as:

- missing/ambiguous dispatch state;
- inability to establish the required claim or merge state safely;
- unsafe/dirty checkout conditions;
- unavailable required role/model routing;
- unresolved material RF/product/API/architecture decisions;
- unavailable required standards, papers, or fixtures;
- uncertain provenance/licensing obligations;
- authentication or GitHub mutation failures.

A failure, blocked cycle, or no-work cycle must not mutate unrelated Issues or generate speculative backlog.

## ChatGPT role

Scheduled ChatGPT is not required in the normal execution path once the Codex Planner/Worker host integration is validated.

ChatGPT may be used on demand as a governor/auditor to inspect whether autonomous work remains aligned with `docs/DEVELOPMENT_DIRECTION.md`, whether recent PRs represent meaningful capability/correctness progress, whether the Planner is drifting into conformance/cleanup tunnels, and whether repository policy should be adjusted.

# AGENTS.md — rfkit-rs engineering contract

This repository is intended to be developed aggressively with AI assistance, but RF correctness must remain stronger than implementation velocity.

## Primary objective

Build a production-quality Rust-native RF/microwave network-analysis library that can eventually cover the useful functionality of scikit-rf while preserving Rust-native APIs and strong numerical verification.

Read `docs/DEVELOPMENT_DIRECTION.md` for the current repository-level development horizon and the definition of meaningful autonomous progress.

## Source hierarchy

Use sources in this order when resolving behavior:

1. RF/microwave mathematics and published standards/papers.
2. scikit-rf current behavior and tests as a reference oracle.
3. Instrument/file-format specifications (for example Touchstone).
4. rust-rf and rust-skrf as prior-art implementations, never as unquestioned truth.

## Mandatory implementation workflow

For every substantive numerical feature:

1. Identify the mathematical definition and edge cases.
2. Inspect equivalent scikit-rf behavior/tests.
3. Inspect rust-rf/rust-skrf only for prior-art comparison when useful.
4. Decide explicitly whether prior-art code should be `REUSE`, `ADAPT`, or `REWRITE`.
5. Record any copied/adapted third-party code or fixture in `docs/PROVENANCE.md` and retain required license notices.
6. Implement a Rust-native API.
7. Add deterministic unit tests.
8. Add differential/conformance fixtures against scikit-rf when applicable.
9. Add property/invariant tests where mathematically meaningful.
10. Run fmt, clippy, and the full test suite before considering the task complete.

## Numerical rules

- Do not assume 50-ohm reference impedance.
- Support complex reference impedance where the underlying operation permits it.
- Prefer N-port algorithms over special-casing 2-port behavior unless the mathematics is inherently 2-port.
- Document wave definitions where relevant (power, pseudo, traveling).
- Every conversion should have round-trip tests where mathematically valid.
- Near-singular matrices and ill-conditioned operations need deliberate error/tolerance handling, not silent garbage.
- Never claim scikit-rf compatibility without a machine-executed comparison.

## Conformance strategy

`tools/oracle/` is reserved for Python/scikit-rf reference generation. Keep oracle output deterministic and versioned by:

- scikit-rf version/commit
- NumPy version where relevant
- random seed
- operation name
- tolerance policy

Prefer generated fixtures or a reproducible test runner over hand-copied expected values.

## Architecture

Keep the numerical core independent of plotting frameworks, Python, WASM, GUI frameworks, and instrument I/O. Add integrations as separate crates when they become necessary.

## Public API

Read `docs/PUBLIC_API.md` before any work that adds or changes public RF-analysis operations.

That document is the human-approved policy for the first public `Network` surface. Implementing a bounded slice within it is authorized; changing its semantic distinctions, frequency policy, wave-policy approach, core model, or stabilization assumptions is not an implementation detail and must be escalated.

Do not accumulate additional private-only RF kernels merely to route around the approved public-API checkpoint unless fresh repository evidence shows that a private prerequisite materially blocks correctness or implementation of the approved surface.

## Third-party provenance

BSD-3-Clause permits reuse but attribution still matters. Do not erase lineage to make code appear original. If copying or closely adapting code, preserve the applicable copyright/license terms and log the source commit/path.

## Autonomous loop engineering

Read `docs/LOOP_ENGINEERING.md` when operating as a scheduled Planner or Worker.

For the scheduled Codex Planner:

- use fresh GitHub and `main` state rather than cached assumptions;
- protect WIP=1 and resolve existing autonomous work before dispatching new work;
- recover an interrupted pre-PR Worker claim on the same Issue only under the execution-evidence and retry limits in `docs/LOOP_ENGINEERING.md`; elapsed time alone never authorizes recovery;
- when an existing autonomous PR needs bounded, implementation-ready corrections, record or reuse a concrete change request for its current head and re-dispatch the same linked Issue for a correction pass instead of creating replacement work;
- do not repeat an already-sufficient change request against an unchanged PR head merely to show activity;
- choose at most one next Issue by marginal value under `docs/DEVELOPMENT_DIRECTION.md`;
- after the public-API checkpoint in `docs/DEVELOPMENT_DIRECTION.md` is reached, prefer bounded public-usability increments from `docs/PUBLIC_API.md` over additional private-only capability unless a concrete prerequisite blocks them;
- treat no-op as valid when useful bounded work is unavailable;
- do not turn README scope, scikit-rf surface area, or `docs/CONFORMANCE.md` into a mechanical backlog;
- never implement a newly dispatched Issue in the same Planner run;
- apply the autonomous merge and escalation gates in `docs/LOOP_ENGINEERING.md`.

For the scheduled Codex Worker:

- only an open Issue explicitly marked `loop:ready` is permission to start an autonomous initial implementation or correction pass;
- do not implement arbitrary open Issues merely because they exist;
- claim the selected Issue before implementation by moving it from `loop:ready` to `loop:in-progress`;
- record the host run ID, base commit, and intended branch on the claimed Issue before editing, so an interrupted pass can be identified;
- process at most one dispatched Issue per Worker run;
- when the dispatched Issue has exactly one unambiguously linked, safely writable open autonomous PR, apply the authorized correction to that PR's existing branch and do not create a duplicate PR;
- fail closed when the Issue-to-PR relationship, correction request, or writable head branch is ambiguous;
- use the Issue as the product contract and own the repository-specific implementation plan after inspection;
- for public RF-operation work, implement within `docs/PUBLIC_API.md` and escalate rather than silently widening or reinterpreting that approved surface;
- follow the configured IssueFlow `issue-to-pr` workflow: Codex Main plans/orchestrates, Luna MAX implements bounded product-code tasks, and a fresh Sol XHIGH independently reviews the integrated candidate;
- Main owns deterministic verification, review-finding adjudication, Git, and PR creation;
- do not silently substitute unspecified child models when required role routing is unavailable;
- if a material product/RF/licensing decision is unresolved, mark blocked and surface it rather than guessing.

The `loop:*` labels are dispatch state, not feature taxonomy. Keep design decisions in repository policy, Issues, and PRs rather than encoding them in labels.

## Avoid

- bulk-porting thousands of lines before the conformance harness exists
- README feature checkboxes unsupported by tests
- Python API mimicry that fights Rust's type system
- introducing plotting/UI dependencies into the numerical core
- optimizing before correctness is characterized
- scheduled implementation of untriaged open Issues without `loop:ready`
- pre-generating a long autonomous roadmap when WIP=1 is configured
- repeated conformance, cleanup, refactor, or documentation work without concrete marginal value
- adding private-only RF capabilities primarily to avoid the approved public-API checkpoint
- measuring autonomous-development quality by commit, Issue, PR, or test-count velocity alone

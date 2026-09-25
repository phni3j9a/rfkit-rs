# Development Direction

This document gives autonomous engineering agents the repository-level direction needed to choose useful work without turning the project scope into a mechanical checklist.

## North star

Build a production-quality Rust-native RF/microwave network-analysis library that covers the useful analytical capabilities expected from mature RF tooling while preserving Rust-native APIs, portability, explicit numerical semantics, and strong verification.

The project should become more **capable**, not merely larger.

## Current development horizon

The current horizon is a coherent end-to-end RF workflow: callers should be able to construct or load network data, apply verified transformations and composition, and inspect or export useful results without rebuilding private pipelines themselves.

Prefer progress that strengthens one or more of:

1. numerical correctness and explicit RF semantics;
2. foundational N-port operations with downstream leverage;
3. network composition and transformation capability;
4. practical data ingress, analysis, and egress through a coherent public API;
5. conformance evidence needed to make those capabilities safe.

The README initial scope is directional context, not an ordered backlog.

### Completed public baseline

The first public `Network` surface defined in `docs/PUBLIC_API.md` is implemented: power-wave Z/Y conversion and renormalization, exact physical-port permutation, Cartesian interpolation, exact-grid and explicit-grid matched connection, and matched inner connection are available through verified public methods.

That list is a baseline, not a permanent allowlist or ceiling. The next useful increments should close concrete workflow gaps, with data ingress/egress and additional high-leverage RF transformations generally worth more than further private-only kernels. This is an outcome horizon rather than a fixed backlog; current repository evidence still determines the next bounded increment.

### Autonomous decision envelope

`docs/LOOP_ENGINEERING.md` defines Green, Yellow, and Red autonomy classes, while `docs/PUBLIC_API.md` applies them to public RF design. The Planner may dispatch and merge Green work and reversible, explicitly recorded Yellow decisions without per-method human approval. Red decisions remain human-governed.

A Red or otherwise blocked decision is local to its Issue. When it has no unresolved implementation or PR consuming the WIP slot, the Planner should continue with an independent eligible Green or Yellow increment rather than treating one design question as a repository-wide stop.

## What counts as meaningful progress

A sequence of autonomous changes should make it possible to answer at least one of these questions with a concrete new answer:

- What RF operation can the library now perform that it could not perform before?
- What previously unsafe or ambiguous behavior is now objectively characterized or corrected?
- What foundational primitive now unlocks multiple useful downstream operations?
- What verified internal capability is now usable through an explicit public `Network` surface?
- Which RF operation now has fewer public entry points to choose among, with no loss of domain or explicit semantics?
- What important implementation or API decision can now be made safely because a concrete blocker was retired?

A run may legitimately do nothing when no high-value bounded increment is available.

## Marginal-value selection

When choosing among plausible next increments, reason about:

- correctness risk reduced;
- capability unlocked;
- downstream leverage;
- user-facing RF value;
- implementation and review cost;
- complexity introduced;
- whether existing evidence already makes the proposed work redundant.

Do not reduce this to a fixed numeric score. The purpose is disciplined comparison, not false precision.

## Avoid local-optimum development

Do not autonomously:

- consume `docs/CONFORMANCE.md` line by line as a roadmap;
- generate one Issue per missing scikit-rf method;
- add tests, fixtures, metadata, abstractions, or refactors mainly because they are easy to name;
- repeatedly polish already-strong foundations while higher-value capability work is available;
- continue adding private-only kernels merely to avoid making a justified public workflow usable;
- add a coexisting public variant of an existing operation, differing only in evaluation strategy or a narrower domain, when generalizing the existing operation is feasible;
- introduce vague public defaults or abstractions outside the autonomy envelope simply to make the next task convenient;
- create speculative multi-Issue roadmaps that become stale before implementation.

After two consecutive merged conformance-only increments, another conformance-only increment requires concrete repository evidence that it blocks correctness or safe capability growth.

After several consecutive cleanup/refactor/documentation-only increments, the planner should explicitly ask whether the library's RF capability is actually advancing before selecting more of the same.

The opposite failure also matters. Capability can advance while the public surface fragments into several variants per operation. `docs/LOOP_ENGINEERING.md` therefore makes public-surface consolidation reachable: the Planner audits the surface periodically, and consolidating a recorded overlap ranks with bounded RF capability.

## Human decision boundary

Human approval is reserved for Red decisions under `docs/LOOP_ENGINEERING.md`, especially:

- declaring or changing an API stability or compatibility promise;
- release, package publication, signing, credential use, or another irreversible external action;
- externally visible RF behavior where authoritative sources materially disagree and explicit side-by-side semantics cannot resolve the conflict;
- unavailable paid standards, papers, datasets, or fixtures that are necessary to establish correctness;
- unresolved licensing, copyright, attribution, or provenance obligations;
- a major, difficult-to-reverse change to the canonical data model, storage representation, or crate/repository architecture;
- security or safety policy beyond a bounded defect fix;
- relaxation of WIP, independent review, autonomous merge, or this decision envelope.

Provisional public API design, ordinary dependency choices, and bounded architecture changes are not Red merely because they are externally visible. They may proceed as Green or Yellow when the semantics are explicit, the evidence and alternatives are recorded in proportion to risk, and the change remains reversible before stabilization.

## Role of ChatGPT

ChatGPT is not part of the normal scheduled execution path.

Its role is repository governor and auditor:

- help the human owner define or revise objectives and Red boundaries;
- inspect autonomous development when asked;
- determine whether recent work is materially advancing RF capability rather than merely producing activity;
- identify loops, tunnel vision, over-engineering, weak task selection, or policy drift;
- propose policy corrections when the autonomous system is optimizing the wrong thing;
- review batches of already-merged Green/Yellow decisions instead of requiring routine pre-approval.

GitHub documents remain the source of truth for the autonomous system. Chat history is not an operational dependency.

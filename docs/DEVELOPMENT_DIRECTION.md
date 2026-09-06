# Development Direction

This document gives autonomous engineering agents the repository-level direction needed to choose useful work without turning the project scope into a mechanical checklist.

## North star

Build a production-quality Rust-native RF/microwave network-analysis library that covers the useful analytical capabilities expected from mature RF tooling while preserving Rust-native APIs, portability, explicit numerical semantics, and strong verification.

The project should become more **capable**, not merely larger.

## Current development horizon

The current horizon is a trustworthy RF `Network` core with enough verified primitives to support real composition and analysis workflows.

Prefer progress that strengthens one or more of:

1. numerical correctness and explicit RF semantics;
2. foundational N-port operations with downstream leverage;
3. network composition and transformation capability;
4. practical analysis capability that can support a coherent public API;
5. conformance evidence needed to make those capabilities safe.

The README initial scope is directional context, not an ordered backlog.

### Public-API checkpoint

The verified private `Network` foundations have reached the point where public usability is becoming higher-value than continuing to accumulate internal kernels. The human-approved first public surface and its semantic constraints are defined in `docs/PUBLIC_API.md`.

Once the currently active explicit-grid matched-composition work is merged (PR #50 or an equivalent successor), the default next direction is to expose those verified capabilities through bounded public `Network` API increments under `docs/PUBLIC_API.md`.

Do not route around this checkpoint by generating additional private-only RF kernels merely because public API work was previously an escalation boundary. A private prerequisite remains valid only when fresh repository evidence shows that it materially blocks correctness or implementation of the approved public surface.

The approved design resolves the decisions written in `docs/PUBLIC_API.md`; implementing within those bounds does not require repeated human escalation. Material deviation from that design still does.

## What counts as meaningful progress

A sequence of autonomous changes should make it possible to answer at least one of these questions with a concrete new answer:

- What RF operation can the library now perform that it could not perform before?
- What previously unsafe or ambiguous behavior is now objectively characterized or corrected?
- What foundational primitive now unlocks multiple useful downstream operations?
- What verified internal capability is now usable through the approved public `Network` surface?
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
- continue adding private-only kernels merely to avoid the approved public-API checkpoint;
- freeze or broaden public APIs beyond `docs/PUBLIC_API.md` simply to make the next autonomous task convenient;
- create speculative multi-Issue roadmaps that become stale before implementation.

After two consecutive merged conformance-only increments, another conformance-only increment requires concrete repository evidence that it blocks correctness or safe capability growth.

After several consecutive cleanup/refactor/documentation-only increments, the planner should explicitly ask whether the library's RF capability is actually advancing before selecting more of the same.

## Human decision boundary

Autonomous development must stop and escalate rather than invent policy when the next useful step requires a material decision about:

- public API semantics, models, defaults, stabilization, or breaking behavior not already approved by `docs/PUBLIC_API.md`;
- competing RF definitions or wave conventions;
- externally visible behavior where authoritative sources disagree;
- unavailable paid standards, papers, datasets, or fixtures;
- licensing, copyright, attribution, or provenance uncertainty;
- major crate/repository architecture;
- releases, package publication, signing, or irreversible external actions.

## Role of ChatGPT

ChatGPT is not part of the normal scheduled execution path.

Its role is repository governor and auditor:

- help the human owner define or revise this direction;
- inspect autonomous development when asked;
- determine whether recent work is materially advancing RF capability rather than merely producing activity;
- identify loops, tunnel vision, over-engineering, weak task selection, or policy drift;
- propose policy corrections when the autonomous system is optimizing the wrong thing.

GitHub documents remain the source of truth for the autonomous system. Chat history is not an operational dependency.

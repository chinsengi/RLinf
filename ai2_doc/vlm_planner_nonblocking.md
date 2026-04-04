# Nonblocking VLM Planner Design

This note describes how to make `VLMPlannerWorker` nonblocking with respect to
the env, rollout, and actor workers in the staged YAM PPO pipeline.

It is a design document only. It does not change runtime behavior yet.

For the current blocking architecture, see [training_architecture](training_architecture.md).

## Problem

Today `EnvWorker` calls the VLM planner synchronously:

- `_compute_top_reward()` issues `compute_top_reward.remote(...)` and then `ray.get(...)`
- `_maybe_update_subtask()` issues `get_next_subtask.remote(...)` and then `ray.get(...)`

Because those waits happen before the env sends the next observation to rollout,
the planner sits on the critical path:

1. rollout cannot receive `obs_{t+1}` until planner work for step `t` finishes
2. actor cannot receive completed trajectories until env has resolved planner work
3. the async PPO runner therefore overlaps actor and rollout less than intended

The planner is already isolated onto its own GPU. The main issue is not
placement; it is synchronous handoff semantics.

## Goals

- Remove planner latency from the `env -> rollout` critical path
- Preserve existing PPO reward semantics as closely as practical
- Preserve current task/subtask safety for `RemoteEnv.task_description`
- Keep compatibility with both `transformers` and `sglang_http` backends
- Fit RLinf's current worker/channel architecture without requiring a new global scheduler

## Non-Goals

- Replacing TOPReward with a different reward function
- Changing PPO math or actor update logic
- Solving planner throughput for large multi-slot training in the first step

TODO(agent): the current implementation still restricts TOPReward to one
pipeline slot. The design below keeps that assumption for the first rollout,
but the state model is intentionally written so it can become per-slot later.

## Design Summary

The planner should become nonblocking by splitting its role into two phases:

1. submit planner work without waiting
2. resolve planner results later, outside the `env -> rollout` handoff

Concretely:

- `EnvWorker` submits TOPReward and subtask requests asynchronously
- `EnvWorker` sends the next observation to rollout immediately after `chunk_step()`
- planner results are resolved in a separate drain/finalize path
- trajectories are sent to actor only after all reward fields in that trajectory are finalized

This keeps rollout sampling moving while planner GPU work runs in parallel.

## Current Blocking Path

Current step ordering is:

1. rollout sends actions/logprobs/values to env
2. env runs `chunk_step()`
3. env blocks on planner RPCs
4. env sends next obs to rollout
5. rollout predicts again

The blocking call is therefore between steps 2 and 4. That is the wrong place
for planner latency if the goal is to keep rollout busy.

## Proposed Worker Model

### 1. Keep Planner on a Dedicated GPU

The planner should remain a dedicated Ray actor on its own GPU, as it is now.
The nonblocking change is about request semantics, not device topology.

### 2. Add Planner Sessions

Instead of sending the full frame history on every TOPReward call, the planner
should own per-episode session state.

Suggested planner-side state:

- `slot_id`
- `episode_id`
- `segment_id`
- `frame_buffer`
- `last_resolved_score`
- `pending_request_queue`

Suggested terminology:

- `episode_id`: increments on env reset / episode end
- `segment_id`: increments when the active task description changes
- `chunk_id`: monotonic chunk-step index within an episode

`segment_id` is needed because TOPReward scores are only comparable within the
same instruction segment. A subtask change must reset the baseline.

### 3. Split Planner API into Fire-and-Resolve

Suggested API shape:

```python
open_session(slot_id: int, episode_id: int, initial_task: str) -> None
append_frame(slot_id: int, episode_id: int, chunk_id: int, frame: np.ndarray) -> None

submit_top_reward(
    slot_id: int,
    episode_id: int,
    chunk_id: int,
    instruction: str,
    segment_id: int,
) -> PlannerTicket

submit_subtask(
    slot_id: int,
    episode_id: int,
    source_chunk_id: int,
    main_task: str,
    images: list[np.ndarray],
    segment_id: int,
) -> PlannerTicket

poll_ready(ticket_ids: list[str]) -> list[PlannerResult]
close_session(slot_id: int, episode_id: int) -> None
```

Key property: `submit_*` must return immediately. No `ray.get(...)` is allowed
on the env hot path.

The `PlannerTicket` / `PlannerResult` payload should include:

- request kind: `top_reward` or `subtask`
- `slot_id`
- `episode_id`
- `segment_id`
- `chunk_id` or `source_chunk_id`
- planner output
- timing / status metadata

## EnvWorker Changes

`EnvWorker` should split into three logical responsibilities.

### A. Control Loop

This is the existing action/observation loop, but it stops waiting for planner
results inline.

After `chunk_step()`:

1. update local per-slot bookkeeping
2. submit planner work
3. send `obs_{t+1}` to rollout immediately

That means the env loop no longer waits for:

- TOPReward score for step `t`
- next subtask text generated from the latest image

### B. Planner Drain Loop

The env worker needs a background path that continuously resolves ready planner
results and applies them to per-slot state.

Suggested env-side state:

```python
PendingTopRewardStep:
    episode_id
    segment_id
    chunk_id
    reward_ticket
    env_output_ref

PendingSubtaskRequest:
    episode_id
    segment_id
    source_chunk_id
    ticket
```

The drain loop:

- polls ready planner tickets
- writes TOPReward deltas into pending step records
- applies subtask updates if they are still current
- records metrics like latency, queue depth, and stale drops

### C. Trajectory Finalizer

Actor training still requires concrete rewards, dones, and values. So before a
trajectory is emitted to `actor_channel`, the env worker must ensure all
planner-derived rewards in that trajectory are resolved.

This means:

- rollout should be nonblocked
- actor handoff should remain correctness-preserving
- unresolved planner work becomes a trajectory-finalization delay, not a rollout delay

In the async runner this is still useful because rollout can continue generating
future epochs while planner results for earlier epochs resolve.

## TOPReward Resolution Semantics

### Per-Step Score Ordering

TOPReward is currently based on score deltas:

`reward_t = score_t - score_{t-1}`

That means score application must be ordered by:

- `episode_id`
- `segment_id`
- `chunk_id`

Planner results may complete out of order. The env worker must therefore buffer
resolved scores until all earlier scores in the same segment have been applied.

Suggested env-side fields:

- `last_applied_score`
- `last_applied_chunk_id`
- `resolved_score_buffer: dict[int, float]`

### Baseline Resets

The score baseline must reset when:

- a new episode starts
- the episode ends
- a new subtask is applied

That mirrors current behavior while removing the synchronous planner wait.

### Why Actor Can Stay Unchanged

Actor training does not need planner work to finish immediately after each env
step. It only needs rewards to be finalized before the completed trajectory is
converted into the actor batch.

So the actor API can stay unchanged if env finalizes planner-backed rewards
before calling `send_rollout_trajectories(...)`.

## Subtask Planning Semantics

Subtask planning is different from TOPReward because it changes future language
conditioning instead of filling a scalar field in an already-generated step.

The safe rule is:

- subtask updates are applied only to future observations
- subtask results are dropped if they are stale

A subtask result is stale if:

- `episode_id` no longer matches
- `segment_id` no longer matches
- a newer subtask has already been applied

Recommended policy:

- allow at most one inflight subtask request per slot
- if a newer trigger fires while one is inflight, skip or coalesce rather than queueing many requests

This prevents an old planner result from overwriting a newer task description.

## Planner Worker Internals

### Serial Compute, Async Interface

For correctness and backend simplicity, the planner worker can remain
single-model and internally serialized even after becoming nonblocking.

That is:

- env/rollout/actor should not block on planner execution
- planner itself may still execute requests one at a time

This is enough for the first design milestone because the planner already has
its own GPU and the main user-visible issue is env-side waiting.

### Session-Owned Frame Buffer

The planner should own the frame buffer instead of receiving the entire
`episode_frames` list on every request.

Benefits:

- less serialization overhead between env and planner
- less CPU memory churn in `EnvWorker`
- simpler per-segment reset logic
- easier future batching inside planner

### Optional Future Improvement: Request Coalescing

Subtask requests can be coalesced aggressively because only the newest one
matters.

TOPReward requests cannot be dropped as freely because PPO still needs the
score delta for each finalized step.

## Runner and Channel Impact

The staged async PPO runner can keep the same top-level structure:

- env loop remains a long-running task
- rollout loop remains a long-running task
- actor loop still consumes completed trajectories

The major behavioral change is just where waiting happens:

- current: wait before env sends obs to rollout
- proposed: wait only when finalizing a trajectory for actor

This preserves most of the existing `Channel` topology:

- `env_channel`: unchanged
- `rollout_channel`: unchanged
- `actor_channel`: unchanged

No new global data plane is required for the first implementation. Planner
tickets can be tracked with Ray object refs or a thin planner polling API.

## Failure Handling

Recommended failure behavior:

- TOPReward failure: record warning, mark the step as resolved with fallback delta `0.0`
- subtask failure: drop the result and keep the current task description
- stale subtask result: drop silently or metric-only log
- planner timeout during trajectory finalization: configurable fallback to `0.0` reward and no subtask update

This keeps training alive while making planner failures observable in metrics.

## Metrics to Add

- planner request latency by kind
- planner queue depth
- pending TOPReward steps per slot
- pending subtask requests per slot
- actor wait time attributable to unresolved planner tickets
- stale subtask result count
- TOPReward fallback count

These metrics are required to show that the redesign actually improves overlap.

## Suggested Rollout Plan

### Phase 1: Env-Side Deferred Resolution

Minimal code movement:

- keep the current planner compute APIs
- replace inline `ray.get(...)` with deferred object-ref tracking
- finalize planner rewards only before trajectory handoff

Pros:

- smallest code diff
- proves the nonblocking control flow

Cons:

- still serializes full frame histories every step

### Phase 2: Planner Sessions

Add planner-owned frame buffers and explicit session ids.

Pros:

- lower serialization cost
- cleaner reward/subtask state management

Cons:

- larger API change between env and planner

### Phase 3: Optional Planner Dispatcher / Microbatching

If planner GPU becomes the dominant bottleneck even after overlap:

- add internal planner request batching
- or split dispatch from inference into separate actors

This is only worth doing after Phase 1 and Phase 2 are measured.

## Testing Plan

Minimum tests needed for implementation:

- unit: out-of-order TOPReward responses still produce ordered deltas
- unit: stale subtask results are dropped
- unit: subtask application increments segment id and resets score baseline
- unit: trajectory finalizer waits for unresolved planner steps
- unit: planner timeout falls back to `0.0` reward without crashing
- integration: rollout continues to receive observations while planner requests remain unresolved
- integration: actor receives only fully resolved trajectories

## Recommended First Implementation

The best first implementation is:

1. keep the current planner worker process model
2. add deferred planner ticket tracking in `EnvWorker`
3. move planner waits from the env hot path to trajectory finalization
4. measure overlap improvement
5. only then add planner-side sessions

That sequence gives the highest confidence-to-diff ratio.

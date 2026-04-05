# TODO

Current discovered problems to track:

## Open issues

### 1. Remote YAM cooldown chunks are entering training data

Status: open

Context:
- During `RobotServer` restart/cooldown, `ChunkStep()` returns an idle chunk with zero rewards and a final-step `truncated=True` flag.
- `RemoteEnv.chunk_step()` forwards that chunk as a normal env result.
- `EnvWorker` appends the chunk into rollout trajectories.
- In the current remote YAM config, `ignore_terminations: true`, so the actor path does not build a `loss_mask` to filter these cooldown steps.

Impact:
- The actor can train on zero-reward idle/cooldown actions, which contaminates learning.

Relevant files:
- `rlinf/envs/remote/robot_server.py`
- `rlinf/envs/remote/remote_env.py`
- `rlinf/workers/env/env_worker.py`
- `rlinf/workers/actor/fsdp_actor_worker.py`
- `rlinf/utils/metric_utils.py`

Potential directions:
- Treat cooldown as a collection pause instead of an RL transition.
- Block `ChunkStep()` until restart completes.
- Or return an explicit non-training signal and skip `append_step_result()` / `append_transitions()` for cooldown chunks.

### 2. TOPReward state can leak across episode boundaries

Status: open

Context:
- When TOPReward and adaptive subtask planning are both enabled, terminal steps can leave planner reward state populated across an env episode boundary.
- The next subtask update can then use stale reward/plateau state from the previous episode.

Impact:
- A new episode can start with a subtask plan derived from the previous episode’s TOPReward state.

Relevant files:
- `rlinf/workers/env/vlm_planner_client.py`
- `rlinf/workers/env/env_worker.py`

Potential directions:
- Clear terminal TOPReward state before any post-step adaptive subtask check runs for the next episode.

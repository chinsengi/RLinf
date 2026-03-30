# YAM PPO + TOPReward + Subtask Planning (`yam_ppo_openpi_subtask_async` / `yam_ppo_openpi_subtask_sync`)

This is the AI2-facing Markdown guide for the staged YAM PPO configs:
`examples/embodiment/config/yam_ppo_openpi_subtask_async.yaml` and
`examples/embodiment/config/yam_ppo_openpi_subtask_sync.yaml`.

This config runs:

- PPO + GAE
- π₀.5 / OpenPI policy
- TOPReward dense reward
- VLM subtask planning enabled (`subtask_interval > 0`)

Runtime split:

- `yam_ppo_openpi_subtask_async` uses
  `train_embodied_agent_staged_async.py` and
  `algorithm.loss_type: decoupled_actor_critic`
- `yam_ppo_openpi_subtask_sync` uses `train_embodied_agent_staged.py` and
  `algorithm.loss_type: actor_critic`

For the simpler TOPReward-only variant, see
[yam_ppo_openpi](yam_ppo_openpi.md).

## Topology

Canonical topology:

- Beaker runs all Ray workers
- the desktop runs only `RobotServer`
- `RemoteEnv` connects over gRPC through the reverse SSH tunnel

Main component placement:

- GPU 0: actor
- GPU 1: rollout
- GPU 2: VLM planner / TOPReward / subtask planning
- CPU: `RemoteEnv`

## VLM Planner Placement

`train_embodied_agent_staged_async.py` now launches the VLM planner through RLinf's
placement stack instead of as a standalone Ray `num_gpus=1` actor. This matters
because actor and rollout workers were already using RLinf-managed GPU
isolation, while the older VLM planner path only used a best-effort
`CUDA_VISIBLE_DEVICES` heuristic.

Current behavior:

- actor, rollout, and VLM planner all use RLinf placement-backed GPU isolation
- the default YAM single-node layout still resolves to GPU 0 for actor, GPU 1
  for rollout, and GPU 2 for the VLM planner
- the planner uses the `beaker_vlm` node group by default

Planner placement overrides:

```yaml
vlm_planner:
  # Optional explicit GPU index inside the planner node group.
  placement: 2
  # Optional node-group override. Defaults to "beaker_vlm".
  node_group: "beaker_vlm"
```

Use an explicit `vlm_planner.placement` override if you want to pin the VLM to
a different GPU than the default inferred one.

## Standard Workflow

Submit training from the repo root:

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_subtask_async \
    --model-path /path/to/RLinf-Pi05-SFT

bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_subtask_sync \
    --model-path /path/to/RLinf-Pi05-SFT
```

Then start the desktop-side robot server:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --use-follower-servers \
    --remote-host <tailscale-ip>
```

For pipeline testing without hardware:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --dummy
```

## Key Config Knobs

Model paths:

```yaml
rollout:
  model:
    model_path: "/path/to/RLinf-Pi05-SFT"

actor:
  model:
    model_path: "/path/to/RLinf-Pi05-SFT"

vlm_planner:
  model_path: "Qwen/Qwen3-VL-8B-Instruct"
```

Task description (must be a concrete episode goal, not a generic capability label):

```yaml
env:
  train:
    task_description: "fold the towel"
```

Reward / planner settings:

```yaml
env:
  train:
    top_reward_enabled: True
    top_reward_max_frames: 16
    # Anchor TOPReward to the episode-level goal (stable across subtask changes).
    # Use "current_task" only if stage-conditioned dense reward is needed.
    top_reward_instruction_source: initial_task
    subtask_interval: 1

vlm_planner:
  max_new_tokens_subtask: 64
  max_new_tokens_reward: 16
  success_threshold: 0.5
```

Subtask planning requires a non-empty `task_description`. The planner receives
the main task and the current observation image — there is no planner memory
buffer. The prompt asks the VLM for the next subtask given the episode goal and
the current visual context.

## Local Simulated Desktop Mode

`train_embodied_agent_staged_async.py` now supports simulating the remote desktop
input path locally. This keeps the normal `RemoteEnv -> gRPC -> RobotServer`
flow, but the training process starts a local dummy `RobotServer`
automatically, so no separate desktop machine or reverse SSH tunnel is needed.

Enable it with:

```bash
python examples/embodiment/train_embodied_agent_staged_async.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_subtask_async \
    env.remote_desktop_simulation.enabled=true

python examples/embodiment/train_embodied_agent_staged.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_subtask_sync \
    env.remote_desktop_simulation.enabled=true
```

Optional overrides:

```bash
python examples/embodiment/train_embodied_agent_staged_async.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_subtask_async \
    env.remote_desktop_simulation.enabled=true \
    env.remote_desktop_simulation.env_config_path=/path/to/yam_pi05_follower.yaml \
    env.train.remote_server_url=localhost:50051 \
    env.eval.remote_server_url=localhost:50051

python examples/embodiment/train_embodied_agent_staged.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_subtask_sync \
    env.remote_desktop_simulation.enabled=true \
    env.remote_desktop_simulation.env_config_path=/path/to/yam_pi05_follower.yaml \
    env.train.remote_server_url=localhost:50051 \
    env.eval.remote_server_url=localhost:50051
```

Shared config block:

```yaml
env:
  remote_desktop_simulation:
    enabled: true
    dummy: true
    env_config_path: null
    startup_timeout: 30.0
```

Notes:

- Only local RemoteEnv URLs are supported in this mode, such as
  `localhost:50051` or `127.0.0.1:50051`.
- This mode is for dummy input simulation only.
- For real hardware, keep using `start_robot_server.sh` on the desktop.

## Beaker End-to-End Validation With Simulated Robot Input

Use this when you want to verify the full staged YAM pipeline on Beaker without
real hardware or a desktop SSH tunnel. This path exercises the same staged
entrypoint as production, including TOPReward and subtask planning, but the
dummy `RobotServer` is launched locally inside the Beaker session.

### Step 1: Start an interactive Beaker session

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_subtask_async \
    --interactive --allow-dirty

bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_subtask_sync \
    --interactive --allow-dirty
```

Attach after Beaker prints the session ID:

```bash
beaker session attach <session-id>
cd /weka/oe-training-default/shiruic/RLinf
source .venv/bin/activate
```

### Step 2: Run staged training with simulated desktop mode enabled

```bash
python examples/embodiment/train_embodied_agent_staged_async.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_subtask_async \
    env.remote_desktop_simulation.enabled=true \
    env.remote_desktop_simulation.dummy=true \
    env.train.remote_server_url=localhost:50051 \
    env.eval.remote_server_url=localhost:50051 \
    actor.model.model_path=thomas0829/folding_towel_pi05 \
    rollout.model.model_path=thomas0829/folding_towel_pi05

python examples/embodiment/train_embodied_agent_staged.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_subtask_sync \
    env.remote_desktop_simulation.enabled=true \
    env.remote_desktop_simulation.dummy=true \
    env.train.remote_server_url=localhost:50051 \
    env.eval.remote_server_url=localhost:50051 \
    actor.model.model_path=thomas0829/folding_towel_pi05 \
    rollout.model.model_path=thomas0829/folding_towel_pi05
```

This uses the same 3-GPU staged layout as the normal Beaker run:

- GPU 0: actor
- GPU 1: rollout
- GPU 2: VLM planner / TOPReward / subtask planning

### What to look for

The run is wired correctly if you see all of the following:

- a startup log from `train_embodied_agent_staged_async.py` saying it is starting a
  simulated `RobotServer` on `localhost:50051`
- actor, rollout, env, and `VLMPlannerWorker` all start successfully
- `EnvWorker` logs `TOPReward: score=..., delta=...`
- new subtask text appears during rollout
- training proceeds past rollout collection into advantage computation and at
  least one policy update

If this works, the following staged path has been validated inside Beaker:

`RemoteEnv -> local dummy RobotServer -> env output -> TOPReward / subtask planner -> GAE -> PPO update`

## Related Docs

- [quickstart](quickstart.md)
- [training_architecture](training_architecture.md)
- [network_infrastructure](network_infrastructure.md)

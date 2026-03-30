# YAM PPO + TOPReward (`yam_ppo_openpi_async` / `yam_ppo_openpi_sync`)

This is the AI2-facing Markdown guide for the baseline YAM PPO configs:
`examples/embodiment/config/yam_ppo_openpi_async.yaml` and
`examples/embodiment/config/yam_ppo_openpi_sync.yaml`.

This config runs:

- PPO + GAE
- π₀.5 / OpenPI policy
- TOPReward dense reward
- no VLM subtask planning (`subtask_interval: 0`)

Runtime split:

- `yam_ppo_openpi_async` uses `train_embodied_agent_staged_async.py` and
  `algorithm.loss_type: decoupled_actor_critic`
- `yam_ppo_openpi_sync` uses `train_embodied_agent_staged.py` and
  `algorithm.loss_type: actor_critic`

For the variant that also enables VLM subtask planning, see
[yam_ppo_openpi_subtask](yam_ppo_openpi_subtask.md).

## Topology

Canonical topology:

- Beaker runs all Ray workers
- the desktop runs only `RobotServer`
- `RemoteEnv` connects over gRPC through the reverse SSH tunnel

Main component placement:

- GPU 0: actor
- GPU 1: rollout
- GPU 2: VLM planner / TOPReward
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

Planner backend options:

- `backend: "transformers"` keeps Qwen3-VL inside `VLMPlannerWorker` and is the
  default in the shipped configs.
- `backend: "sglang_http"` keeps preprocessing in `VLMPlannerWorker` but sends
  tokenized requests to an external SGLang server via `vlm_planner.server_url`.

External SGLang serve example:

```bash
git clone --branch sglang-http https://github.com/xslingcn/sglang.git ../sglang
cd ../sglang
python -m sglang.launch_server \
    --model-path ~/Model/qwen-vl-instruct \
    --host 127.0.0.1 \
    --port 30000 \
    --skip-tokenizer-init
```

Use `xslingcn/sglang` branch `sglang-http` for this path.

Matching config override:

```yaml
vlm_planner:
  backend: "sglang_http"
  server_url: "http://127.0.0.1:30000"
  request_timeout: 120.0
```

Observed numerical alignment against `../TOPReward` on 8 synthetic cases:

- mean absolute delta: `0.15625`
- max absolute delta: `0.25` when compared at the same effective `24 fps`
- one larger `0.625` outlier only appeared when RLinf was run at `2 fps` while
  `../TOPReward` fell back internally to `24 fps` due to missing `video_metadata`

## Standard Workflow

Submit training from the repo root:

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_async \
    --model-path /path/to/RLinf-Pi05-SFT

bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_sync \
    --model-path /path/to/RLinf-Pi05-SFT
```

Then start the desktop-side robot server:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --train-config examples/embodiment/config/yam_ppo_openpi_async.yaml \
    --use-follower-servers \
    --remote-host <tailscale-ip>
```

For pipeline testing without hardware:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --train-config examples/embodiment/config/yam_ppo_openpi_async.yaml \
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

Task description:

```yaml
env:
  train:
    task_description: "Fold the towel."
```

Reward / planner settings:

```yaml
env:
  train:
    top_reward_enabled: True
    subtask_interval: 0
```

Timing / shutdown behavior:

```yaml
env:
  return_home_minutes: 2
  train:
    control_rate_hz: 10.0
    max_episode_steps: 24000
    max_steps_per_rollout_epoch: 24000
    reset_on_rollout_epoch: True
```

Desktop server timing:

```yaml
# examples/embodiment/config/env/yam_pi05_follower.yaml
episode_duration_s: 120
episode_cooldown_minutes: 1
client_idle_timeout_s: 0
```

- Change only `env.return_home_minutes` when you want a different cadence.
- Change only `env.server_cooldown_minutes` when you want a different restart
  wait time on the desktop server.
- At `10 Hz`, `40` minutes becomes `24000` steps.
- `episode_duration_s` is the desktop-side hard stop: once it expires, the
  server returns to home, starts the cooldown countdown, then restarts from
  home instead of continuing the old chunk.
- `episode_cooldown_minutes` controls how long the server waits at home before
  accepting the restarted episode. Set it to `0` for an immediate restart.
- `client_idle_timeout_s: 0` disables the desktop-side "no RPC seen" watchdog,
  so the server keeps running until you explicitly stop it. Set a positive
  number of seconds only if you want automatic safe recovery after a silent
  Beaker disconnect.
- The YAM OpenPI training configs now use the async PPO runtime
  (`algorithm.loss_type: decoupled_actor_critic`) so the robot can keep
  executing rollout chunks while actor training runs in the background.
- The matching sync config keeps the same topology and reward setup, but uses
  `train_embodied_agent_staged.py` with `algorithm.loss_type: actor_critic`.
- A Beaker-side `Ctrl+C` now asks the desktop server to return home and enter
  zero-torque / zero-gravity while staying alive for the next client.
- A desktop-side `Ctrl+C` still performs the full local shutdown: return home,
  enter zero-torque, then stop the server.
- The Beaker main process now also shows a bottom-line countdown during the
  rollout phase so you can see the same home-return cycle from the training
  terminal.

## Local Simulated Desktop Mode

`train_embodied_agent_staged_async.py` now supports simulating the remote desktop
input path locally. This keeps the normal `RemoteEnv -> gRPC -> RobotServer`
flow, but the training process starts a local dummy `RobotServer`
automatically, so no separate desktop machine or reverse SSH tunnel is needed.

Enable it with:

```bash
python examples/embodiment/train_embodied_agent_staged_async.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_async \
    env.remote_desktop_simulation.enabled=true

python examples/embodiment/train_embodied_agent_staged.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_sync \
    env.remote_desktop_simulation.enabled=true
```

Optional overrides:

```bash
python examples/embodiment/train_embodied_agent_staged_async.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_async \
    env.remote_desktop_simulation.enabled=true \
    env.remote_desktop_simulation.env_config_path=/path/to/yam_pi05_follower.yaml \
    env.train.remote_server_url=localhost:50051 \
    env.eval.remote_server_url=localhost:50051

python examples/embodiment/train_embodied_agent_staged.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_sync \
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

Use this when you want to verify the full YAM pipeline on Beaker without real
hardware or a desktop SSH tunnel. This path keeps the normal
`RemoteEnv -> gRPC -> RobotServer` flow, but the staged training script starts
the dummy `RobotServer` inside the Beaker session.

### Step 1: Start an interactive Beaker session

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_async \
    --interactive --allow-dirty

bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_sync \
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
    --config-name yam_ppo_openpi_async \
    env.remote_desktop_simulation.enabled=true \
    env.remote_desktop_simulation.dummy=true \
    env.train.remote_server_url=localhost:50051 \
    env.eval.remote_server_url=localhost:50051 \
    actor.model.model_path=thomas0829/folding_towel_pi05 \
    rollout.model.model_path=thomas0829/folding_towel_pi05

python examples/embodiment/train_embodied_agent_staged.py \
    --config-path examples/embodiment/config \
    --config-name yam_ppo_openpi_sync \
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
- GPU 2: VLM planner / TOPReward

### What to look for

The run is wired correctly if you see all of the following:

- a startup log from `train_embodied_agent_staged_async.py` saying it is starting a
  simulated `RobotServer` on `localhost:50051`
- actor, rollout, env, and `VLMPlannerWorker` all start successfully
- `EnvWorker` logs `TOPReward: score=..., delta=...`
- training proceeds past rollout collection into advantage computation and at
  least one policy update

If this works, the following path has been validated inside Beaker:

`RemoteEnv -> local dummy RobotServer -> env output -> TOPReward injection -> GAE -> PPO update`

## Related Docs

- [quickstart](quickstart.md)
- [training_architecture](training_architecture.md)
- [network_infrastructure](network_infrastructure.md)
- [openpi_joint_logprob](openpi_joint_logprob.md)
- [openpi_nan_gradients](openpi_nan_gradients.md)

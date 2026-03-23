# YAM + marl End-to-End Runbook

This is the current end-to-end runbook for the YAM real-world training stack:

- `RLinf` runs actor, rollout, and env workers on Beaker
- `marl` runs as a sidecar service on Beaker GPU 2
- the desktop runs only `RobotServer` (and optional follower servers)
- `RemoteEnv` connects to the desktop through a reverse SSH tunnel on `localhost:50051`

This guide assumes the current repo state on `csr/dev` plus the marl PRs, but
does **not** assume any local `uv` changes inside `RLinf`. When you need the
local `openpi` checkout instead of the default `RLinf/openpi` install path, use
the explicit `uv run ... --with-editable ../openpi` flow described below.

For the high-level network picture, see [network_infrastructure](network_infrastructure.md).
For the older config-specific notes, see [yam_ppo_openpi](yam_ppo_openpi.md) and
[yam_ppo_openpi_topreward](yam_ppo_openpi_topreward.md).

## Repo Layout

The expected multi-repo layout is:

```text
<workspace>/
  RLinf/
  marl/
  openpi/     # needed for local/manual runs when RLinf uv is unchanged
  sglang/     # patched fork required by marl, sibling of marl
```

`marl` expects the patched `sglang` checkout at `../sglang/python` relative to
the `marl` repo. See [marl README](../../marl/README.md).

## Models And Secrets

You need:

- an OpenPI checkpoint for policy inference/training
  - e.g. `thomas0829/folding_towel_pi05`
- a Qwen3-VL checkpoint for `marl`
  - configured through `marl.yaml`
- Beaker secrets:
  - `hf_token_shirui`
  - `tailscale_authkey_shirui`

The Beaker scripts in `RLinf` already assume those secret names.

## Canonical Production Topology

Use this when you want the standard Beaker + desktop setup:

- GPU 0: actor
- GPU 1: rollout
- GPU 2: `marl`
- CPU: `EnvWorker` + `RemoteEnv`
- desktop: `RobotServer` + optional YAM follower servers

The current YAM configs are:

- [yam_ppo_openpi.yaml](../examples/embodiment/config/yam_ppo_openpi.yaml)
  - TOPReward only, no subtask planning
- [yam_ppo_openpi_topreward.yaml](../examples/embodiment/config/yam_ppo_openpi_topreward.yaml)
  - TOPReward + subtask planning

Both configs use the `marl` sidecar. The difference is only whether planner
updates are enabled.

## Recommended Path: Beaker Job + Desktop RobotServer

### 1. Submit the Beaker job

From the `RLinf` repo root:

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi_topreward \
    --model-path thomas0829/folding_towel_pi05 \
    --allow-dirty
```

What this script does today:

- installs `RLinf` deps through `bash requirements/install.sh embodied --model openpi --env remote`
- starts Tailscale and SSH on the Beaker node
- starts Ray
- starts `marl` automatically as a sidecar on GPU 2
- runs [train_embodied_agent_marl.py](../examples/embodiment/train_embodied_agent_marl.py)

Important detail:

- this automatic path uses the `openpi` package installed by `requirements/install.sh`
- with `uv` unchanged, that means `git+https://github.com/RLinf/openpi`
- it does **not** automatically use your local sibling `../openpi` checkout

### 2. Get the Beaker Tailscale address

Watch the Beaker logs for:

```text
=== Tailscale IP ===
100.a.b.c
==================
```

### 3. Start the desktop-side RobotServer

On the robot desktop:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --use-follower-servers \
    --remote-host beaker-0
```

Dummy hardware mode:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --dummy
```

This script:

- resets CAN
- optionally starts follower servers
- starts `RobotServer`
- starts a persistent reverse SSH tunnel with `autossh`

The training job always talks to `localhost:50051` on Beaker. The reverse SSH
tunnel maps that back to the desktop `RobotServer`.

## Manual Path: Keep RLinf uv Unchanged But Use Local openpi

If you need the local sibling `../openpi` checkout without changing
`RLinf/pyproject.toml`, do **not** rely on the default Beaker training command.
Instead:

1. start an idle Beaker cluster or interactive session
2. SSH into the Beaker container
3. run training manually with `uv run`

Example from the `RLinf` repo root:

```bash
export MARL_BASE_URL=http://127.0.0.1:8080
export EMBODIED_PATH=examples/embodiment

uv run --project . --extra embodied \
  --with 'chex==0.1.90' \
  --with-editable ../openpi \
  python examples/embodiment/train_embodied_agent_marl.py \
  --config-name yam_ppo_openpi_topreward \
  actor.model.model_path=thomas0829/folding_towel_pi05 \
  rollout.model.model_path=thomas0829/folding_towel_pi05
```

Why the explicit extras are needed when `uv` is unchanged:

- `--with-editable ../openpi`
  - uses the local `openpi` checkout instead of the default installed package
- `--with 'chex==0.1.90'`
  - keeps `jax/jaxlib` aligned with the current `openpi` path
  - plain `--with chex` can resolve to newer `jax/jaxlib` versions that break
    `orbax-checkpoint` during actor initialization

## Local 5-Step Smoke Test

For local plumbing checks without hardware:

1. start the dummy `marl` server
2. run `train_embodied_agent_marl.py` with dummy remote-desktop simulation
3. force a 5-step rollout and single-GPU placement

Start dummy `marl` from the `marl` repo:

```bash
uv run --project . python -m marl.dummy_server \
  --host 127.0.0.1 \
  --port 18080 \
  --planner-prefix dummy-subtask \
  --reward-mode global_step
```

Then from the `RLinf` repo:

```bash
export MARL_BASE_URL=http://127.0.0.1:18080
export EMBODIED_PATH=examples/embodiment

uv run --project . --extra embodied \
  --with 'chex==0.1.90' \
  --with-editable ../openpi \
  python examples/embodiment/train_embodied_agent_marl.py \
  --config-name yam_ppo_openpi_topreward \
  runner.max_epochs=1 \
  runner.logger.log_path=/tmp/rlinf-smoke-5step \
  actor.model.model_path=/path/to/folding_towel_pi05 \
  rollout.model.model_path=/path/to/folding_towel_pi05 \
  env.remote_desktop_simulation.enabled=true \
  env.remote_desktop_simulation.dummy=true \
  env.remote_desktop_simulation.env_config_path=examples/embodiment/config/env/yam_pi05_follower_dummy.yaml \
  env.train.max_steps_per_rollout_epoch=50 \
  env.eval.max_steps_per_rollout_epoch=50 \
  env.train.subtask_interval=1 \
  marl.planner.interval=1 \
  cluster.component_placement.actor.placement=0 \
  cluster.component_placement.rollout.placement=0 \
  cluster.component_placement.env.placement=0 \
  actor.micro_batch_size=5 \
  actor.global_batch_size=5
```

This smoke test validates:

- `obs -> marl /image-sets`
- `obs -> marl /topreward`
- `obs -> marl /plan`
- `subtask -> next VLA input`
- `action -> RemoteEnv -> RobotServer -> YAMEnv`

It does **not** guarantee numerically healthy actor updates; the current actor
training path can still hit non-finite gradients after rollout completes.

## Current Caveats

- The current `quickstart.md`, `yam_ppo_openpi.md`, and
  `yam_ppo_openpi_topreward.md` still contain some older `VLMPlannerWorker` /
  `train_embodied_agent_staged.py` wording.
- The canonical runtime path is now:
  - [train_embodied_agent_marl.py](../examples/embodiment/train_embodied_agent_marl.py)
  - [submit_yam_training.sh](../scripts/submit_yam_training.sh)
  - [start_robot_server.sh](../scripts/start_robot_server.sh)
- The current known training blocker after rollout is actor-side non-finite
  gradients in the OpenPI training path. That is separate from the `marl`
  transport and runbook flow documented here.

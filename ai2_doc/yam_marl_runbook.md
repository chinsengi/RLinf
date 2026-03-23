# YAM + marl Runbook

This document describes the exact setup for the current YAM + `marl` stack.

Use this setup if you want all of these at once:

- `RLinf` on Beaker
- `marl` as a Beaker sidecar
- desktop `RobotServer` over reverse SSH
- local forked `openpi`
- local forked `sglang`

This runbook assumes `RLinf`'s `uv` config is unchanged.

## 1. Repo Layout

The required sibling layout is:

```text
<root>/
  RLinf/
  marl/
  openpi/
  sglang/
```

Clone exactly these repos:

```bash
git clone https://github.com/xslingcn/marl
git clone -b marl https://github.com/xslingcn/openpi
git clone -b marl https://github.com/xslingcn/sglang
```

Use exactly these sources:

- `marl`: `https://github.com/xslingcn/marl`
- `openpi`: `https://github.com/xslingcn/openpi/tree/marl`
- `sglang`: `https://github.com/xslingcn/sglang/tree/marl`

Why the sibling layout matters:

- `RLinf` manual training command uses `--with-editable ../openpi`
- `marl` expects the patched `sglang` checkout at `../sglang/python`
- `submit_yam_training.sh` defaults `MARL_REPO_DIR` to the sibling repo `../marl`

## 2. Models And Secrets

You need:

- an OpenPI checkpoint `thomas0829/folding_towel_pi05`
- a Qwen3-VL checkpoint for `marl`
  - configured in `marl/marl.yaml`
- Beaker secrets:
  - `hf_token_shirui`
  - `tailscale_authkey_shirui`

## 3. Use Interactive Beaker Mode

Do **not** use the default non-interactive training path if you want this exact
fork-based stack.

Reason:

- `submit_yam_training.sh` in normal mode installs `openpi` through
  `requirements/install.sh`
- with unchanged `uv`, that means `git+https://github.com/RLinf/openpi`
- it does not use your sibling `../openpi` checkout

For the exact setup in this document, start an interactive Beaker session:

```bash
cd RLinf

bash scripts/submit_yam_training.sh \
  --config yam_ppo_openpi_topreward \
  --interactive \
  --allow-dirty
```

Then attach to the session:

```bash
beaker session attach <session-id>
```

Inside the container, the expected repo layout is still sibling-based, for
example:

```text
/weka/oe-training-default/shiruic/
  RLinf/
  marl/
  openpi/
  sglang/
```

## 4. Start the Desktop RobotServer

On the desktop:

```bash
cd RLinf

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

This starts:

- follower servers, if requested
- `RobotServer`
- a persistent reverse SSH tunnel to Beaker

`RLinf` always connects to `localhost:50051` on Beaker. The tunnel maps that
back to the desktop `RobotServer`.

## 5. Start marl on Beaker

Inside the attached Beaker shell:

```bash
cd /weka/oe-training-default/shiruic/RLinf

export MARL_REPO_DIR=/weka/oe-training-default/shiruic/marl
export MARL_CONFIG_PATH=${MARL_REPO_DIR}/marl.yaml
export MARL_BASE_URL=http://127.0.0.1:8080

nohup env CUDA_VISIBLE_DEVICES=2 \
  uv run --project ${MARL_REPO_DIR} --python 3.12 \
  python -m marl --config ${MARL_CONFIG_PATH} --log-level info \
  > ${MARL_REPO_DIR}/marl_server.log 2>&1 &

curl -fsS ${MARL_BASE_URL}/healthz
```

## 6. Start Training on Beaker

Still inside the attached Beaker shell:

```bash
cd /weka/oe-training-default/shiruic/RLinf

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

If you want the reward-only variant, switch the config:

```bash
--config-name yam_ppo_openpi
```

Why these flags are required with unchanged `uv`:

- `--with-editable ../openpi`
  - forces the sibling `openpi` fork
- `--with 'chex==0.1.90'`
  - keeps `jax/jaxlib` aligned with the current `openpi` path
  - plain `--with chex` can resolve to a newer `jax/jaxlib` pair that breaks
    `orbax-checkpoint` during actor initialization

## 7. Actual Runtime Topology

This exact setup gives:

- GPU 0: actor
- GPU 1: rollout
- GPU 2: `marl`
- CPU: `EnvWorker` + `RemoteEnv`
- desktop: `RobotServer` + optional follower servers

The runtime path is:

```text
OpenPI rollout -> EnvWorker -> RemoteEnv -> RobotServer -> YAMEnv
                                 |
                                 +-> marl /image-sets
                                 +-> marl /topreward
                                 +-> marl /plan
```

For `yam_ppo_openpi_topreward`:

- `/topreward` is used for dense reward
- `/plan` is used for subtask planning

For `yam_ppo_openpi`:

- `/topreward` is used
- `/plan` is not used

## 8. Local 5-Step Smoke Test

Use this only for pipeline validation, not real training.

Start dummy `marl`:

```bash
cd marl

uv run --project . python -m marl.dummy_server \
  --host 127.0.0.1 \
  --port 18080 \
  --planner-prefix dummy-subtask \
  --reward-mode global_step
```

Then from `RLinf`:

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

This validates:

- `obs -> marl /image-sets`
- `obs -> marl /topreward`
- `obs -> marl /plan`
- `subtask -> next VLA input`
- `action -> RemoteEnv -> RobotServer -> YAMEnv`

It does not guarantee numerically healthy actor updates.

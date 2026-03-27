# YAM Training Quickstart

All Ray workers run on Beaker; the desktop runs only the gRPC `RobotServer`
exposed via a reverse SSH tunnel. This is the only working topology for standard
YAM experiments.

For network and infrastructure details, see [network_infrastructure](network_infrastructure.md).
For algorithm and implementation details, see [training_architecture](training_architecture.md).
For config-specific guides, see [yam_ppo_openpi](yam_ppo_openpi.md) and
[yam_ppo_openpi_topreward](yam_ppo_openpi_topreward.md).

## Prerequisites

- [x] `autossh` installed on the desktop (`brew install autossh` on macOS,
  `sudo apt-get install autossh` on Ubuntu/Debian)
- [x] Desktop has a Tailscale client connected to the AI2 network
- [x] Beaker secrets written (see [Beaker Secrets](#beaker-secrets) below)
- [x] Model checkpoint available (HuggingFace ID or local path; default: `thomas0829/folding_towel_pi05`)

## Beaker Secrets

The following secrets must exist in the Beaker workspace:

| Secret Name | Purpose |
|---|---|
| `hf_token_shirui` | HuggingFace token for model downloads |
| `SHIRUI_TAILSCALE_KEY` | Tailscale auth key for container VPN setup |

Create them with:

```bash
beaker secret write hf_token_shirui "hf_..."
beaker secret write SHIRUI_TAILSCALE_KEY "tskey-auth-..."
```

Generate a Tailscale auth key at: Tailscale admin console > Settings > Keys >
Generate auth key. Use a **reusable** key if running multiple jobs.

## End-to-End Workflow

### Step 1: Start the interactive Beaker session

Start a Beaker interactive session that brings up Tailscale, installs
dependencies, starts the Ray head node, and leaves you with a shell-ready
session. Training is not submitted yet.

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --interactive --allow-dirty
```

To specify a model checkpoint and/or task description:

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --model-path thomas0829/folding_towel_pi05 \
    --task "Fold the towel." \
    --interactive --allow-dirty
```

To run inference only (no weight updates), set `algorithm.lr=0`:

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --model-path thomas0829/folding_towel_pi05 \
    --task "Fold the towel." \
    --interactive --allow-dirty \
    -- algorithm.lr=0
```

Extra Hydra overrides can be passed after `--`:

```bash
bash scripts/submit_yam_training.sh \
    --config yam_ppo_openpi \
    --interactive --allow-dirty \
    -- algorithm.update_epoch=2
```

Beaker prints a session ID. Keep it for Step 4, where you will attach and start
training manually. Pass `--workspace <beaker-workspace>` if you want to submit
outside the default workspace. This keeps server startup and training startup
decoupled.

If you have access to non-interactive Beaker jobs, you can also start an idle
cluster instead:

```bash
bash scripts/submit_yam_beaker_cluster.sh \
    --config yam_ppo_openpi \
    --allow-dirty
```

That path starts Ray and waits without creating an interactive Beaker session.
In that case, use direct SSH in Step 4 instead of `beaker session attach`.

### Step 2: Get the container's Tailscale IP

Watch the Beaker logs for:

```
=== Tailscale IP ===
100.a.b.c
==================
```

### Step 3: Start the robot server with persistent reverse SSH tunnel

```bash
# Real hardware — tunnel reconnects automatically when new Beaker jobs start
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --use-follower-servers

# Dummy mode (no CAN bus / robot hardware needed — for pipeline testing)
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --dummy
```

Add `--verbose` to inspect robot joint states before serving and log every
chunk step action during execution. The first chunk will be paused until you
approve it by running `touch /tmp/rlinf_approve_chunk` in another terminal:

```bash
bash scripts/start_robot_server.sh \
    --config examples/embodiment/config/env/yam_pi05_follower.yaml \
    --use-follower-servers --verbose
```

The server stays running indefinitely. `autossh` reconnects the reverse tunnel
to each new Beaker job automatically (all jobs register `beaker-0`). You do not
need to restart the robot server between Beaker job submissions.

> **Note:** `autossh` must be installed on the desktop. The script prints
> install instructions if it is missing.

### Step 4: Attach and start training manually

Once the interactive session is up and the robot server tunnel is running,
attach to the Beaker session and launch training manually:

```bash
beaker session attach <session-id>
cd /weka/oe-training-default/shiruic/RLinf
source .venv/bin/activate
python examples/embodiment/train_embodied_agent_staged.py \
    --config-name yam_ppo_openpi \
    actor.model.model_path=thomas0829/folding_towel_pi05 \
    rollout.model.model_path=thomas0829/folding_towel_pi05 \
    'env.train.task_description=Fold the towel.' \
    'env.eval.task_description=Fold the towel.'
```

If the run fails or you want to tweak Hydra overrides, re-run the training
command from the same Beaker session. You do not need to create a new session
unless the session itself exits.

If you started the idle cluster with `submit_yam_beaker_cluster.sh` instead,
SSH into the container and run the same command there:

```bash
ssh shiruic@beaker-0  # or ssh shiruic@<tailscale-ip>
cd /weka/oe-training-default/shiruic/RLinf
source .venv/bin/activate
python examples/embodiment/train_embodied_agent_staged.py \
    --config-name yam_ppo_openpi \
    actor.model.model_path=thomas0829/folding_towel_pi05 \
    rollout.model.model_path=thomas0829/folding_towel_pi05 \
    'env.train.task_description=Fold the towel.' \
    'env.eval.task_description=Fold the towel.'
```

The `RemoteEnv` inside the container connects to `localhost:50051` (routed
through the SSH tunnel to the desktop's `RobotServer`). Actor runs on GPU 0,
Rollout on GPU 1, VLMPlannerWorker on GPU 2. The training loop proceeds:

```
Rollout (GPU 1) ─── generates actions ──────► RemoteEnv ─── gRPC ───► RobotServer
     ▲                                             │                        │
     │ updated weights                             │                    YAMEnv
     │                                             │                    (robot HW)
Actor (GPU 0) ◄──── trajectories + rewards ◄──────┘                        │
     └──── updates weights ─────────────────────► Rollout ◄─ observations ─┘

VLMPlanner (GPU 2) ◄── frames + instruction ── EnvWorker ──────────────────┘
     │   (TOPReward delta injected into rewards; subtasks injected if interval > 0)
     └──────────────────────────────────────────────────────────────────────────►
```

> **Reward note:** Both YAM configs use TOPReward (Qwen3-VL-8B on GPU 2) —
> no custom reward code required. The only difference is `subtask_interval`:
> `yam_ppo_openpi` scores reward only; `yam_ppo_openpi_topreward` also
> generates VLM subtask descriptions injected into the policy's language conditioning.

## Supported Configs

| Config | Reward | Subtask Planning | Startup Command |
|---|---|---|---|
| `yam_ppo_openpi` | TOPReward (dense, VLM-based) | no (`subtask_interval: 0`) | `submit_yam_training.sh --interactive` or `submit_yam_beaker_cluster.sh` |
| `yam_ppo_openpi_topreward` | TOPReward (dense, VLM-based) | yes (`subtask_interval: 3`) | `submit_yam_training.sh --interactive` or `submit_yam_beaker_cluster.sh` |

Both configs use the same decoupled startup flow. Both startup scripts
auto-detect the GPU count from the config name.

## Next Steps

- [Network & infrastructure details](network_infrastructure.md) — Tailscale
  setup, SSH tunnel mechanics, CAN bus, scripts reference, and troubleshooting
- [Training architecture](training_architecture.md) — data flow, tensor shapes,
  PPO/GAE internals, Hydra config reference, and implementation notes
- [YAM PPO + TOPReward config guide](yam_ppo_openpi.md) — includes a Beaker
  simulated-robot-input validation workflow
- [YAM PPO + TOPReward + subtask planning guide](yam_ppo_openpi_topreward.md)
  — includes the staged Beaker simulated-robot-input validation workflow

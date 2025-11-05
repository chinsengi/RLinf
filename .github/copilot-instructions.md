# RLinf AI Coding Agent Instructions

## Project Overview

RLinf is a **distributed reinforcement learning infrastructure** for post-training foundation models (LLMs and VLAs). It supports three main domains:
- **Embodied Intelligence**: RL training for Vision-Language-Action models (π₀, π₀.₅, OpenVLA) in simulators (ManiSkill, LIBERO, BEHAVIOR)
- **Reasoning**: RL fine-tuning for LLM reasoning (math, AIME benchmarks)
- **Agentic RL**: Online RL for coding agents

The core architecture follows a **macro-to-micro flow transformation** pattern with Ray-based distributed workers.

## Architecture & Design Patterns

### Worker-Based Distributed System
All computation happens in specialized **Worker classes** orchestrated by **Runner classes**:
- `ActorWorker` (training), `RolloutWorker` (inference), `RewardWorker`, `EnvWorker` (embodied), `InferenceWorker`
- Workers launch as Ray actors across GPU nodes using **placement strategies**
- Communication via Channel API (`rlinf/scheduler/channel/`), not through main process
- Example: `EmbodiedRunner` coordinates `ActorGroup`, `RolloutGroup`, `EnvGroup`

### Placement Modes (Critical for Scaling)
Located in `rlinf/utils/placement.py`:
- **HYBRID**: Rollout+Actor on same GPUs (embodied tasks only)
- **COLLOCATED**: All components share GPUs (memory-constrained setups)
- **DISAGGREGATED**: Separate GPU pools per component (production scale)
- **AUTO**: Dynamic scheduling via `SchedulerWorker`

Parse from `cluster.component_placement` config (e.g., `actor,inference: 0-4` assigns both to GPUs 0-4).

### Configuration System (Hydra-Based)
- **All configs are Hydra/OmegaConf** (`@hydra.main` decorator on entry points)
- Main config validation: `rlinf/config.py:validate_cfg()` - call this mentally when reasoning about configs
- Task-specific validators: `validate_embodied_cfg()`, `validate_reasoning_cfg()`, `validate_coding_online_rl_cfg()`
- Model configs auto-validated against HuggingFace configs via `validate_model_cfg_by_hf_config()`
- Config structure: `cluster`, `runner`, `algorithm`, `actor`, `rollout`, `data`, `reward`, `critic`

### Dual Training Backends
- **FSDP** (`rlinf/workers/actor/fsdp_actor_worker.py`): Rapid prototyping, HuggingFace-compatible
- **Megatron-LM** (`rlinf/workers/actor/megatron_actor_worker.py`): Large-scale training, requires `/opt/Megatron-LM` in PYTHONPATH

Backend selection via `actor.training_backend` config. Megatron requires checkpoint conversion (see `actor.megatron.ckpt_convertor` config).

### Dual Rollout Backends
- **SGLang** (default, `rollout.sglang` config)
- **vLLM** (`rollout.vllm` config)

Backend selection via `rollout.rollout_backend`. Both share `rollout.sampling_params` config structure.

## Critical Development Workflows

### Running Experiments
1. **Embodied tasks**: `bash examples/embodiment/run_embodiment.sh [config_name]`
   - Requires `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`
   - LIBERO requires `LIBERO_REPO_PATH=/opt/libero` and `export PYTHONPATH`
   - BEHAVIOR requires OmniGibson environment variables (see `run_embodiment.sh`)

2. **Reasoning tasks**: `bash examples/reasoning/run_main_grpo_math.sh [config_name]`
   - Requires Megatron-LM in PYTHONPATH: `export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:$PYTHONPATH`

3. **Coding agents**: `bash examples/coding_online_rl/run_main_coding_online_rl.sh`

### Environment Setup
- **For embodied**: `bash requirements/install.sh openvla|openvla-oft|openpi [--enable-behavior]`
- **For reasoning**: `bash requirements/install.sh reason`
- Use provided Docker images for production: `docker/torch-2.6/Dockerfile`
- Python 3.11 (3.10 for BEHAVIOR tasks)

### Model Checkpointing
- **HF → Megatron conversion** handled by `ckpt_convertor` config when `use_hf_ckpt: True`
- Conversion runs automatically before training if checkpoint doesn't exist
- Use `toolkits/ckpt_convertor/` for manual conversion

### Testing Strategy
- **Unit tests**: `tests/unit_tests/` (pytest-based, test workers, channels, collectives)
- **E2E tests**: `tests/e2e_tests/` split by domain (embodied, reasoning, agent)
- CI/CD: GitHub Actions workflows in `.github/workflows/` (check `ci-tests.yml`)

## Code Style & Patterns

### Configuration Access
```python
# Always validate configs before use
cfg = validate_cfg(cfg)

# Access with get() for optional params
cfg.algorithm.training_batch_size_per_gpu = cfg.algorithm.get("training_batch_size_per_gpu", 1)

# Use OmegaConf for modification
from omegaconf import open_dict
with open_dict(cfg):
    cfg.new_field = value
```

### Worker Implementation
```python
from rlinf.scheduler.worker import Worker, WorkerGroup

class MyWorker(Worker):
    @classmethod
    def create_group(cls, cfg, component_placement=None) -> WorkerGroup:
        world_size = component_placement.get_world_size("my_component")
        return WorkerGroup(worker_cls=cls, world_size=world_size, ...)
```

### Ray Multiprocessing
All entry points use: `mp.set_start_method("spawn", force=True)` before Hydra decorator.

### Logging
- Primary: `rlinf/utils/logger.py` (TensorBoard/WandB/SwanLab backends)
- Config: `runner.logger.logger_backends`, `runner.logger.log_path`

## Common Pitfalls

1. **GPU Placement**: Always check `cluster.component_placement` config. Mismatched placement causes OOM or underutilization.

2. **Sequence Length**: `runner.seq_length` must be > `data.max_prompt_length`. Validated in `validate_reasoning_cfg()`.

3. **Megatron PYTHONPATH**: Megatron requires `/opt/Megatron-LM` in PYTHONPATH. Missing this causes import errors.

4. **Checkpoint Paths**: Use absolute paths for `rollout.model_dir` and `actor.checkpoint_load_path`. Relative paths break in distributed setting.

5. **Task Type Mismatch**: `runner.task_type` must match actual task (embodied/reasoning/coding_online_rl). Drives different validation logic.

6. **Tokenizer Padding**: `actor.tokenizer.padding_side` should be `'right'` for training. Left padding breaks loss computation.

## Key Files to Reference

- **Config validation**: `rlinf/config.py` (understand all validators)
- **Placement logic**: `rlinf/utils/placement.py` (PlacementMode, ComponentPlacement)
- **Algorithm implementations**: `rlinf/algorithms/` (advantages.py, losses.py)
- **Runner orchestration**: `rlinf/runners/` (embodied_runner.py, reasoning_runner.py)
- **Example configs**: `examples/{embodiment,reasoning,coding_online_rl}/config/`

## Contributing Standards

- **Commit format**: Conventional Commits (`feat(embodied):`, `fix(scheduler):`, etc.)
- **Pre-commit hooks**: Run `pre-commit run --all-files` before pushing
- **Sign-off required**: Use `-s` flag on commits
- **CI must pass**: Unit tests, E2E tests, linting (ruff), and commit format checks

## Useful Commands

```bash
# Format code
pre-commit run --all-files

# Run unit tests
pytest tests/unit_tests/test_worker.py -v

# Build docs
cd docs && bash autobuild.sh  # English
cd docs && bash autobuild.sh zh  # Chinese

# Check Ray cluster
bash ray_utils/check_ray.sh
```

## Model-Specific Notes

- **OpenVLA**: Requires `requirements/openvla.txt`, uses DinoV2 vision encoder
- **OpenVLA-OFT**: Output-fine-tuned variant, requires `requirements/openvla_oft.txt`
- **π₀/π₀.₅**: Flow-matching action experts, LoRA support available
- **Qwen2.5**: Math reasoning, uses Megatron backend

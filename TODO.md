# TODO

Current discovered problems to track:

## Open issues

### Investigate offline RL updates during cooldown

Context:
- `RemoteEnv`/RobotServer own the restart and cooldown cycle, and cooldown chunks are now marked as collection pauses instead of training data.
- That means the actor can spend cooldown wall-clock on replay-buffer updates from already collected trajectories, as long as those updates do not consume cooldown/no-op chunks.
- RLinf already has an off-policy replay path for real-world training in the SAC/RLPD configs and workers (`rlinf/workers/actor/fsdp_sac_policy_worker.py`, `rlinf/workers/actor/async_fsdp_sac_policy_worker.py`, `examples/embodiment/config/realworld_*_rlpd_*.yaml`).
- The current YAM setup is still PPO/OpenPI-centric (`examples/embodiment/config/yam_ppo_openpi_subtask_async.yaml`), and the shipped policy is a `π₀.5` / OpenPI diffusion VLA with a PPO value head (`ai2_doc/training_architecture.md`, `ai2_doc/openpi_flowppo.md`).
- That matters: methods built around a separate Gaussian actor + Q critic (`SAC`, `TD3+BC`, `AWAC`, plain `RLPD`) are not drop-in for the deployed policy architecture. They are still useful algorithmic references, but not the easiest direct implementation path.
- In particular, `SAC`-style cooldown training is extra work here because it needs dedicated critic machinery that the current PPO/OpenPI path does not maintain: twin Q functions, target networks, entropy-temperature tuning, and a SAC-compatible actor objective. That is substantially more invasive than adding an offline objective on top of the existing PPO value-head setup.
- So there are really two viable directions:
  1. reuse the existing SAC/RLPD replay learner during cooldown
  2. add an offline learner that matches the current OpenPI / diffusion-VLA PPO path

Summary of literature search:
- `RLPD` ("Efficient Online Reinforcement Learning with Offline Data", Ball et al., 2023, OpenReview: https://openreview.net/forum?id=h11j9w1ucU)
  Best conceptual match for this use case. It is explicitly about improving online RL with prior/offline data, not pure static-dataset training. The main ideas are balanced replay between offline and online data, critic stabilization, and large Q ensembles. This is the closest external validation of "do extra replay-based updates while new interaction is temporarily unavailable."
- `AWAC` ("AWAC: Accelerating Online Reinforcement Learning with Offline Datasets", Nair et al., 2020, arXiv: https://arxiv.org/abs/2006.09359)
  Strong robotics precedent. Advantage-weighted behavioral cloning on top of an off-policy critic makes it practical for leveraging demonstrations and then improving online. Good candidate if we want a simpler actor update than full conservative offline RL.
- `IQL` ("Offline Reinforcement Learning with Implicit Q-Learning", Kostrikov et al., 2021, arXiv: https://arxiv.org/abs/2110.06169)
  One of the strongest general-purpose offline-to-online baselines for continuous control. Attractive because it avoids querying OOD actions during critic fitting. If the data distribution is narrow and robot safety matters, this is a very credible first offline method to test.
- `TD3+BC` ("A Minimalist Approach to Offline Reinforcement Learning", Fujimoto and Gu, 2021, arXiv: https://arxiv.org/abs/2106.06860)
  Lowest-complexity offline baseline. Very useful as the first "does cooldown training help at all?" experiment. If this fails, more complex offline RL probably will not justify itself.
- `CQL` ("Conservative Q-Learning for Offline Reinforcement Learning", Kumar et al., 2020, arXiv: https://arxiv.org/abs/2006.04779)
  Strong conservative baseline when overestimation is the main concern. Better fit for narrow, biased, or safety-critical robot data than plain SAC/TD3, but more pessimistic and more tuning-heavy.
- `Cal-QL` ("Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning", Nakamoto et al., 2023, arXiv: https://arxiv.org/abs/2303.05479)
  Particularly relevant if we want offline pretraining that still fine-tunes well online afterward. It is effectively a fine-tuning-oriented refinement of CQL.
- `BPPO` ("Behavior Proximal Policy Optimization", Zhuang et al., 2023, arXiv: https://arxiv.org/abs/2302.11312)
  Most relevant paper if we want to stay in the PPO family. It argues PPO-style conservatism can work for offline RL without extra constraints. This is the cleanest research lead for a PPO/OpenPI path, but it is less established in robotics practice than AWAC/IQL/RLPD.
- `Diffusion-QL` ("Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning", Wang et al., 2022, arXiv: https://arxiv.org/abs/2208.06193)
  Especially relevant because `π₀.5` / OpenPI is a diffusion-style action model. The key idea is attractive here: keep a behavior-cloning-style diffusion loss and add value-guided improvement, rather than replacing the actor with a conventional SAC/TD3 parameterization.
- `EDP` ("Efficient Diffusion Policies for Offline Reinforcement Learning", Kang et al., 2023, arXiv: https://arxiv.org/abs/2305.20081)
  Also highly relevant to the current policy class. It was designed to make diffusion-policy offline RL practical and explicitly supports coupling diffusion actors with algorithms such as `IQL` and `CRR`. If we want an offline method that still looks like a diffusion VLA instead of a separate control head, this is one of the best leads.
- `CRR` ("Critic Regularized Regression", Wang et al., 2020, arXiv: https://arxiv.org/abs/2006.15134)
  Another practical actor-critic option with a regression-style actor update. Worth considering if AWAC is unstable or overly sensitive to advantage scaling.
- `BCQ` ("Off-Policy Deep Reinforcement Learning without Exploration", Fujimoto et al., 2018, arXiv: https://arxiv.org/abs/1812.02900)
  Historically important and explicitly support-constrained, but heavier than TD3+BC/IQL and less attractive than newer baselines for this project.
- `Decision Transformer` ("Decision Transformer: Reinforcement Learning via Sequence Modeling", Chen et al., 2021, arXiv: https://arxiv.org/abs/2106.01345) and `RvS` ("What is Essential for Offline RL via Supervised Learning?", Emmons et al., 2021, arXiv: https://arxiv.org/abs/2112.10751)
  Good baselines if we want to treat the problem more like conditional imitation or sequence modeling. They are less compelling than IQL/AWAC/RLPD here because the current stack already has replay/Q-learning infrastructure and these methods do not directly exploit the existing actor/critic code.
- `MOPO` ("MOPO: Model-based Offline Policy Optimization", Yu et al., 2020, arXiv: https://arxiv.org/abs/2005.13239) and `COMBO` ("Conservative Offline Model-Based Policy Optimization", Yu et al., 2021, arXiv: https://arxiv.org/abs/2102.08363)
  Model-based offline RL can help when data are very scarce, but this is probably the wrong first step here. Learning a reliable world model for image-conditioned robot behavior with TOPReward/subtask-dependent rewards is much higher risk than model-free replay updates.

Implementation plan:
- **Phase 1 — BPPO baseline (smallest delta from current PPO path):**
  1. Add a cooldown replay buffer that retains recent non-cooldown rollout batches (with `forward_inputs`, `prev_logprobs`, `prev_values`, `versions`) instead of discarding them after the PPO update. Reuse or extend the existing `TrajectoryReplayBuffer` (`rlinf/data/replay_buffer.py`).
  2. In the actor worker, when `_has_trainable_rollout_batch()` returns `False` (cooldown), sample from the replay buffer and run a BPPO-style update: recompute current-policy logprobs via `get_log_prob_value()` from saved `chains`/`denoise_inds`, compute importance ratio against saved `prev_logprobs`, apply clipped surrogate loss with a tighter clip than online PPO.
  3. Add a budget mechanism: fixed number of minibatches per cooldown period (configurable in YAML), plus a staleness cutoff that drops replay entries older than N policy versions (using the `versions` field).
  4. Add reward-version tagging: stamp each replay entry with the TOPReward/subtask-plan version so that stale-reward trajectories can be filtered or reweighted.
  5. Validate: compare cooldown-BPPO vs. idle-cooldown on a YAM subtask config. Primary metric: sample efficiency (episodes to convergence).

- **Phase 2 — Diffusion-QL / EDP (diffusion-native offline objective):**
  1. Implement a Q-network that conditions on denoising timestep `t` in addition to (state, image, action). Start from the existing `CompactMultiQHead` in DSRL mode (`openpi_action_model.py:1241`), adding a timestep embedding input.
  2. Implement post-hoc trajectory reweighting: during cooldown replay updates, compute Q-values at saved chain points and apply `loss = Σ_t [log π(chain_t | s) − α · Q(s, chain_t)]` as the actor objective.
  3. Train the Q-network on replay data using standard Bellman backup with the clipped double-Q trick (reuse DSRL's twin-Q infrastructure).
  4. Compare against Phase 1 BPPO baseline on the same YAM config.

- **Phase 3 (optional) — IQL/CRR as critic components:**
  - Only if Phase 2 shows Q-overestimation issues on narrow cooldown data.
  - Swap the Bellman-backup critic with IQL's expectile regression or CRR's advantage-weighted objective, keeping the diffusion actor and post-hoc reweighting loss from Phase 2.

Key files to modify:
- `rlinf/workers/actor/async_ppo_fsdp_worker.py` — cooldown replay loop, BPPO loss
- `rlinf/workers/actor/fsdp_actor_worker.py` — replay buffer integration, budget mechanism
- `rlinf/models/embodiment/openpi/openpi_action_model.py` — timestep-conditioned Q-head (Phase 2)
- `rlinf/data/replay_buffer.py` — reward-version tagging, staleness filtering
- `rlinf/data/embodied_io_struct.py` — reward-version field if not already present
- `examples/embodiment/config/` — new YAML config for cooldown-RL experiments

Recommendation:
- For the actual shipped `π₀.5` / OpenPI VLA, the best-fit direct path is no longer generic `RLPD`. The first methods worth prioritizing are the ones that preserve a PPO/diffusion-style actor:
  1. `BPPO` if we want the smallest conceptual change from the current PPO value-head training
  2. `Diffusion-QL` or `EDP` if we want a genuinely offline RL objective that still respects the diffusion actor parameterization
  3. `IQL` or `CRR` only if they are used as critic/objective components inside a diffusion-policy training recipe, not as a naive actor swap
- `RLPD` is still a strong systems-level reference for how to use idle wall-clock on replay updates, but for `π₀.5` it is better viewed as inspiration for scheduling and replay usage than as the exact algorithm to implement.
- `SAC`/`RLPD` should therefore be treated as a second-stage option only if we explicitly decide to pay the cost of a separate replay learner with twin-Q critics and target networks.
- If we are willing to train a separate auxiliary replay policy instead of directly updating the VLA, then the existing SAC/RLPD path becomes viable again. But that would be a different deployment story: either distill back into OpenPI later or switch the deployed actor class.

Repo-specific caveats:
- TOPReward and VLM subtask planning make reward semantics drift over time. Offline updates on stale trajectories may need reward versioning or reward recomputation, otherwise the actor trains against inconsistent targets.
- Cooldown/no-op chunks must remain excluded from any offline learner.
- The cooldown learner needs a strict budget, for example a fixed number of minibatches or a wall-clock cap per cooldown, otherwise it can overfit stale data.
- Because `π₀.5` is a VLA/diffusion policy, offline RL methods that assume tractable action log-probs or one-shot Gaussian action heads may not map cleanly onto the current OpenPI forward path.
- The docs explicitly say the current YAM configs run standard PPO on OpenPI and that `Diffusion NFT` is still a planned upgrade (`ai2_doc/training_architecture.md`). So any offline RL method should be judged partly by whether it can coexist with the current PPO-style logprob recomputation path.
- A SAC-style branch would also duplicate optimization state and model structure: PPO currently has a value head, while SAC/RLPD would introduce twin Q heads, target critics, and alpha tuning. That increases code surface, checkpoint complexity, and the risk of maintaining two incompatible training paths for the same deployed policy.
- If the deployment policy remains PPO/OpenPI, we need to verify whether we can add an offline critic cleanly, whether the current saved `forward_inputs` are sufficient for offline policy improvement, and whether behavior-policy snapshots are needed for BPPO-style updates.
- `TODO(agent)`: verify whether the current OpenPI/YAM actor exposes enough saved behavior-policy information for BPPO-style updates, or whether we would need an explicit behavior policy model / BC warm start.
  - **Verdict: sufficient.** The actor already saves `prev_logprobs` ([B, action_dim]), `prev_values` ([B, 1]), model `versions` ([B, 1]), and the full denoising trajectory in `forward_inputs` (`chains` [B, num_steps+1, action_horizon, action_dim], `denoise_inds` [B, num_steps]) along with all observation tensors. The `get_log_prob_value()` method (`openpi_action_model.py:987`) can recompute logprobs from saved chains, and `compute_proximal_logprobs()` (`async_ppo_fsdp_worker.py:101`) already demonstrates multi-version ratio computation. No explicit behavior policy model or BC warm start is needed.
  - **What IS missing for BPPO:** (1) off-policy importance-ratio clipping in the loss function, (2) a code path that retains cooldown-period rollout data for replay instead of discarding it in `_has_trainable_rollout_batch()`, (3) a budget mechanism (max minibatches or wall-clock cap) to prevent overfitting on stale cooldown replay data.

- `TODO(agent)`: verify whether a diffusion-policy offline objective can reuse the existing OpenPI `forward_inputs` / denoising state instead of introducing a second policy implementation just for cooldown training.
  - **Verdict: yes, no second policy needed.** The saved `forward_inputs` already contain the full denoising trajectory (`chains`), step indices (`denoise_inds`), and all observation context. Per-step log-probs are computed via Gaussian transition densities (`get_logprob_norm`, `openpi_action_model.py:966`). The DSRL mode already has a Q-head (`sac_q_forward`, `openpi_action_model.py:1241`) with `CompactMultiQHead` that takes (state, image, action) — this can be extended for Diffusion-QL/EDP.
  - **Two viable implementation paths:**
    1. **Post-hoc trajectory reweighting** (simpler, recommended first): access `forward_inputs["chains"]` during training, compute Q-values at chain points, apply loss `log π(chain_t) − α·Q(s, chain_t)`. No changes to rollout or denoising code.
    2. **Q-guided denoising** (more expressive): modify `sample_mean_var_val()` to accept a Q-network and shift the denoising mean/variance toward high-Q regions during sampling.
  - **Caveats:** (1) the DSRL Q-head is trained on noise actions, not trajectory points — it needs retraining or a parallel Q-net on trajectory data; (2) the Q-network currently lacks timestep conditioning (needed for proper Diffusion-QL); (3) start with post-hoc reweighting to avoid destabilizing the denoising loop.

Background survey:
- `Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems` (Levine et al., 2020, arXiv: https://arxiv.org/abs/2005.01643)
  Useful overview of the core failure mode: distribution shift causes Q-value overestimation on unsupported actions, so the main design question is how conservative we need the cooldown learner to be.

Recently resolved:
- Remote YAM cooldown chunks now emit an explicit collection-pause signal, and `EnvWorker` skips them when building training trajectories.
- `VLMPlannerClient` now blocks adaptive subtask updates across terminal episode boundaries until pending TOPReward state has been resolved and reset.

## Follow-up cleanup

### Remove temporary actor-side rollout-batch validation

Context:
- Temporary rollout-batch validation was added in the actor workers while debugging the cooldown / collection-pause path.
- Once the cooldown path is validated end to end, that defensive validation should be removed so the actor code returns to the minimal guard needed for empty cooldown batches.

Plan:
- **Keep** `_has_trainable_rollout_batch()` (`fsdp_actor_worker.py:1147-1162`) — this is the minimal guard that skips processing for empty cooldown batches and is still needed.
- **Remove** `_validate_trainable_rollout_batch()` (`fsdp_actor_worker.py:1164-1199`) — this is the temporary defensive validation (shape checks on rewards, prev_logprobs, dones, forward_inputs) added for debugging.
- **Remove all call sites** of `_validate_trainable_rollout_batch()`:
  - `async_ppo_fsdp_worker.py:67` (in `compute_advantages_and_returns`)
  - `async_ppo_fsdp_worker.py:105` (in `compute_proximal_logprobs`)
  - `async_ppo_fsdp_worker.py:169` (in `run_training`)
  - `fsdp_actor_worker.py:1290` (in `EmbodiedFSDPActor.compute_advantages_and_returns`)
  - `fsdp_actor_worker.py:1468` (in `EmbodiedFSDPActor.run_training`)
- Verify the cooldown path end-to-end before removing (run e2e tests with a config that exercises cooldown).


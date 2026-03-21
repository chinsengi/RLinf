# OpenPI FlowPPO in RLinf

This note explains the complete FlowPPO algorithm used by RLinf's OpenPI
policy, not just the `joint_logprob` flag.

The short version is:

- OpenPI is a flow-matching / denoising action policy
- RLinf treats each rollout action as the result of a multi-step denoising chain
- PPO is applied to the denoising transitions, not to a one-shot Gaussian policy
- `joint_logprob` controls whether PPO optimizes one denoising transition or the
  whole denoising trajectory

Relevant code:

- `rlinf/models/embodiment/openpi/openpi_action_model.py`
- `rlinf/models/embodiment/openpi/__init__.py`
- `rlinf/algorithms/utils.py`
- `rlinf/algorithms/losses.py`
- `rlinf/runners/async_ppo_embodied_runner.py`
- `examples/embodiment/config/yam_ppo_openpi.yaml`
- `examples/embodiment/config/maniskill_ppo_openpi.yaml`

## 1. What "FlowPPO" Means Here

For a standard PPO policy, the policy directly outputs an action distribution:

```math
a \sim \pi_\theta(a \mid o)
```

OpenPI is different. It starts from noise and iteratively denoises:

```text
x_T -> x_{T-1} -> ... -> x_1 -> x_0
```

where `x_0` is the final action chunk that is sent to the environment.

So in RLinf, PPO is not optimizing a single one-shot action distribution. It is
optimizing the denoising transition model:

```math
x_{t-1} \sim \pi_\theta(\cdot \mid x_t, o, t)
```

with:

- `o`: observation prefix, including images, language, and optional state
- `x_t`: current noisy action sample
- `t`: denoising timestep

That is the sense in which this is "FlowPPO": PPO on top of a flow / denoising
action policy.

## 2. End-to-End Training Loop

At a high level, RLinf does this:

1. Encode the observation prefix once
2. Build a KV cache for the prefix
3. Start from noise `x_T`
4. Run `num_steps` denoising steps to produce `x_0`
5. Execute `x_0` in the environment
6. Save the denoising chain and rollout metadata
7. Recompute the denoising log-probs during training
8. Apply standard PPO actor-critic loss on those log-probs and values

The OpenPI-specific pieces are:

- rollout-time sampling: `sample_actions()`
- training-time recomputation: `get_log_prob_value()`
- embodied runner-side proximal recomputation: `compute_proximal_logprobs()`
- PPO preprocessing: `preprocess_loss_inputs()`
- PPO loss: `compute_ppo_actor_critic_loss()`

## 3. Rollout-Side Algorithm

During rollout, `sample_actions()` runs the denoising policy.

### 3.1 Prefix encoding

The model first builds the observation prefix:

- image embeddings
- language token embeddings
- optional state handling through the suffix path

Then it caches the prefix attention KV once:

```text
prefix -> prefix_output, past_key_values
```

This cache is reused for every denoising step.

### 3.2 Denoising chain

The rollout starts from:

```math
x_T \sim \mathcal{N}(0, I)
```

or from externally supplied noise in DSRL mode.

At each denoising step `t`, the model predicts:

- a mean-like update `x_t_mean`
- a standard deviation `x_t_std`
- optionally a value estimate

Then it samples:

```math
x_{t-1} = x_t^{\text{mean}} + \epsilon_t \cdot x_t^{\text{std}}
```

with:

```math
\epsilon_t \sim \mathcal{N}(0, I)
```

The full chain is stored as:

```text
chains = [x_T, x_{T-1}, ..., x_0]
```

Along with:

- `prev_logprobs`
- `prev_values`
- `denoise_inds`

## 4. Transition Distribution Used by PPO

The log-probability used by PPO is always a Gaussian transition log-probability:

```math
\log \pi_\theta(x_{t-1} \mid x_t, o, t)
= -\log \sigma_\theta
- \frac{1}{2}\log(2\pi)
- \frac{1}{2}\left(\frac{x_{t-1} - \mu_\theta}{\sigma_\theta}\right)^2
```

In code, that is implemented by `get_logprob_norm()`.

Numerically:

- `mu` is `x_t_mean`
- `sigma` is `x_t_std`
- `sample` is the next denoised sample in the chain

## 5. How the Denoising Update Is Computed

`sample_mean_var_val()` computes the denoising transition.

First, the model predicts a velocity-like term:

```math
v_t = f_\theta(x_t, o, t)
```

Then RLinf constructs:

```math
x_0^{\text{pred}} = x_t - v_t \cdot t
```

```math
x_1^{\text{pred}} = x_t + v_t \cdot (1 - t)
```

The final transition mean and std depend on `noise_method`.

### 5.1 `noise_method: flow_sde`

This uses a schedule-derived stochastic update:

```math
\sigma_i =
\text{noise\_level}
\cdot
\sqrt{\frac{t_i}{1 - t_i}}
```

and then:

```math
x_t^{\text{std}} = \sqrt{\Delta t} \cdot \sigma_i
```

So `sigma` is not learned by a network here; it comes from the schedule.

### 5.2 `noise_method: flow_noise`

This uses a learned noise head:

```math
x_t^{\text{std}} = \sigma_\theta(x_t, o, t)
```

This is more expressive, but also much easier to destabilize because PPO now
backpropagates through a learned standard deviation.

### 5.3 `noise_method: flow_cps`

This uses a cosine / sine parameterization controlled by `noise_level`.

It is less commonly discussed in RLinf configs than `flow_sde` and
`flow_noise`, but it is implemented in the same transition function.

## 6. Where PPO Enters

Rollout stores:

- `chains`
- `denoise_inds`
- `prev_logprobs`
- `prev_values`

Then during training, `default_forward()` calls `get_log_prob_value()` to
recompute:

- current log-probs
- current values
- entropy

from the saved denoising chain.

So PPO compares:

- `old_logprobs`: saved during rollout
- `logprobs`: recomputed by the current policy

and forms the standard PPO ratio:

```math
r(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})
```

Then RLinf applies the usual clipped PPO objective:

```math
\mathcal{L}_{\text{actor}}
=
-\min\left(
r(\theta) A,\ \mathrm{clip}(r(\theta), 1-\epsilon_\text{low}, 1+\epsilon_\text{high}) A
\right)
```

plus a critic loss:

```math
\mathcal{L}_{\text{critic}} = \text{value loss}(V_\theta, R)
```

The final OpenPI PPO loss is:

```math
\mathcal{L}_{\text{PPO-Flow}} =
\mathcal{L}_{\text{actor}} + \mathcal{L}_{\text{critic}}
```

with entropy regularization added afterwards.

## 7. What `joint_logprob` Actually Changes

This flag decides whether PPO uses:

- one denoising transition
- or the whole denoising trajectory

### 7.1 `joint_logprob: false`

Only one denoising step is used for policy gradient.

Operationally:

- rollout samples one denoise index `k`
- only that transition is treated as a training transition
- `get_log_prob_value()` loops once

Conceptually:

```math
\ell_\theta = \log \pi_\theta(x_{k-1} \mid x_k, o, k)
```

This is usually more stable because PPO is only pushing on one transition.

### 7.2 `joint_logprob: true`

All denoising transitions are included, plus the initial noise prior term.

Conceptually:

```math
\log \pi_\theta(x_{0:T} \mid o)
\approx
\log p(x_T)
+
\sum_{t=1}^{T}\log \pi_\theta(x_{t-1}\mid x_t, o, t)
```

In RLinf's current implementation, rollout and recompute both collect the
per-step terms and then average across the denoising-step axis before PPO sees
them. That mean includes the initial noise prior term.

This gives denser supervision, but also a much stronger and often noisier
gradient.

### 7.3 Practical interaction with `noise_method`

Typical rule of thumb in RLinf:

- `flow_sde` -> usually `joint_logprob: false`
- `flow_noise` -> usually `joint_logprob: true`

That is why many OpenPI configs explicitly comment:

```yaml
joint_logprob: True  # False for flow_sde, True for flow_noise
```

## 8. Reward / Log-Prob / Entropy Aggregation

This part is easy to miss and matters a lot.

OpenPI log-probs are naturally shaped like:

```text
[batch, action_chunk, action_env_dim]
```

RLinf then aggregates them according to `algorithm.logprob_type`.

### 8.1 `algorithm.logprob_type`

Handled in `preprocess_loss_inputs()`:

- `token_level`
  Keeps per-dimension log-probs
- `action_level`
  Sums over action dimension, keeps per-chunk structure
- `chunk_level`
  Sums over both chunk and action dimensions, produces one scalar per sample

So even if OpenPI generates per-action-dimension transition log-probs, PPO may
see them at a coarser level depending on this flag.

### 8.2 `algorithm.reward_type`

If `reward_type == "chunk_level"`, RLinf flattens the advantage/value/return
side to match chunk-level PPO bookkeeping.

This is the common OpenPI PPO setup.

### 8.3 `algorithm.entropy_type`

Entropy is reshaped by `reshape_entropy()`:

- `action_level`: sum entropy over action dimension
- `chunk_level`: sum entropy over the last dimension and keep per-chunk structure

Important implementation detail:

- for `flow_noise`, entropy comes from the learned Gaussian std
- for `flow_sde` and `flow_cps`, RLinf currently returns zero entropy tensors

For OpenPI, configs commonly use:

- `reward_type: chunk_level`
- `entropy_type: chunk_level`

while `logprob_type` may vary by experiment.

## 9. Value Function in OpenPI PPO

OpenPI can attach a PPO value head in two ways.

### 9.1 Expert-side value

If:

- `add_value_head: true`
- `value_after_vlm: false`

then the value is computed from the suffix / expert features.

This is common for `pi0`.

### 9.2 VLM-side value

If:

- `add_value_head: true`
- `value_after_vlm: true`

then the value is computed from the prefix VLM features (`prefix_output`).

This is common for `pi0.5`.

`value_vlm_mode` then controls which prefix tokens are pooled:

- `mean_token`
- `last_token`
- `first_token`

### 9.3 Critic-side stabilization flags

When using expert-side values, these matter:

- `chunk_critic_input`
  Use only the first `action_chunk` steps for the critic input
- `detach_critic_input`
  Detach the critic features from the actor graph before value prediction

## 10. Complete FlowPPO Flag Reference

This section groups the most important flags by role.

### 10.1 Where the flags live in YAML

OpenPI settings are split across two config levels.

Top-level model interface flags live under:

```yaml
actor:
  model:
```

These are the ones users usually edit in experiment configs:

- `num_action_chunks`
- `action_dim`
- `num_steps`
- `add_value_head`

Those values are then mirrored into the nested OpenPI config defaults in
`model/pi0.yaml` and `model/pi0_5.yaml`, for example:

```yaml
actor:
  model:
    num_action_chunks: 10
    action_dim: 14
    num_steps: 4
    add_value_head: true
    openpi:
      action_chunk: ${actor.model.num_action_chunks}
      action_env_dim: ${actor.model.action_dim}
      num_steps: ${actor.model.num_steps}
      add_value_head: ${actor.model.add_value_head}
```

The actual OpenPI-specific knobs live under:

```yaml
actor:
  model:
    openpi:
```

### 10.2 OpenPI denoising-policy flags

Core denoising flags:

- `noise_method`
  One of `flow_sde`, `flow_noise`, `flow_cps`
- `num_steps`
  Number of denoising steps per action sample. Usually set at
  `actor.model.num_steps` and mirrored into `actor.model.openpi.num_steps`
- `action_horizon`
  How many action steps are predicted in one denoising rollout
- `action_chunk`
  How many action steps PPO focuses on. Usually mirrored from
  `actor.model.num_action_chunks`
- `action_env_dim`
  Effective action dimension used by PPO post-processing. Usually mirrored from
  `actor.model.action_dim`

Noise-shape / sampling flags:

- `noise_level`
  Fixed noise scale for `flow_sde` or `flow_cps`
- `noise_anneal`
  Whether to anneal the noise level over training
- `noise_params`
  Used for noise annealing; interpreted as a schedule config
- `noise_logvar_range`
  Min / max standard deviation range for learned `flow_noise`

Trajectory log-prob flags:

- `joint_logprob`
  Use one transition or the full denoising trajectory
- `ignore_last`
  In non-joint mode, prevents sampling the last denoising step as the PPO step
- `safe_get_logprob`
  Uses a simplified `-(x-\mu)^2` surrogate instead of the full Gaussian
  log-prob. This changes the objective and should be treated as a stability
  fallback, not as an equivalent formulation

Architecture / training flags:

- `train_expert_only`
  Freeze the VLM side and train the expert side
- `double_layer`
  Special acceleration-related path; cannot be enabled together with
  `joint_logprob`
- `num_images_in_input`
  How many image streams are actually used in the prefix

Value-head flags:

- `add_value_head`
  Required for PPO actor-critic. Usually set at `actor.model.add_value_head`
  and mirrored into `actor.model.openpi.add_value_head`
- `value_after_vlm`
  Use prefix VLM features for value
- `value_vlm_mode`
  Token pooling mode for VLM value head
- `chunk_critic_input`
  Restrict value input to action chunk
- `detach_critic_input`
  Stop actor gradients from flowing through the critic input

### 10.3 PPO-side RL flags

These live under:

```yaml
algorithm:
```

Core PPO flags:

- `adv_type`
  Usually `gae`
- `loss_type`
  Usually `actor_critic`
- `update_epoch`
  PPO update epochs per rollout batch
- `clip_ratio_high`
  PPO upper clip
- `clip_ratio_low`
  PPO lower clip
- `value_clip`
  Critic clip range
- `huber_delta`
  Huber critic loss parameter
- `gamma`
  Discount
- `gae_lambda`
  GAE lambda
- `normalize_advantages`
  Advantage normalization

Aggregation flags:

- `reward_type`
  Often `chunk_level`
- `logprob_type`
  `token_level`, `action_level`, or `chunk_level`
- `entropy_type`
  `action_level` or `chunk_level`

Regularization / misc flags:

- `entropy_bonus`
  Entropy coefficient
- `kl_beta`
  Present in shared PPO configs, but not typically important for embodied
  OpenPI PPO in current configs

### 10.4 Rollout flags that matter for FlowPPO

These live under:

```yaml
rollout:
```

Most important:

- `collect_prev_infos: true`
  Needed for GAE because OpenPI PPO uses `prev_values`
- `recompute_logprobs`
  In the embodied async runner, controls whether the actor first recomputes
  proximal log-probs before advantage / PPO updates
- `return_logprobs`
  If rollout-side log-probs are needed directly, this must be enabled on the
  rollout worker

Important note:

- embodied configs use `rollout.recompute_logprobs`
- reasoning configs use `algorithm.recompute_logprobs`

Do not mix those up when reading the rest of RLinf.

## 11. Recommended Config Patterns

### 11.1 Stable starting point for `flow_sde`

Usually:

- `noise_method: flow_sde`
- `joint_logprob: false`
- `entropy_bonus: 0.0`
- `reward_type: chunk_level`
- `logprob_type`: often `token_level` or `chunk_level`, experiment dependent
- `entropy_type: chunk_level`

Why:

- `sigma` is schedule-defined, not learned
- one-step PPO supervision is usually more stable

### 11.2 Stronger but riskier `flow_noise`

Usually:

- `noise_method: flow_noise`
- `joint_logprob: true`
- `entropy_bonus: 0.005` or similar
- `detach_critic_input: true` is often helpful
- careful monitoring of log-prob / entropy / grad stability

Why:

- `sigma` is learned
- the full denoising trajectory often gives better supervision
- but learned `sigma` plus PPO can blow up if not controlled

## 12. Why FlowPPO Can Go Unstable

The biggest numerical risk is the Gaussian log-prob:

```math
-\log \sigma_\theta
- \frac{1}{2}\left(\frac{x_{t-1} - \mu_\theta}{\sigma_\theta}\right)^2
```

If `sigma` gets too small:

- `-\log \sigma` explodes
- the squared residual term explodes
- gradients become huge

This gets worse when:

- `joint_logprob: true`
- `noise_method: flow_noise`
- `num_steps` is large
- `train_expert_only: true`
- critic and actor both push on unstable expert features

## 13. Mental Model

If you want one sentence to remember:

> RLinf OpenPI FlowPPO is PPO over a denoising trajectory, where each PPO
> "action probability" is really a Gaussian transition probability between two
> denoising states.

Then the main knobs are:

- how you parameterize denoising noise: `noise_method`
- how many denoising steps you optimize: `joint_logprob`, `num_steps`
- how PPO aggregates those log-probs: `logprob_type`
- how the critic is attached: `add_value_head`, `value_after_vlm`,
  `detach_critic_input`, `chunk_critic_input`

## 14. Minimal Config Snippets

Flow-SDE style:

```yaml
actor:
  model:
    num_steps: 10
    add_value_head: true
    openpi:
      noise_method: flow_sde
      joint_logprob: false

algorithm:
  adv_type: gae
  loss_type: actor_critic
  reward_type: chunk_level
  entropy_type: chunk_level
  entropy_bonus: 0.0

rollout:
  collect_prev_infos: true
```

Flow-Noise style:

```yaml
actor:
  model:
    num_steps: 10
    add_value_head: true
    openpi:
      noise_method: flow_noise
      joint_logprob: true
      noise_logvar_range: [0.08, 0.16]
      detach_critic_input: true

algorithm:
  adv_type: gae
  loss_type: actor_critic
  reward_type: chunk_level
  entropy_type: chunk_level
  entropy_bonus: 0.005

rollout:
  collect_prev_infos: true
```

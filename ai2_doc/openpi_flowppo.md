# OpenPI FlowPPO in RLinf

This note explains the complete FlowPPO algorithm used by RLinf's OpenPI
policy

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

### 1.1 Flow matching background

Flow matching defines a probability path that interpolates between a noise
distribution $p_1 = \mathcal{N}(0, I)$ and the data distribution $p_0$.
The conditional flow ODE is:

```math
\frac{dx}{dt} = v_\theta(x, t)
```

where $v_\theta$ is a learned velocity field. Given a data sample $x_0$ and
noise $x_1 \sim \mathcal{N}(0, I)$, the linear interpolation path is:

```math
x_t = (1 - t)\, x_0 + t\, x_1
```

Differentiating gives the ground-truth velocity:

```math
v^*(x_t, t) = x_1 - x_0
```

The flow matching training loss regresses the network velocity toward this
target:

```math
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}
\bigl[\lVert v_\theta(x_t, t) - (x_1 - x_0) \rVert^2 \bigr]
```

At inference the ODE is integrated backward from $t{=}1$ (noise) to $t{=}0$
(data). FlowPPO replaces the deterministic ODE with a stochastic transition
and applies PPO to the resulting chain.

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

The denoising timestep schedule is defined as:

```math
\{t_i\}_{i=0}^{K} = \bigl\{1,\; 1 - \tfrac{1}{K},\; 1 - \tfrac{2}{K},\;
\ldots,\; \tfrac{1}{K},\; 0\bigr\}
```

where $K$ = `num_steps` (typically 10). The step size at index $i$ is
$\delta_i = t_i - t_{i+1}$.

The rollout starts from:

```math
x_T \sim \mathcal{N}(0, I)
```

or from externally supplied noise in DSRL mode.

At each denoising step $i$, the network predicts a velocity
$v_\theta(x_{t_i}, o, t_i)$. From this velocity the model derives two
auxiliary predictions:

```math
\hat{x}_0 = x_{t_i} - v_\theta \cdot t_i
\qquad\text{(predicted clean action)}
```

```math
\hat{x}_1 = x_{t_i} + v_\theta \cdot (1 - t_i)
\qquad\text{(predicted noise)}
```

These are combined into a transition mean and standard deviation
(see Section 5 for the per-method formulas):

```math
\mu_i = \alpha_0^{(i)}\, \hat{x}_0 + \alpha_1^{(i)}\, \hat{x}_1
```

Then the next sample is drawn from a Gaussian transition:

```math
x_{t_{i+1}} = \mu_i + \sigma_i \,\epsilon_i,
\qquad \epsilon_i \sim \mathcal{N}(0, I)
```

Equivalently, the transition defines a conditional distribution:

```math
\pi_\theta(x_{t_{i+1}} \mid x_{t_i}, o, t_i)
= \mathcal{N}(x_{t_{i+1}};\; \mu_i,\; \sigma_i^2 I)
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

Each denoising transition is modelled as a diagonal Gaussian. The
per-dimension log-probability used by PPO is:

```math
\log \pi_\theta(x_{t-1} \mid x_t, o, t)
= \sum_{d=1}^{D} \left[
-\log \sigma_d
- \frac{1}{2}\log(2\pi)
- \frac{1}{2}\left(\frac{x_{t-1,d} - \mu_d}{\sigma_d}\right)^2
\right]
```

where $D$ is the action dimension (`action_env_dim`). In code, this is
implemented by `get_logprob_norm()`, which returns the un-summed
per-dimension terms with shape `[batch, action_chunk, action_dim]`.

A numerical safety guard is applied: if $\sigma_d \le 0$, the
corresponding log-prob is set to zero and $\sigma$ is replaced with 1 to
avoid NaN.

When `safe_get_logprob` is enabled, a simplified surrogate replaces the
full Gaussian:

```math
\log \pi^{\text{safe}} = -(x_{t-1} - \mu)^2
```

This drops the normalizing constant and variance terms, changing the
objective. It should be treated as a stability fallback, not an equivalent
formulation.

In code:

- `mu` is `x_t_mean`
- `sigma` is `x_t_std`
- `sample` is the next denoised sample in the chain

## 5. How the Denoising Update Is Computed

`sample_mean_var_val()` computes the denoising transition.

First, the model predicts a velocity-like term:

```math
v_t = f_\theta(x_t, o, t)
```

Then RLinf constructs two auxiliary predictions by inverting the linear
interpolation path $x_t = (1 - t)\,x_0 + t\,x_1$:

```math
\hat{x}_0 = x_t - v_t \cdot t
```

```math
\hat{x}_1 = x_t + v_t \cdot (1 - t)
```

The transition mean is always a weighted combination of these:

```math
\mu_i = \alpha_0\, \hat{x}_0 + \alpha_1\, \hat{x}_1
```

The weights $\alpha_0, \alpha_1$ and the standard deviation $\sigma_i$
depend on `noise_method`. Let $t' = t_i - \delta_i$ denote the target
timestep (i.e. $t_{i+1}$).

### 5.1 `noise_method: flow_sde`

This corresponds to an SDE discretization of the probability flow. The
diffusion coefficient is derived from the schedule:

```math
\sigma(t) = \text{noise\_level} \cdot
\sqrt{\frac{t}{1 - t}}
```

The transition weights and std at step $i$ are:

```math
\alpha_0 = 1 - t', \qquad
\alpha_1 = t' - \frac{\sigma_i^2 \,\delta_i}{2\, t_i}
```

```math
\sigma_i^{\text{std}} = \sqrt{\delta_i}\;\sigma(t_i)
```

The $\alpha_1$ correction term $-\sigma_i^2 \delta_i / (2t_i)$ arises from
the score-based SDE reverse drift. To see why: the reverse-time SDE for
a variance-exploding diffusion is

```math
dx = \bigl[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\bigr] dt + g(t)\, d\bar{W}
```

With the flow matching parameterisation $f(x,t) = v_\theta$ and
$g(t)^2 = \sigma(t)^2$, the Euler–Maruyama discretization over step
$\delta_i$ gives the drift correction
$-\sigma_i^2 \delta_i / (2 t_i)$ on the $\hat{x}_1$ coefficient.

So `sigma` is not learned by a network here; it comes entirely from the
schedule and `noise_level`.

### 5.2 `noise_method: flow_noise`

Here the standard deviation is produced by a learned noise head
(`ExploreNoiseNet`):

```math
\sigma_i^{\text{std}} = \sigma_\theta(x_{t_i}, o, t_i)
```

The transition weights are the simple interpolation coefficients:

```math
\alpha_0 = 1 - t', \qquad \alpha_1 = t'
```

The noise head output is clamped to `noise_logvar_range = [lo, hi]` to
prevent collapse or explosion:

```math
\sigma_\theta \in [lo, hi]
```

This is more expressive than `flow_sde` because the policy can
adapt its exploration noise per state, but also much easier to destabilize
because PPO now backpropagates through a learned standard deviation.

### 5.3 `noise_method: flow_cps`

This uses a cosine–sine parameterization controlled by `noise_level`:

```math
\alpha_0 = 1 - t', \qquad
\alpha_1 = t' \cos\!\bigl(\tfrac{\pi}{2}\,\text{noise\_level}\bigr)
```

```math
\sigma_i^{\text{std}} = t' \sin\!\bigl(\tfrac{\pi}{2}\,\text{noise\_level}\bigr)
```

As `noise_level` → 0 the transition becomes deterministic (pure ODE);
as `noise_level` → 1 the noise scale equals the interpolation coefficient,
similar to a VP-SDE. It is less commonly used than `flow_sde` and
`flow_noise` in RLinf configs but implemented in the same transition
function.

### 5.4 Summary of transition formulas

| Method | $\alpha_0$ | $\alpha_1$ | $\sigma_i^{\text{std}}$ | Learned? |
|---|---|---|---|---|
| `flow_sde` | $1 - t'$ | $t' - \frac{\sigma_i^2 \delta_i}{2 t_i}$ | $\sqrt{\delta_i}\;\sigma(t_i)$ | No |
| `flow_noise` | $1 - t'$ | $t'$ | $\sigma_\theta(x_t, o, t)$ | Yes |
| `flow_cps` | $1 - t'$ | $t'\cos(\frac{\pi}{2}\eta)$ | $t'\sin(\frac{\pi}{2}\eta)$ | No |

where $\eta$ = `noise_level` and $t' = t_i - \delta_i$.

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

Then RLinf applies the clipped PPO objective with **asymmetric** clip bounds:

```math
\mathcal{L}_{\text{actor}}
=
-\mathbb{E}\!\left[\min\!\bigl(
r(\theta)\, \hat{A},\;
\mathrm{clip}\!\bigl(r(\theta),\; 1-\epsilon_{\text{low}},\; 1+\epsilon_{\text{high}}\bigr)\, \hat{A}
\bigr)\right]
```

where $\epsilon_{\text{low}}$ = `clip_ratio_low` and
$\epsilon_{\text{high}}$ = `clip_ratio_high` (both default to 0.2).
Asymmetric clipping allows tighter control when advantages are negative
(preventing the policy from moving too far away from bad actions) vs
positive.

An optional **dual-clip** further bounds the loss when $\hat{A} < 0$:

```math
\mathcal{L}_{\text{actor}}^{\text{dual}} = \max\!\bigl(
\mathcal{L}_{\text{actor}},\;
-c\,\hat{A}
\bigr)
\quad\text{when } c > 1
```

where $c$ = `clip_ratio_c`. This prevents excessively large policy updates
on transitions with strongly negative advantage.

### 6.1 Critic loss

The value function is trained with a clipped Huber loss:

```math
V_{\text{clip}} = V_{\text{old}} + \mathrm{clip}\!\bigl(V_\theta - V_{\text{old}},\;
-\epsilon_v,\; \epsilon_v\bigr)
```

```math
\mathcal{L}_{\text{critic}} = \max\!\bigl(
L_\delta(R - V_\theta),\;
L_\delta(R - V_{\text{clip}})
\bigr)
```

where $R$ is the GAE return target (Section 8.4) and $L_\delta$ is the
Huber loss:

```math
L_\delta(e) = \begin{cases}
\frac{1}{2} e^2 & \text{if } |e| < \delta \\
\delta\,(|e| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
```

with $\delta$ = `huber_delta` (default 1.0) and $\epsilon_v$ = `value_clip`.

### 6.2 Entropy bonus

For `flow_noise`, the Gaussian entropy of each transition is:

```math
\mathcal{H}(\sigma_d) = \frac{1}{2}\log(2\pi e\, \sigma_d^2)
```

This is summed/averaged according to `entropy_type` (see Section 8.3).
For `flow_sde` and `flow_cps` the entropy is zero because $\sigma$ is not
learned.

### 6.3 Full FlowPPO loss

The final loss optimized by the actor worker is:

```math
\mathcal{L}_{\text{PPO-Flow}} =
\mathcal{L}_{\text{actor}}
+ \mathcal{L}_{\text{critic}}
- \beta_H \,\mathcal{H}
```

where $\beta_H$ = `entropy_bonus` (default 0). The gradient is accumulated
over `gradient_accumulation` micro-batches before each optimizer step.

## 7. What `joint_logprob` Actually Changes

This flag decides whether PPO uses:

- one denoising transition
- or the whole denoising trajectory

### 7.1 `joint_logprob: false` (single-step)

Only one denoising step is used for policy gradient.

Operationally:

- rollout samples one denoise index $k \sim \mathrm{Uniform}\{0,\ldots,K-1\}$
  (optionally excluding the last step if `ignore_last: true`)
- only that transition is treated as a training transition
- `get_log_prob_value()` loops once

The log-probability used by PPO is:

```math
\ell_\theta = \log \pi_\theta(x_{k-1} \mid x_k, o, k)
```

This is an unbiased single-sample estimate of the per-step policy gradient:

```math
\nabla_\theta \mathcal{L} \propto
\mathbb{E}_{k}\!\left[
\nabla_\theta \log \pi_\theta(x_{k-1} \mid x_k, o, k) \,\hat{A}
\right]
```

It is usually more stable because PPO is only pushing on one transition per
update, reducing gradient variance.

### 7.2 `joint_logprob: true` (full trajectory)

All denoising transitions are included, plus the initial noise prior term.
The joint log-probability of the full denoising trajectory factorises as:

```math
\log \pi_\theta(x_{0:T} \mid o)
=
\underbrace{\log p(x_T)}_{\text{prior}}
+
\sum_{i=0}^{K-1}\log \pi_\theta(x_{t_{i+1}}\mid x_{t_i}, o, t_i)
```

where $\log p(x_T) = -\frac{1}{2}\|x_T\|^2 - \frac{D}{2}\log(2\pi)$ is
the standard normal prior.

In RLinf's implementation, the per-step terms (including the prior) are
**averaged** across the $K{+}1$ terms before PPO sees them:

```math
\bar{\ell}_\theta = \frac{1}{K+1}\left[
\log p(x_T) + \sum_{i=0}^{K-1}\log \pi_\theta(x_{t_{i+1}}\mid x_{t_i}, o, t_i)
\right]
```

This gives denser supervision (every step contributes), but also a much
stronger and often noisier gradient because the ratio
$r(\theta) = \exp(\bar{\ell}_\theta - \bar{\ell}_{\theta_{\text{old}}})$
compounds errors across all $K$ steps.

### 7.3 Variance comparison

Under single-step mode, the gradient variance scales as $\mathrm{Var}(\nabla)$.
Under joint mode with $K$ steps, the averaged log-prob has variance
$\approx \mathrm{Var}(\nabla) / K$ per term but introduces covariance between
steps, so the effective variance can be either lower (correlated improvements
across steps) or higher (compounding errors). In practice, joint mode
requires more careful hyperparameter tuning.

### 7.4 Practical interaction with `noise_method`

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

Handled in `preprocess_loss_inputs()`. The raw per-dimension log-prob tensor
has shape $[B, C, D]$ where $B$ = batch, $C$ = `action_chunk`,
$D$ = `action_env_dim`:

- `token_level` — keeps the full $[B, C, D]$ shape. Advantages and loss
  masks are broadcast to match:

```math
\ell_{b,c,d} = \log \pi_\theta^{(d)}
```

- `action_level` — sums over the action dimension:

```math
\ell_{b,c} = \sum_{d=1}^{D} \log \pi_\theta^{(d)}
```

  Result shape: $[B, C]$.

- `chunk_level` — sums over both chunk and action dimensions:

```math
\ell_b = \sum_{c=1}^{C}\sum_{d=1}^{D} \log \pi_\theta^{(c,d)}
```

  Result shape: $[B]$.

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

### 8.4 GAE (Generalized Advantage Estimation)

RLinf computes advantages using GAE. Given a trajectory of rewards
$\{r_0, \ldots, r_{T-1}\}$, values $\{V_0, \ldots, V_T\}$, and
done flags $\{d_0, \ldots, d_T\}$, define the TD residual:

```math
\delta_t = r_t + \gamma\,(1 - d_{t+1})\,V_{t+1} - V_t
```

The GAE advantage is computed by a backward pass:

```math
\hat{A}_T = 0
```

```math
\hat{A}_t = \delta_t + \gamma\,\lambda\,(1 - d_{t+1})\,\hat{A}_{t+1}
\qquad t = T{-}1, \ldots, 0
```

This is the exponentially-weighted average of $n$-step TD errors. When
expanded:

```math
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma\lambda)^l \,\delta_{t+l}
\prod_{j=1}^{l}(1 - d_{t+j})
```

The return target for the critic is:

```math
R_t = \hat{A}_t + V_t
```

When `normalize_advantages` is enabled:

```math
\hat{A}_t \leftarrow
\frac{\hat{A}_t - \mathrm{mean}(\hat{A})}
{\mathrm{std}(\hat{A}) + 10^{-5}}
```

Default parameters: $\gamma$ = `gamma` (1.0), $\lambda$ = `gae_lambda`
(1.0). With $\gamma = \lambda = 1$ and no dones, GAE reduces to the
Monte Carlo return $\hat{A}_t = \sum_{l \ge 0} r_{t+l} - V_t$.

## 9. Value Function in OpenPI PPO

OpenPI can attach a PPO value head in two ways.

### 9.1 Expert-side value

If:

- `add_value_head: true`
- `value_after_vlm: false`

then the value is computed from the suffix / expert features:

```math
h_{\text{val}} = \begin{cases}
\frac{1}{C}\sum_{c=1}^{C} z_c & \text{if \texttt{chunk\_critic\_input}} \\
\frac{1}{H}\sum_{j=1}^{H} z_j & \text{otherwise}
\end{cases}
```

where $z_j$ are the suffix output tokens, $C$ = `action_chunk`, and
$H$ = `action_horizon`. If `detach_critic_input` is set:

```math
h_{\text{val}} \leftarrow \mathrm{stopgrad}(h_{\text{val}})
```

Then:

```math
V_\theta = W_v\, h_{\text{val}} + b_v
```

This is common for `pi0`.

### 9.2 VLM-side value

If:

- `add_value_head: true`
- `value_after_vlm: true`

then the value is computed from the prefix VLM features (`prefix_output`).

`value_vlm_mode` controls which prefix tokens are pooled:

- `mean_token` — average over image + language tokens
- `last_token` — use only the final prefix token
- `first_token` — use only the first prefix token

```math
h_{\text{vlm}} = \mathrm{pool}(\{p_j : j \in \mathcal{M}\})
```

where $\mathcal{M}$ is the set of selected prefix token indices, and
$V_\theta = W_v\, h_{\text{vlm}} + b_v$.

This is common for `pi0.5`.

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

The biggest numerical risk is the Gaussian log-prob. Consider the
per-dimension gradient with respect to the mean:

```math
\frac{\partial \log \pi}{\partial \mu}
= \frac{x_{t-1} - \mu}{\sigma^2}
```

and with respect to the standard deviation:

```math
\frac{\partial \log \pi}{\partial \sigma}
= -\frac{1}{\sigma} + \frac{(x_{t-1} - \mu)^2}{\sigma^3}
```

As $\sigma \to 0$:

- The log-normalizer $-\log \sigma \to +\infty$
- The residual $(x_{t-1} - \mu)^2 / \sigma^2 \to +\infty$ unless the
  mean is exact
- $\partial / \partial \sigma \propto 1/\sigma^3$ — gradients blow up
  cubically

The PPO ratio compounds this: with joint log-probs the ratio is

```math
r(\theta) = \exp\!\Bigl(\sum_{i} \bigl[\log\pi_\theta^{(i)}
- \log\pi_{\theta_\text{old}}^{(i)}\bigr]\Bigr)
```

Even small per-step log-prob changes accumulate across $K$ steps, producing
extreme ratios that clipping alone cannot contain when $K$ is large.

This gets worse when:

- `joint_logprob: true` — ratio compounds across all $K$ steps
- `noise_method: flow_noise` — learned $\sigma$ can collapse
- `num_steps` is large — more terms in the product
- `train_expert_only: true` — all gradient pressure concentrates on the
  expert parameters
- critic and actor both push on unstable expert features

Mitigations in RLinf:

- `noise_logvar_range` clamps learned $\sigma$ to $[\sigma_{\min}, \sigma_{\max}]$
- `safe_get_logprob` drops the $\sigma$-dependent terms entirely
- `clip_log_ratio_min/max` bounds the log-ratio before exponentiation
- `detach_critic_input` prevents critic gradients from destabilizing the
  actor backbone

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

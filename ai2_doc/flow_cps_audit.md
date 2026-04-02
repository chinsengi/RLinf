# Flow-CPS Audit

This note audits RLinf's `flow_cps` implementation. The concrete production bug
we found and fixed is that PPO was allowed to sample and score the final
`flow_cps` denoising step even though that step is deterministic. The rest of
this note keeps the broader math discussion as a design review for future work.

Relevant code:

- `rlinf/models/embodiment/openpi/openpi_action_model.py`
- `rlinf/models/embodiment/pi05/inference_adapter.py`
- `rlinf/models/embodiment/dexbotic_pi/dexbotic_pi_policy.py`
- `tests/unit_tests/test_pi05_cps_schedule.py`

## Short Conclusion

The concrete PPO bug was:

- the last `flow_cps` denoising step has `t' = 0`
- therefore its transition standard deviation is exactly zero
- but rollout and PPO scoring still treated that step as eligible for training

That yields a zero-information log-prob term. In single-step PPO it wastes a
sample whenever that final step is chosen; in joint-logprob PPO it dilutes the
aggregated log-prob with a deterministic term.

RLinf now excludes that final `flow_cps` step from PPO scoring.

Separately, the current `flow_cps` implementation reuses the deterministic flow inversion

```math
\hat{x}_0 = x_t - t v_t,\qquad
\hat{x}_1 = x_t + (1 - t) v_t
```

and then adds a cosine-sine stochastic transition on top:

```math
x_{t'} \sim \mathcal{N}\!\left(
(1 - t') \hat{x}_0 + t' \cos\theta \,\hat{x}_1,\;
t'^2 \sin^2\theta\, I
\right),
\qquad \theta = \frac{\pi}{2}\,\text{noise\_level}.
```

That combination is not the conditional transition induced by a coherent
constant-preserving flow path. The deterministic inversion formula is only exact
for the linear ODE path

```math
x_t = (1 - t) x_0 + t x_1.
```

Once `flow_cps` injects stochasticity into the endpoint mixture, the same
inversion no longer recovers the underlying endpoints. In other words: the
current `flow_cps` branch keeps the `flow_sde` / ODE inversion algebra after
changing the path family.

That broader design question remains open, but it is distinct from the concrete
PPO bug above.

## 1. What the code currently does

In `sample_mean_var_val()` the code first predicts a velocity-like tensor
`v_t = f_\theta(x_t, o, t)` and then forms

```math
\hat{x}_0 = x_t - t v_t,
\qquad
\hat{x}_1 = x_t + (1 - t) v_t.
```

For `flow_cps`, it then uses

```math
\alpha_0 = 1 - t',
\qquad
\alpha_1 = t' \cos\theta,
\qquad
\sigma = t' \sin\theta,
\qquad
t' = t - \delta.
```

So the implemented transition is

```math
\mu_{\text{impl}}(x_t, v_t)
=
(1 - t') \hat{x}_0 + t' \cos\theta \,\hat{x}_1.
```

Expanding in terms of `x_t` and `v_t`:

```math
\mu_{\text{impl}}
=
\Bigl[(1 - t') + t' \cos\theta\Bigr] x_t

+ \Bigl[-t(1 - t') + (1 - t)t'\cos\theta\Bigr] v_t.
```

This is the policy kernel that rollout and training both use today.

## 2. Why the inversion is valid for deterministic flow

For the ordinary linear flow path

```math
x_t = (1 - t)x_0 + t x_1,
```

the true velocity is

```math
v^* = x_1 - x_0.
```

Then the inversion used in code is exact:

```math
x_t - t v^*
=
(1 - t)x_0 + t x_1 - t(x_1 - x_0)
= x_0,
```

```math
x_t + (1 - t) v^*
=
(1 - t)x_0 + t x_1 + (1 - t)(x_1 - x_0)
= x_1.
```

So for the deterministic ODE case, the formulas for `\hat{x}_0` and
`\hat{x}_1` make sense.

## 3. Why this breaks for a constant-preserving noisy path

If `flow_cps` is intended to represent a constant-preserving noisy endpoint
mixture, the natural path is something like

```math
x_t = (1 - t)x_0 + t\bigl(\cos\theta\, x_1 + \sin\theta\, \varepsilon\bigr),
\qquad
\varepsilon \sim \mathcal{N}(0, I).
```

Then the true time derivative is

```math
\frac{d x_t}{dt}
=
-x_0 + \cos\theta\, x_1 + \sin\theta\, \varepsilon.
```

This is not equal to `x_1 - x_0`, and the code's inversion no longer works.

If we plug the noisy path into the current inversion formulas, we get

```math
\hat{x}_0
=
x_t - t v^*
=
x_0 + t(\cos\theta - 1)x_1 + t\sin\theta\,\varepsilon,
```

which is not `x_0` unless `\theta = 0`.

Likewise,

```math
\hat{x}_1
=
x_t + (1 - t) v^*
=
\bigl((1 - t) + t\cos\theta\bigr)x_1 + t\sin\theta\,\varepsilon,
```

which is not `x_1`.

So the current code is using endpoint reconstruction formulas that are exact
only for the deterministic interpolation path, while `flow_cps` changes the
path family.

## 4. What this means operationally

The problem is not just cosmetic naming. It changes the semantics of the PPO
transition density.

The current implementation defines a Gaussian policy kernel, but it is not the
conditional kernel of a coherent constant-preserving flow process derived from
the same latent path assumptions as the ODE / `flow_sde` case.

That mismatch means at least one of these must be false:

1. `v_t` is still the deterministic flow velocity / endpoint difference.
2. `\hat{x}_0 = x_t - t v_t` and `\hat{x}_1 = x_t + (1-t) v_t` are valid
   endpoint reconstructions.
3. The `flow_cps` Gaussian step is a correct constant-preserving transition.

The current code assumes all three simultaneously.

## 5. Smaller concrete bug: final step can still be chosen for PPO

For the last denoising step:

```math
t = \delta
\quad\Longrightarrow\quad
t' = t - \delta = 0.
```

Then in `flow_cps`:

```math
\alpha_1 = 0,
\qquad
\sigma = 0.
```

So the final step is deterministic:

```math
x_{t'} = \hat{x}_0.
```

However, the single-step PPO code can still sample this step as the training
transition. When that happens, `get_logprob_norm()` masks out the zero-variance
term and returns zero log-prob, so PPO gets a zero-information transition.

That is not the main theoretical issue above, but it is a concrete training bug
that wastes samples.

## 6. Why the existing test does not catch the real issue

`tests/unit_tests/test_pi05_cps_schedule.py` only checks that the helper
returns the same cosine-sine coefficients the code already hard-codes.

That verifies

```math
\alpha_1 = t' \cos\theta,\qquad \sigma = t' \sin\theta
```

but it does not verify that these coefficients are compatible with the
`(\hat{x}_0, \hat{x}_1)` reconstruction or with a coherent stochastic flow
transition.

So the test is self-consistency with the implementation, not correctness of the
underlying math.

## 7. Most likely fix directions

There are two coherent ways to fix this.

### Option A: keep the endpoint inversion, drop `flow_cps`

If the model output is meant to stay as the usual flow velocity / endpoint
difference, then `flow_cps` should not pretend to be a principled
constant-preserving path. In that case the safest fix is to remove or disable
the `flow_cps` branch.

### Option B: derive a new parameterization for CPS

If `flow_cps` is intended to be real, then the model needs a parameterization
whose state reconstruction is valid for the CPS path. That means deriving the
transition from a consistent latent-variable model and updating:

1. the definition of the model target `v_t`
2. the formulas for `\hat{x}_0`, `\hat{x}_1` (or replacing them entirely)
3. the rollout-time sampling kernel
4. the training-time recomputed log-prob kernel

Without doing all four together, the current implementation remains internally
inconsistent.

## 8. Bottom line

The likely mistake is here:

- the `flow_cps` branch changes the stochastic path coefficients
- but the code still uses the deterministic-flow endpoint inversion formulas

That is the core mathematical inconsistency.

Separately:

- the final `flow_cps` step is deterministic
- but the single-step PPO sampler can still choose it as the training step

So there is both a theory-level mismatch and a smaller concrete sampling bug.

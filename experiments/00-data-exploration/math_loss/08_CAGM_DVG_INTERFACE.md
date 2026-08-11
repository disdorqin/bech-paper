# CAGM-DVG Interface and Risk Decomposition

> Part of HCH v2 Derived Loss Math Design v0.1

## 9.1 Division of Labor

| Module | What It Provides | How It's Computed |
|---|---|---|
| **FS Skew-t Head** | `pi_plus, pi_minus, M_plus, M_minus, m_plus, m_minus` | Closed-form CDF + partial moment formulas |
| **Candidate Generator** | `Delta_down = -lambda_neg * M_minus, Delta_up = lambda_pos * M_plus` | Distribution-derived with S3 calibration |
| **CAGM (Memory)** | Day keys, retrieved neighbor days, per-action historical gains | S3-only: no S4 data |
| **DVG (Router)** | Risk-adjusted action values `V_down, V_identity=0, V_up` | From CAGM's retrieved gains |
| **Action Selector** | `argmax(V_down, 0, V_up)` with Identity=0 baseline | argmax |

## 9.2 Risk Terms

Three sources of uncertainty exist in the HCH pipeline:

### 9.2.1 Aleatoric Uncertainty (from distribution head)

```
U_aleatoric(Z) = Var(R | Z) = sigma(Z)^2 * Var_FS(gamma(Z), nu(Z))
```

This is the intrinsic uncertainty in the residual given Z. It measures
"how spread out are residuals even with perfect Z knowledge."

Available as a DVG feature: high aleatoric variance -> corrections may
be noisy -> DVG should be cautious.

### 9.2.2 Retrieval Epistemic Uncertainty (from CAGM)

```
U_retrieval(day_d) = Var_{k in N(d)} [G_k^action]
```

The variance of action gains among retrieved neighbor days. High variance
means similar historical days had inconsistent action outcomes.

### 9.2.3 Candidate Error Uncertainty (distribution misspecification)

```
U_candidate(Z) = |M_plus(Z) - M_plus_empirical(Z)|  (S2 residual)
```

The gap between model-predicted partial moment and empirical partial moment
in the S2 training data. Large gaps indicate distribution misspecification.

## 9.3 Non-Duplicative Risk Decomposition

**Risk of double-counting**: If the distribution head says "uncertain" AND
CAGM says "large gain variance", these may reflect the same underlying
phenomenon (extreme tail events). Using both as independent risk terms
would double-penalize tail corrections.

**Recommended approach**:

The DVG should receive three features per candidate:
1. `G_bar`: historical mean gain (from CAGM) -- primary signal
2. `sigma_gain`: std of gains among neighbors -- retrieval risk
3. `sigma_resid`: predicted residual std from distribution -- aleatoric risk

Then use a lightweight NN (2 dense layers) to map `(G_bar, sigma_gain, sigma_resid) -> V_action`.
This lets the NN learn to downweight when risks are redundant, rather than
us hard-coding a decomposition.

## 9.4 DVG Decision Rule

Given the three value estimates `(V_down, 0, V_up)`:

```
If max(V_down, V_up) <= 0:
    action = Identity  (correction not worth the risk)
elif V_down > V_up:
    action = Down
    correction = lambda_neg * M_minus(Z)  # from distribution
else:
    action = Up
    correction = lambda_pos * M_plus(Z)
```

**Note**: The DVG selects the ACTION (which of three), not the magnitude.
The magnitude comes from the distribution head (possibly with S3 lambda shrinking).

## 9.5 Candidate Uncertainty as DVG Feature

The distribution head should provide:
```
candidate_down = {
    "delta": M_minus,
    "pi": pi_minus,
    "m": m_minus,
    "sigma_resid": sigma * sqrt(Var_FS(gamma, nu)),
    "log_lik_ratio": log(p_tail / p_body)  # tail vs body log-density ratio
}
```

These feed into the DVG score network, allowing it to learn when to trust
the distribution head's magnitude estimates.

## 9.6 eta/tau Parameters

The current DVG has hardcoded `k/eta/tau`. Under the new design:

- `k` (number of neighbors): should be a hyperparameter tuned on S3, not S4
- `eta` (temperature for softmax over neighbors): can be absorbed into the
  DVG NN's learned weighting
- `tau` (action probability threshold): replaced by the DVG's `max(V_down, V_up) > 0` check

The distribution head does not need `eta/tau`. It provides soft probabilities
and expected values; the DVG makes the hard binary choice.

## 9.7 Tensor Interface Contract

### Input to FS skew-t head (from encoder):
```
Shape: (batch_size, d_feat)
Content: [Z_state_features, s_t, hour_onehot, dow_onehot, month_onehot]
```

### Output from FS skew-t head:
```
Shape per sample: (4,)  # (mu, log_sigma, nu_tilde, log_gamma)
```

### Output from candidate generator:
```
{
    "pi_neg": (batch_size,),     # P(R < 0 | Z)
    "pi_pos": (batch_size,),     # P(R > 0 | Z)
    "M_neg":  (batch_size,),     # E[R * 1(R<0) | Z]  (<= 0)
    "M_pos":  (batch_size,),     # E[R * 1(R>0) | Z]  (>= 0)
    "m_neg":  (batch_size,),     # E[-R | R<0, Z]
    "m_pos":  (batch_size,),     # E[R | R>0, Z]
    "sigma_aleatoric": (batch_size,),  # sqrt(Var(R|Z))
}
```

### Input to CAGM (per day):
```
Shape: (24, d_feat + 7)  # 24 hours x (features + 7 distribution outputs)
```

### Output from CAGM + DVG:
```
Shape: (24,)  # per-hour action: {-1, 0, 1} for Down/Identity/Up
Shape: (24,)  # per-hour correction magnitude
```

## 9.8 Loss Functions

| Component | Loss | Data |
|---|---|---|
| FS skew-t NLL | `-log f_FS(r | mu, sigma, nu, gamma)` | S2 training |
| DVG value network | `(V_action - G_historical)^2` | S3 calibration |
| S3 lambda grid search | Maximize LCB (MAE reduction) s.t. harm <= budget | S3 calibration |

No joint multi-task loss. The distribution head is trained first (S2),
then frozen during DVG training (S3). This prevents the distribution from
adapting to make DVG look good.

# P0 Decision: Is Extreme-Price Correction an Episode-Editing Problem?

> Date: 2026-08-07  
> Protocol: S1-S4 chronological isolation; S4 complete 24-hour days only  
> Scope: 17 public price series x 2 frozen backbones (Linear, GBDT)  
> Decision: **problem anchor GO; algorithm object NOT YET READY**

## 1. Evidence

### Negative prices are episodes, not isolated points

- Across markets with S4 negative events: 1,945 episodes / 10,300 event hours.
- 83.4% of episodes last at least two hours; 59.6% last at least four hours.
- In the day-ahead subset alone: 998 episodes / 4,570 hours; 83.0% are multi-hour and 54.1% last at least four hours.
- In eight UniElecPrice day-ahead markets, the median episode duration is about four hours; each market with events has a multi-hour rate of 80.0%-90.4%.

This pattern is not specific to NEM spot prices. It appears in both day-ahead and spot/real-time series.

### Frozen backbones fail at episode level

On the 998 day-ahead negative episodes:

| Backbone | Episode recall | Complete-miss rate | Point recall |
|---|---:|---:|---:|
| Linear | 17.7% | 82.3% | 13.8% |
| GBDT | 7.8% | 92.2% | 6.6% |

The dominant failure is not noisy pointwise fragmentation. It is failure to insert an entire missing event, followed by inaccurate start/end boundaries among matched events.

### Current BECH only partially addresses the structure

On NEM negative episodes:

| Backbone | Base recall | BECH recall | Delta |
|---|---:|---:|---:|
| Linear | 46.9% | 56.7% | +9.8 pp |
| GBDT | 40.2% | 53.4% | +13.2 pp |

BECH improves point/event recall, but most recovered events remain boundary mismatches. On the day-ahead subset, BECH makes no event-recall improvement for either audited backbone. This is a concrete limitation of the current pointwise correction design.

## 2. Positive-Spike Caveat

The current positive label is a fixed S1 p99 threshold. Under regime shift it ceases to mean a rare spike: in the audited UniElecPrice S4 segments, 48.0% of hours exceed the old S1 p99, versus 0.56% in NEM. Multi-hour "spike episodes" in that table can therefore be price-level regime changes rather than transient spikes.

Decision: do not use the current spike topology to justify the new algorithm. Keep positive-spike abstention as a separate safety behavior until the event definition is repaired using a pre-registered market rule or cutoff-safe adaptive threshold.

## 3. What This Does and Does Not Validate

Validated:

- Negative-price correction should operate on a 24-point event structure rather than independent hours.
- Missing-event insertion and boundary correction are the main unresolved actions.
- The gap exists in public day-ahead datasets with only `timestamp, price`; paired DA/RT inputs are unnecessary.

Not validated:

- TV/fused-lasso, HMM/HSMM, CRF, Viterbi decoding, or duration constraints as a new method. These were already rejected as standard structured-prediction components.
- A new theorem. Contiguity and exact no-action fallback are useful properties but not independent theoretical innovations.
- A-level novelty. The evidence establishes a real problem, not a novel solution.

## 4. Refined Algorithm Hypothesis

Represent the frozen base forecast and truth as sets of signed extreme episodes. During training, compute a minimum-cost matching between base and true episodes, producing an edit script:

- `KEEP/SCALE`: retain an event and correct its magnitude/shape;
- `SHIFT`: move its start/end boundaries;
- `DELETE`: remove a false base episode;
- `INSERT`: add an episode entirely missed by the base.

A model-agnostic correction head predicts this edit set from the base 24-point vector and cutoff-safe features, then modifies only the selected supports. With no accepted edit, the output is exactly the base forecast.

This is materially different from the previously rejected "smooth the whole trajectory with TV/HMM" proposal because the learning target is the **minimal edit script relative to a frozen base**, directly aligned with the observed failure taxonomy. It is still vulnerable to reduction as temporal set prediction + residual correction, so it remains a hypothesis pending formula-level collision testing.

## 5. Next Gates

1. Compare the edit operator against the strongest reduction attack: DETR-style temporal set prediction, HSMM/CRF duration decoding, TV/fused-lasso repair, RIGS-style local refinement, CRC, and ordinary pointwise residual correction.
2. Define the matching cost and edit supervision without using test labels or arbitrary duration constants.
3. Build a minimal prototype on S2/S3 only; evaluate S4 event recall, boundary error, duration error, event magnitude error, normal-hour harm, and exact fallback.
4. Stop if direct 24-vector MLP/Transformer or HSMM matches the edit operator, or if `INSERT/SHIFT` ablations do not contribute independently.

## 6. Artifacts

- `run_episode_audit.py`: reproducible audit.
- `label_episode_summary.csv`: event topology.
- `model_episode_summary.csv`: base/BECH event metrics.
- `event_failure_detail.csv`: per-event failure taxonomy.
- `bech_episode_deltas.csv`: per dataset/backbone deltas.
- `episode_audit.md`: generated raw report.

**Bottom line: the episode-level negative-price failure is real and cross-market. The paper now has a defensible problem anchor, but the event-edit operator must still survive direct-neighbor search and a reduction attack before it becomes the v7 main innovation.**

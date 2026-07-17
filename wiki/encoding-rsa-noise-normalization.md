# Encoding + RSA noise normalization

Status: implementation contract for runner schema 3, revised 2026-07-16.
Human review companion: [`figures/noise_normalization_review.html`](../figures/noise_normalization_review.html).

## Decision

- Keep the Algonauts encoding benchmark and all raw scores.
- Keep Triple-N encoding at the original coarse `area_label` level. Region and
  `unit_type` are not encoding segmentations.
- Add targetwise noise-normalized signed and squared scores. Alpha selection remains
  based on raw correlation. Algonauts keeps every response target for its raw score;
  Triple-N deliberately fits and evaluates only the paper-compatible reliable-unit
  population (`reliability_best >= 0.4`) so its raw and normalized scores describe
  the same units.
- Keep raw and normalized plots separate. Never clip normalized values to `[0, 1]`;
  finite-sample estimates above one are diagnostic information.

## DECISION OVERRIDE: reject `region x UnitType` encoding

The earlier schema-3 plan proposed EVC/IT crossed with `UnitType`. The user rejected
that design after validation showed that `UnitType` is not a temporal-response
cluster. This conflicts with the earlier decision recorded in this wiki; the current
and authoritative design is **one encoding score per coarse `area_label`**. RSA is
maintained separately and was not changed by this encoding rollback.

## Why `UnitType` is not an encoding cluster

Existing source comments describe the MAT field `UnitType` as a firing-dynamics or
PSTH k-means cluster. That interpretation conflicts with the Triple-N data pipeline
and must not be propagated.

The stored `UnitType` field is a **spike-sorting quality class**: single-unit activity
(SU), multi-unit activity (MUA), or non-somatic unit. Triple-N preprocessing uses
BombCell to classify units into noise, SU, MUA, and non-somatic classes; the released
good-unit file excludes noise. The paper's Extended Data Fig. 2 uses the same three
retained quality classes. The paper also analyzes three temporal response types, but
that is a separate derived clustering and is not what this repository currently loads
from `session["UnitType"]`.

Therefore `EVC | 1`, `EVC | 2`, etc. would be quality-stratified neural populations,
not temporal clusters, and are not produced by encoding. If true PSTH clusters are
added later, load their dedicated field under a distinct name such as
`temporal_cluster` and version the analysis again.

Evidence: the [Triple-N preprocessing README](https://github.com/liyipeng-moon/Triple-N/tree/main/Preprocess)
describes BombCell quality classification, and [Li et al. (2026), Extended Data Fig. 2](https://doi.org/10.1038/s41593-026-02322-z)
names SU, MUA, and non-somatic unit classes. The official
[dataset documentation](https://liyipeng-moon.github.io/Triple-N-Docs/) and
[dataset record](https://doi.org/10.57760/sciencedb.33556) are the authority for
future field changes.

## Encoding formulas

For target `v`, compute out-of-sample Pearson correlation across held-out stimuli:

```text
r_v = corr(y_v, yhat_v)
```

Ceilings are represented in squared-correlation / reliability units `NC_v`. Normalize
**per target before averaging**, never by dividing a group-average correlation by a
group-average ceiling:

```text
signed_v  = r_v / sqrt(NC_v)
squared_v = r_v^2 / NC_v
```

The signed metric preserves direction and is the primary scientific diagnostic. The
squared metric is compatible with the official Algonauts 2023 leaderboard convention
and expresses estimated predictable variance captured. Keep both because squaring can
make a strongly wrong-signed prediction look good. See the official
[Algonauts 2023 evaluation definition](https://algonautsproject.com/2023/challenge.html)
and Schoppe et al. (2016), [doi:10.3389/fncom.2016.00010](https://doi.org/10.3389/fncom.2016.00010).

### Algonauts ceiling source

Use the official per-vertex arrays, aligned with response targets after applying the
same subject, hemisphere, and ROI masks:

```text
<ceiling-root>/subjXX/test_split/noise_ceiling/lh_noise_ceiling.npy
<ceiling-root>/subjXX/test_split/noise_ceiling/rh_noise_ceiling.npy
```

An explicitly configured `noise_ceiling_dir` is authoritative and fails visibly if
wrong. When it is omitted, discovery checks the configured Algonauts root and then a
sibling `test/` root beside the training root (the standard unpacked layout).

Only finite `NC_v > 0` targets enter normalized summaries. Missing files should leave
normalized outputs unavailable with an explicit warning, while raw encoding remains
usable. A vector-length mismatch is an alignment error and must fail loudly.

The official arrays are estimated for the held-out challenge responses, whereas this
repository cross-validates on the released training images. Using them here assumes a
vertex's repeat-noise level transfers across the two image splits. Report raw scores
alongside normalized scores so that assumption remains visible rather than baked into
the only metric.

### Triple-N ceiling source and threshold

Use each unit's released `reliability_best`, a Spearman-Brown-corrected repeat
reliability. Treat it as `NC_v` for the formulas above. The normalized Triple-N summary
uses only finite units with `reliability_best >= 0.4`, matching the paper's reliable-unit
analysis cutoff. The cutoff is applied before Triple-N nested CV, so raw correlation,
alpha selection, and normalization all describe the same reliable-unit population.
Record the threshold and `n_ceiling_targets` in every row so coverage cannot be
mistaken for model quality.

This follows the general recommendation to separate signal reliability from model
accuracy and to report exactly how reliability enters normalization. See Van Bree,
Styrnal & Hebart (2025), [doi:10.31234/osf.io/gjk45_v2](https://doi.org/10.31234/osf.io/gjk45_v2),
which is also cited by the Triple-N paper.

## RSA formulas

**Decision correction:** the first normalization implementation silently replaced the
historical raw score with a different estimand. That changed the non-normalized plots
and conflicted with the requirement to add normalization without redefining the raw
baseline. The authoritative schema is now:

- `spearman_rho`: historical correlation of the model RDM with the group-mean RDM;
  this drives raw tables, plots, and summaries.
- `spearman_rho_individual_mean`: mean model-to-individual Spearman correlation
  (subjects for Algonauts, macaques for Triple-N).
- normalized columns: the individual-mean score divided by its matched upper ceiling.

Correlation with a mean RDM is not equal to the mean of correlations, so these values
must remain separate. The normalized columns use the upper ceiling in matching
individual-correlation units:

```text
noise_normalized_rho  = spearman_rho_individual_mean / noise_ceiling_high
noise_normalized_rho2 = spearman_rho_individual_mean^2 / noise_ceiling_high^2
```

Non-finite or non-positive upper ceilings yield `NaN`. Values above one are retained.
Lower and upper ceilings remain in the table because they communicate inter-subject or
inter-macaque consistency rather than acting as error bars. This matches standard RSA
noise-ceiling practice: Nili et al. (2014),
[doi:10.1371/journal.pcbi.1003553](https://doi.org/10.1371/journal.pcbi.1003553).

**Triple-N matched-target rule:** construct one RDM per macaque and neural subset, then
average model-to-macaque correlations. Its ceiling uses those same macaque RDMs, neural
subset, and ordered stimuli. Never normalize a pooled-unit or group-mean correlation
with an individual-level ceiling. A group without enough independent macaque RDMs has
no defensible empirical ceiling and stays `NaN`.

The runner does not retain every large individual RDM. It caches the group-mean RDM
plus the mean of the individually standardized rank vectors, a sufficient statistic
for the average Spearman correlation. This keeps the corrected estimand without
multiplying RSA memory by the number of subjects or macaques.

## Schema 3 and review surface

Schema 3 makes normalized eligibility auditable and intentionally invalidates schema-2
encoding checkpoints. Re-run the runner to produce the new results; do not append new
columns onto an old checkpoint by hand.

Encoding CSVs retain the raw columns and add:

```text
noise_ceiling_r
noise_ceiling_threshold
mean_noise_normalized_r
std_noise_normalized_r
mean_noise_normalized_r2
std_noise_normalized_r2
n_ceiling_targets
```

RSA CSVs include `spearman_rho`, `spearman_rho_individual_mean`,
`noise_ceiling_low`, `noise_ceiling_high`, `noise_normalized_spearman_rho`, and
`noise_normalized_spearman_rho2`.

Review raw and normalized sibling heatmaps together. The normalized heatmap has a
reference ceiling of `1.0`; the raw heatmap retains its ceiling row. Triple-N encoding
columns must be coarse `area_label` values only.

## Fault-tolerant validation checklist

1. Confirm Algonauts raw schema-3 correlations match the same schema-2 run within
   floating-point tolerance. Triple-N keeps the schema-2 area grouping but is not
   directly comparable because schema 3 filters to the reliable-unit population.
2. Assert every ceiling vector exactly matches the response target order and length.
3. Verify a toy case: `r=0.4`, `NC=0.25` gives signed `0.8`, squared `0.64`.
4. Verify a negative case: `r=-0.4`, `NC=0.25` gives signed `-0.8`, squared `0.64`.
5. Verify no clipping: `r=0.8`, `NC=0.25` gives signed `1.6`, squared `2.56`.
6. Verify zero, negative, missing, and below-threshold ceilings are excluded from the
   normalized mean and counted out, not converted to zero.
7. Confirm Triple-N `n_targets` and normalized counts both include only
   `reliability_best >= 0.4` and match one another.
8. Inspect the interactive HTML companion before accepting a results run.

## Runtime boundary

Python processes already running when this code was changed retain their imported old
modules. They are not interrupted, modified, or retroactively normalized. Schema-3
area-label grouping and normalization begin on the next runner invocation. A fresh
schema-3 area-label run can subsequently resume exact completed models with
`--resume`; schema-2 and retired region/quality checkpoints are deliberately rejected.

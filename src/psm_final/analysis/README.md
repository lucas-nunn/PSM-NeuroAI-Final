**classes and methods for conducting RSA and encoding analyses**

- [model.py](./model.py): generic base class which handles RDM construction and comparison with datasets agnostic of model type
- [beta_vae_analysis](./beta_vae_analysis.ipynb): class for conducting analyses of beta VAEs
- [correlating.py](./correlating.py): RDM, noise-ceiling, and shared noise-normalization helpers
- [encoding.py](./encoding.py): nested-CV neural encoding; Triple-N is grouped by coarse `area_label` with `reliability_best >= 0.4`
- [runner.py](./runner.py): resumable RSA/encoding orchestration and raw + normalized review artifacts

Raw correlations are always retained. Encoding reports signed `r / sqrt(NC)` and
Algonauts-style `r² / NC`; RSA keeps the historical correlation against the
group-mean RDM and separately reports the mean model-to-individual rho used for
normalization by the matched upper ceiling. See
[`wiki/encoding-rsa-noise-normalization.md`](../../../wiki/encoding-rsa-noise-normalization.md)
for the scientific contract and validation checklist.

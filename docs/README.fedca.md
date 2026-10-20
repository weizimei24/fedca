# FedCA: compatibility-aware absorption (zsRE, Qwen3-1.7B, q/v LoRA r=8)

## The method in this codebase

| Method step | Where | What runs |
|---|---|---|
| 1. Protection signal from the global model | `utils/ca_utils.py: not_true_kl` | The downloaded global adapter is a frozen teacher. On every supervised answer position (answer tokens + EOS), the gold token is removed from both distributions; the KL between what is left is the **excess change**. Raising the client's own answer costs nothing; reshaping the rest of what the global model would say costs. |
| 2. Compatibility-constrained local learning | `alg/fedca.py: CompatibilityTrainer` | FedIT's local loop, unchanged, with loss = answer CE + `ca_lambda` × excess. The teacher forward pass swaps the global LoRA in (`LoraSwap`), runs in eval mode on the unpadded prefix, and swaps back. |
| 3. Compatibility-aware absorption | `alg/fedca.py: Server.aggregate` | `θ_g ← θ_g + η · Σ_i p_i (θ_i − θ_g)`, applied per LoRA factor. For each candidate η, every client scores the **merged** update on a fixed subset of its own training rows. The server takes the η with the largest mean answer-CE gain. |

The defaults are in `config.zsre_fedca.yaml`. Everything else (data, seed, rounds, optimizer, 63 local steps, rank 8 on q/v) is identical to `config.zsre_fedit.yaml`.

## Implementation choices and why

1. **Step 3 scores the joint update, not per-client scalars.** The framework said "clients report how much they deviate; the server absorbs compatible updates more". Two problems made that weak in practice:
   - Compatibility is a property of the *combination*. An update can look harmless on its own and still interfere once merged.
   - Down-weighting a client lowers how much of *that client's* knowledge gets absorbed, which directly lowers macro EM.

   Scoring the merged update on every client's own data measures exactly "does absorbing this much still help everyone". The per-client version is still available as `ca_beta` (weights `∝ p_i (mean_excess / excess_i)^β`).

2. **The strict excess guard is off by default (`ca_kappa: inf`).** With κ = 1, the merged excess may not exceed a single client's own excess. Your diagnostics show client updates largely share the same generic changes (functional footprint ≈ 0.96), and in that regime the plain average already reaches the limit, so the guard pins η ≈ 1. The tiny-model smoke run did exactly this.

   The CE gain already charges cross-client harm, because it includes the EOS token and is measured on every client's data. The guarded variant runs in the core comparison as `fedca_k1`.

3. **Absorption is per factor, with no SVD.** The global model stays a rank-8 LoRA by construction, and at η = 1 the code path *is* FedIT. FedIT is also the strongest existing aggregator on this setup (9.45 vs FlexLoRA 9.40, FedFH 8.63), so there was no reason to change the aggregator itself.

4. **The teacher pass runs on the unpadded prefix.** zsRE rows are right-padded to 512 but have about 50 real tokens, and right padding cannot affect earlier positions under causal attention (there is a test for this). This keeps the teacher pass cheap. The student pass is FedIT's, unchanged.

5. **No test data anywhere in training or step selection.** Probe rows are a fixed, seeded subset of each client's *training* file. Test questions are read only by the evaluators, after training.

## Correctness checks

`python -X utf8 -m unittest tests.test_fedca -v` runs 17 CPU tests on a tiny random Qwen3:

- the KL matches a brute-force reference; it is 0 for identical models and for gold-only changes; its gradient reaches the student only;
- the swap restores weights exactly, and trimming preserves the answer logits;
- the probe returns zero excess for an identical candidate and restores model state;
- `select_eta` and the FedExP limits behave as specified;
- the constraint lowers local excess while the client still learns;
- every server mode completes full rounds;
- **FedCA with `ca_lambda 0 --ca_eta_mode fixed --ca_eta 1` is bitwise identical to `alg/fedit.py` over two rounds.** The same equality was also checked through `main.py` end to end.

## Running

```powershell
# core comparison: 7 runs x 10 rounds (FedIT itself is not rerun; see above)
python -X utf8 -u tools/run_fedca_suite.py --stage core --rounds-eval 2,4,6,8
# does the constraint raise the safe absorption step?
python -X utf8 -u tools/run_fedca_suite.py --stage mechanism
python -X utf8 -u tools/run_fedca_suite.py --summary
```

To finish an interrupted core experiment without retraining completed rounds:

```powershell
python -X utf8 -u tools/run_fedca_suite.py --only fedexp --resume-training --rounds-eval 2,4,6,8
```

This loads the latest `round_XXX/lora_weights.pt`, keeps the original round
index for learning-rate decay and client RNG/shuffling, then runs the usual
final and saved-round evaluations. Local optimizers are recreated every round
in the original trainer, so no optimizer checkpoint is needed at this boundary.
Only full-participation prototype runs with every round saved are supported.
Round records and weights must agree, and a failed continuation does not create
a final adapter. Original training logs are preserved; continuation logs have
separate timestamped names. Do not run this concurrently with the original job.

Each experiment writes to `exp/zsre8_chat_eos_v2/fedca_suite/<name>/`:

- `adapter/.../fedca_rounds.jsonl`: per round, the chosen η, the probe's gain and excess for every candidate, each client's train CE / excess / local gain, and the probe time;
- `adapter/.../round_XXX/lora_weights.pt`: the global adapter after XXX completed rounds;
- `evaluation/zsre/metrics.json` (round 10) and `evaluation/zsre/rounds/` (EM curve).

**Cost:** the extra teacher pass covers about 50 tokens per step, against FedIT's 512-token forward and backward. The probe adds |grid| + 1 short forward passes on 32 rows per client per round. Expect roughly 10–20% more time per round than FedIT. Each client uploads 2·|grid| + 3 extra scalars per round.

## Saved results

The [2026-10-15 result snapshot](exp/zsre8_chat_eos_v2/fedca_suite/README.md)
records completed evaluations and pending experiments. Configurations, training
diagnostics, logs, metrics and predictions are versioned; adapter weight files
remain local and are excluded from Git. A fresh clone therefore needs training
before evaluating these adapters; restart skipping only applies when the local
weight files are still present.

## Reading the results (decided before seeing them)

- **Main claim:** `fedca` at round 10 beats FedIT (9.45) **and** the best of `srv_eta2`, `srv_eta3` and `fedexp`. The fixed-η baselines are picked as the best value on test, which flatters them; that makes this a conservative test of FedCA.
- **Step 2's claim** ("compatible updates can be absorbed harder"): compare `ntd_eta{3,4,6}` with `srv_eta{3,4,6}`. Support means the constrained runs keep improving, or degrade less, at η where the unconstrained runs stall or collapse. The probe's chosen η should also be larger in `fedca` than in `probe_only`.
- **Step 3's claim:** `fedca` > `ntd_only`, and `probe_only` ≥ the best `srv_eta`.
- Report every run, including the sensitivity stage. Do not pick λ or κ on test. These are single-seed results; rerun the top configurations with two more seeds before claiming a difference smaller than about 1 EM.

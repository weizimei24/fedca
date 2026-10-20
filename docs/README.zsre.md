# zsRE factual-knowledge consolidation sanity benchmark

This extension leaves FedIT and FedRotLoRA unchanged. It creates eight
equal-sized semantic clients from public ROME/MEND zsRE records. Train files
supervise `src -> alt`; client-local test files evaluate `rephrase -> alt`.
The original `answers[0]` is retained only in metadata and is never a target.

Install only the preprocessing additions (this does not upgrade PyTorch or
Transformers):

```powershell
python -m pip install -r requirements-zsre.txt
```

Run the disposable offline smoke test before downloading data:

```powershell
python -X utf8 dataset/smoke_zsre.py
```

Build the actual dataset and its centralized pool:

```powershell
python -X utf8 dataset/download_zsre.py --output_dir dataset/zsre/raw
python -X utf8 dataset/process_zsre.py --raw_dir dataset/zsre/raw --output_dir dataset/zsre --num_clients 8 --samples_per_client 500 --seed 42
python -X utf8 dataset/validate_zsre.py --data_dir dataset/zsre --num_clients 8 --samples_per_client 500
python -X utf8 dataset/build_zsre_pooled.py --data_dir dataset/zsre --num_clients 8 --seed 42
```

If a spectral cluster has fewer than 500 valid examples, preprocessing prints
all eight raw cluster sizes and uses their common minimum without oversampling.
In that case run validation without `--samples_per_client 500` and use the
recorded `selected_samples_per_client` in `dataset/zsre/stats.json`.
The default clusters a deterministic 10,000-record candidate pool, matching
the common zsRE 10k scale while keeping semantic clustering practical. This
is recorded in `stats.json`; use `--max-clustering-candidates 0` only when
resources permit clustering every valid raw record.

The default is deterministic semantic MiniBatchKMeans on the public
`all-MiniLM-L6-v2` subject embeddings. It is chosen because a nearest-neighbor
spectral graph on zsRE's large set of mostly unique entity names can fragment
and collapse into one giant cluster; no artificial client balancing is applied.
For the requested spectral variant, pass `--clustering-method spectral`; its
raw cluster sizes are still reported and the common-minimum rule still applies.

Run every experiment with the same client files. `eval_zsre.py` performs the
official final comparison with deterministic greedy generation, all test rows,
normalized EM, token F1, and containment. The source prompt and generation
configuration are identical for all four methods.

```powershell
# Base (no training)
python -X utf8 eval_zsre.py --config config.zsre_base.yaml --base-only --method base

# Local-only
python -X utf8 main.py --config config.zsre_localonly.yaml
python -X utf8 eval_zsre.py --config config.zsre_localonly.yaml --method local

# Centralized pooled LoRA
python -X utf8 main.py --config config.zsre_centralized.yaml
python -X utf8 eval_zsre.py --config config.zsre_centralized.yaml --method centralized

# Federated FedIT, all eight clients participate each round
python -X utf8 main.py --config config.zsre_fedit.yaml
python -X utf8 eval_zsre.py --config config.zsre_fedit.yaml --method fedit

# FedRotLoRA (same data and optimisation; only its existing aggregation differs)
python -X utf8 main.py --config config.zsre_fedrotlora.yaml
python -X utf8 eval_zsre.py --config config.zsre_fedrotlora.yaml --method fedrotlora

# Comparison table (EM is the primary metric)
python -X utf8 summarize_stage1.py --run zsre8_stage1 --metric em
```

Results are isolated below `exp/zsre8_stage1/{base,local,centralized,fedit}`.
Each evaluation writes auditable per-client prediction JSONL plus a metrics
file. `summarize_stage1.py` writes `stage1_summary_em.csv`, including
`LocalGain = Local - Base` and `ConsolidationGap = Local - FedIT` per client
and as an unweighted macro average.

## Corrected chat/EOS rerun

The zsRE formatter now uses Qwen3's chat template with `enable_thinking=False`
and a short-answer instruction. Training supervises the full target plus EOS;
prompt and padding tokens are masked. Evaluation uses the identical prompt
encoder, greedy generation and the unchanged 32-token limit. EM/F1 remain
unchanged. Containment now requires a whole-token target span in the prediction
(so `male` does not match `female`). EOS/max-length rates and token counts are
saved with predictions and metrics.

Use the CUDA-enabled interpreter in this workspace:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 dataset/check_zsre_protocol.py --gpu-smoke
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u run_zsre_sequence.py --run zsre8_chat_eos_v2
```

The sequence re-evaluates Base, then trains/evaluates Local-only, FedIT and
FedRotLoRA, then summarizes EM. It preserves `exp/zsre8_stage1` and overrides
each YAML's old suffix with `exp/zsre8_chat_eos_v2/<method>`. It refuses an
existing sequence state, stops on any failure and validates every completed
evaluation as eight clients times 500 samples using the new protocol.

## RoLoRA extension

`alg/rolora.py` now begins with a B update (PEFT initializes B to zero),
alternates B/A thereafter, and retains both factors in the global adapter so
every client starts from the same full state and final evaluation loads both.
Its separate configuration matches FedIT's dataset, model, seed, rounds,
optimizer, and LoRA settings. Run it only after confirming the output directory
`exp/zsre8_chat_eos_v2/rolora` does not already contain a run:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u main.py --config config.zsre_rolora.yaml
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u eval_zsre.py --config config.zsre_rolora.yaml --method rolora
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 summarize_stage1.py --run zsre8_chat_eos_v2 --methods base,local,centralized,fedit,fedrotlora,rolora --centralized-dir centralized_roundwise --metric em --output stage1_summary_em_with_rolora.csv
```

The final evaluation uses all eight 500-example test groups. The six-method
summary adds `rolora_consolidation_gap = local - rolora` without changing
previous summary files.

## FLoRA extension

`alg/flora.py` stacks client LoRA factors with data-size weights and applies
PEFT's `lora_alpha / lora_rank` scaling when merging the update into the base
weights. Because the generic adapter checkpoint stores only LoRA factors,
FLoRA also saves its updated base layers as `merged_weights.pt`; the zsRE
evaluator reloads both files. The separate configuration uses the same zsRE
split, model, seed, rounds, optimizer, and rank-8 LoRA budget as FedIT. Check
that `exp/zsre8_chat_eos_v2/flora` is unused before starting:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u main.py --config config.zsre_flora.yaml
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u eval_zsre.py --config config.zsre_flora.yaml --method flora
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 summarize_stage1.py --run zsre8_chat_eos_v2 --methods base,local,centralized,fedit,fedrotlora,flora --centralized-dir centralized_roundwise --metric em --output stage1_summary_em_with_flora.csv
```

Run each command only after the previous one exits successfully. Final
evaluation covers all eight 500-example test groups. The summary adds
`flora_consolidation_gap = local - flora` without overwriting older results.
Keep `merged_weights.pt` with `lora_weights.pt` for subsequent FLoRA evaluation.

## FedEx-LoRA, FFA-LoRA, and FlexLoRA extensions

All three new configurations use the same eight zsRE clients, Qwen3-1.7B,
seed, ten rounds, local training budget, rank 8, and corrected chat/EOS
generation protocol as FedIT. Run from the repository root. First verify each
target directory below `exp/zsre8_chat_eos_v2/` is unused; do not rerun a
completed method into the same directory.

FedEx-LoRA uses weighted A/B aggregation and merges the exact-aggregation
residual into base weights. Its evaluation requires both `lora_weights.pt`
and `merged_weights.pt`:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u main.py --config config.zsre_fedexlora.yaml
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u eval_zsre.py --config config.zsre_fedexlora.yaml --method fedexlora
```

FFA-LoRA trains and uploads only B, while retaining the original shared A
in the final checkpoint for deterministic evaluation:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u main.py --config config.zsre_ffalora.yaml
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u eval_zsre.py --config config.zsre_ffalora.yaml --method ffalora
```

FlexLoRA reconstructs the data-weighted product and redistributes its best
rank-8 approximation by SVD. This configuration keeps all client ranks at 8
for a matched-budget ablation; it does not exercise FlexLoRA's heterogeneous-
rank capability:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u main.py --config config.zsre_flexlora.yaml
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u eval_zsre.py --config config.zsre_flexlora.yaml --method flexlora
```

Run each evaluation only after its training command succeeds. When all three
evaluations have finished, write a separate EM comparison without replacing
the previous summary:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 summarize_stage1.py --run zsre8_chat_eos_v2 --methods base,local,centralized,fedit,fedrotlora,fedexlora,ffalora,flexlora --centralized-dir centralized_roundwise --metric em --output stage1_summary_em_extra_lora.csv
```

## ILoRA extension

The project's `alg/ilora.py` implements **ILoRA**, not **iFLoRA**. It now
initializes the common rank-8 LoRA subspace once using QR, keeps the
compensated base weights fixed throughout training, and uses concatenated QR
product aggregation. Both those base weights and final LoRA factors are
required for evaluation. This is a homogeneous-rank ILoRA-style QR baseline,
**not** a complete reproduction of the rank-heterogeneous paper setting or
ILoRA-S with control variates. Check that `exp/zsre8_chat_eos_v2/ilora` is
unused, then run:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u main.py --config config.zsre_ilora.yaml
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u eval_zsre.py --config config.zsre_ilora.yaml --method ilora
```

After ILoRA and the other three additional methods have all been evaluated:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 summarize_stage1.py --run zsre8_chat_eos_v2 --methods base,local,centralized,fedit,fedrotlora,fedexlora,ffalora,flexlora,ilora --centralized-dir centralized_roundwise --metric em --output stage1_summary_em_extra_lora_ilora.csv
```

## FedFH extension (ours)

`alg/fedfh.py` keeps FedIT's local training unchanged. After training, each
client runs one gradient-free pass over its training rows and uploads, per
decoder layer, the top-`fh_k` eigenpairs of the LoRA input second moment
(q_proj and v_proj read the same input and share it), its trace and the
(k+1)-th eigenvalue. The server returns the optimal rank-8 LoRA of the
input-geometry-weighted objective via a whitened truncated SVD, re-gauged to
FlexLoRA's factor convention; base weights are never modified, so evaluation
loads only `lora_weights.pt`. `fh_gamma -> infinity` reduces exactly to
FlexLoRA, so FlexLoRA is the primary comparison. Per-round diagnostics are
appended to `fh_diagnostics.jsonl` next to the adapter.

Offline checks (CPU, tiny random Qwen3, no downloads):

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -m unittest tests.test_fedfh -v
```

Check that `exp/zsre8_chat_eos_v2/fedfh` is unused, then:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u main.py --config config.zsre_fedfh.yaml
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 -u eval_zsre.py --config config.zsre_fedfh.yaml --method fedfh
```

If FlexLoRA has not been evaluated yet, run it first (see above), then:

```powershell
& 'D:\sofrware\anaconda3\envs\wzm\python.exe' -X utf8 summarize_stage1.py --run zsre8_chat_eos_v2 --methods base,local,centralized,fedit,flexlora,fedfh --centralized-dir centralized_roundwise --metric em --output stage1_summary_em_with_fedfh.csv
```

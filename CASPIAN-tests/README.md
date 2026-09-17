# CASPIAN-tests

Artifacts for the CASPIAN validation/evaluation study (MMLU-Pro, 8 agents,
0 malicious agents). The full write-up is in [`FINAL-REPORT.md`](FINAL-REPORT.md).

```
configs/    experiment YAML configurations (generation, evaluation, HPS)
data/       generated MMLU-Pro datasets (TrainDataGeneration.py output)
results/    MainEvaluation result JSONs, offline replay summaries, sweeps
logs/       captured stdout/stderr of every generation/evaluation run
snapshots/  frozen CASPIAN implementations (v0 baseline ... v5 final) + hashes
analysis/   evaluation-only helper scripts (replay, math audit, sweeps)
reports/    auxiliary reports (links to FINAL-REPORT.md)
```

## Versions

| File | Description |
| --- | --- |
| `snapshots/CASPIAN-v0-baseline.py` | baseline as shipped (`sha256:492350db...`) |
| `snapshots/CASPIAN-v1-instant-onset.py` | Algorithm-1 instant timing only |
| `snapshots/CASPIAN-v2-copula-partial.py` | Appendix-C partial-correlation copula only |
| `snapshots/CASPIAN-v3-v2-no-novelty.py` | v2 without the novelty factor (rejected) |
| `snapshots/CASPIAN-v4-raw-lambda1.py` | v1 + raw-matrix lambda1 growth (rejected) |
| `snapshots/CASPIAN-v5-onset-copula.py` | final (`sha256:7169821a...`) |

`defense-models/CASPIAN.py` currently holds v5.

## Key commands

```bash
# data generation
python TrainDataGeneration.py CASPIAN-tests/configs/generation-mmlupro-8a-0m.yaml

# baseline / final evaluation
python MainEvaluation.py CASPIAN-tests/configs/evaluation-caspian-baseline.yaml
python MainEvaluation.py CASPIAN-tests/configs/evaluation-caspian-onset-copula.yaml

# HPS sensitivity
python MainEvaluation-search.py CASPIAN-tests/configs/hps-caspian-emadecay.yaml

# offline replay on fixed generated data
python CASPIAN-tests/analysis/replay_caspian.py --pkl CASPIAN-tests/data/MMLUPro-8a-0m-seed500.pkl \
    --impl CASPIAN-tests/snapshots/CASPIAN-v5-onset-copula.py \
    --out CASPIAN-tests/results/EXP-00-replay-v5-onset-copula.json --dump-records

# formula audit
python CASPIAN-tests/analysis/audit_math.py --impl defense-models/CASPIAN.py
```

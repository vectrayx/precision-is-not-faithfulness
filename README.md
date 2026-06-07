# Precision Is Not Faithfulness

**Precision Is Not Faithfulness: Coverage-Aware Evaluation of Grounded Generation with a
Complete Oracle** — NLP project for EMNLP/Findings.

Reference-free faithfulness metrics measure only **precision** and reward *abstention*
(a model scores high by saying little). With a **complete** structured oracle we also
measure **recall** (coverage of the facts that mattered); requiring coverage inverts the
model ranking. Validated in two domains (F1 telemetry, NOAA weather), multilingual EN/ES/PT.

> Not race prediction. The object of study is **language** (NLG + faithfulness).

## Structure
```
src/data/    FastF1 pipeline -> structured strategic events
src/eval/    faithfulness metric
src/models/  baselines and fine-tuning
data/        raw (gitignored) / structured / annotations
experiments/ runs and results
paper/       manuscript
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # fastf1, pandas, numpy
```

## End-to-end (offline, no API key / GPU)
```bash
# 1. Build structured ground truth (one race or a whole season)
python -m src.data.build_dataset --year 2024 --gp Monza
python -m src.data.build_dataset --year 2024 --season

# 2. Build benchmark instances (EN/ES/PT)
python -m src.data.segments

# 3. Run the pilot (template baselines) + metric-validation experiment
python experiments/run_pilot.py            # -> experiments/results/

# 4. Tests for the faithfulness metric
python tests/test_faithfulness.py

# 5. Build the paper (auto-fills tables from real results)
cd paper && make
```

## Cloud runs (use credit — see scripts/)
```bash
pip install -r requirements-cloud.txt
bash scripts/setup_azure_openai.sh         # frontier baseline endpoint
python experiments/run_pilot.py --generators azure_openai --lang en
bash scripts/gcp_gpu_vm.sh                  # GPU VM for serving + fine-tuning
python src/models/build_sft.py && python src/models/finetune.py
```


## Data & licensing
Raw F1/FOM data is **not redistributed**. We release only code, derived structured data,
and annotations. See the paper's ethics section.

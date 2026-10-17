# LowRes-EffTransNAS

**Neural Architecture Search for Efficient Transformer Models in Low-Resource NLP**

> Big Data Mining and Analytics

Complete research codebase with **experiment runner**, **agentic NAS agent**, **Jupyter notebooks**, and **ablation study** scripts.

---

## 1. Quick Start

```bash
pip install -r requirements.txt

# Classic experiment pipeline
python run_experiment.py --stage all --exp-name demo --pop-size 20 --generations 10

# Agentic research agent (autonomous multi-stage loop)
python -m agents.nas_agent \
    --goal "models under 15M params with accuracy > 85" \
    --exp-name agentic_run \
    --max-steps 6 \
    --pop-size 20 \
    --generations 8

# Ablation study
python research/ablation_study.py --exp-name ablation_v1
```

---

## 2. Experiment Interface

```bash
python run_experiment.py --stage all|setup|search|report|eval \
                         --exp-name NAME \
                         --pop-size 24 \
                         --generations 12 \
                         --seed 42 \
                         --device cpu
```

Produces:

```
experiments/<name>/
├── experiment_meta.json
├── pareto_front.json
├── search_history.json
├── REPORT.md
└── checkpoints/
```

---

## 3. Agentic Research Agent

An autonomous agent that **observes → plans → executes → reflects** until the research goal is met.

```bash
python -m agents.nas_agent \
    --goal "Discover efficient Transformers under 15M params with accuracy > 85" \
    --exp-name agentic_run \
    --max-steps 8 \
    --target-params 15 \
    --target-acc 85
```

Agent actions: `run_search`, `run_ablation`, `run_analysis`, `write_report`, `finish`.

Artefacts include `agent_memory.json`, `ablation.json`, `analysis.json`, `AGENT_REPORT.md`.

---

## 4. Jupyter Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/01_explore_search_space.ipynb` | Sample & visualise the discrete search space |
| `notebooks/02_run_evolutionary_search.ipynb` | Interactive NSGA-II search + Pareto plots |
| `notebooks/03_agentic_analysis.ipynb` | Analyse agent memory & sweet-spot models |

```bash
jupyter notebook notebooks/
```

---

## 5. Repository Layout

```
LowRes_EffTransNAS/
├── run_experiment.py              # main experiment CLI
├── agents/
│   └── nas_agent.py               # agentic research agent
├── models/
│   ├── attention.py               # softmax / linear / sparse-local
│   └── transformer.py             # searchable encoder
├── nas/
│   ├── search_space.py
│   └── evolutionary.py            # multi-objective EA
├── training/
│   ├── progressive_shrinkage.py
│   └── trainer.py
├── evaluation/
│   └── metrics.py
├── research/
│   └── ablation_study.py
├── notebooks/
│   ├── 01_explore_search_space.ipynb
│   ├── 02_run_evolutionary_search.ipynb
│   └── 03_agentic_analysis.ipynb
├── configs/default.yaml
├── utils/logging_utils.py
└── experiments/                   # all run outputs
```

---

## 6. Search Space (paper)

| Dimension   | Choices |
|-------------|---------|
| Depth       | 4, 6, 8, 10, 12 |
| Hidden dim  | 128 … 768 |
| Heads       | 2, 4, 6, 8, 12 |
| FFN ratio   | 1.0 … 4.0 |
| Attention   | softmax, linear, sparse-local |

---

## 7. Citation

```bibtex
@article{lowres_efftransnas,
  title={Neural Architecture Search for Efficient Transformer Models in Low-Resource NLP},
  journal={Big Data Mining and Analytics},
  year={2024}
}
```

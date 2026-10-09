# RATO — Independent Base-Paper Reproduction

**Status:** Runnable independently authored research simulator and baseline training code; **not** the original authors' implementation and **not** a verified exact numerical reproduction.

Base paper: Xiaoxuan Hu et al., *Joint Resource Allocation and Task Offloading for Heterogeneous Cloud-Edge-End Networks Assisted by NOMA*, IEEE Transactions on Green Communications and Networking, 10 (2026), DOI **10.1109/TGCN.2026.3653056**.

The paper specifies 20 UEs, 5 edge servers, 1 cloud server, NOMA with SIC, a Digital Twin, and DT-MADDPG with edge-load balancing. This repository includes the executable environment, five learning algorithms, throughput baselines, evaluation utilities, tests, plots, and time-stamped output folders.

**Crucial scientific limitation:** the paper omits several task distributions, neural architecture dimensions, learning rates, link parameters, authentication costs, and reward coefficients. This repository MUST NOT be described as an exact reproduction or the authors' original code. See `docs/assumptions.md` before relying on research results. The physical system and authentication are simulated; there is no real NOMA radio or cryptographic authentication service.

## 1. Platform

Designed for Ubuntu 24.04 and Python 3.11. The paper reports Python 3.11.5, PyTorch 2.1.2, and CUDA 12.1. The GPU is optional for small tests. A workstation with NVIDIA RTX 2000 Ada, 64 GiB RAM, and Xeon CPU is sufficient for this scale, although full multi-agent runs can take considerable time.

## 2. Install

```bash
cd RATO
# Recommended: install conda or miniforge externally if not already available.
conda create -n rato python=3.11.5 -y
conda activate rato
python -m pip install --upgrade pip
# GPU PyTorch, original paper version (requires a compatible NVIDIA driver):
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/check_system.py
python -m pytest -q
```

For CPU-only use the officially supplied CPU build of PyTorch; GPU and CPU testing can give small floating-point differences.

## 3. Run the simulator and quick training

```bash
python scripts/generate_data.py --samples 100 --seed 42
python scripts/train.py --algorithm dt_maddpg --episodes 2 --steps 3 --num-ues 3 --num-edge 2 --batch-size 2 --device cpu
```

The quick test overrides paper node counts and exists only as a sanity check. **Do not use these settings for publication results.** For full paper-scale training, use:

```bash
python scripts/train.py --algorithm dt_maddpg --seed 42 --device auto
python scripts/train.py --algorithm maddpg --seed 42 --device auto
python scripts/train.py --algorithm ddpg --seed 42 --device auto
python scripts/train.py --algorithm d3qn --seed 42 --device auto
python scripts/train.py --algorithm madqn --seed 42 --device auto
```

The default is 400 episodes × 12 steps (an **assumption**, not reported paper settings), so expect appreciable compute time. Start with one small run before scheduling parallel trainings. All algorithms run on the same simulator modules, and joint transitions are stored together.

## 4. Evaluate trained model

Use a real checkpoint path emitted by the train script:

```bash
python scripts/evaluate.py --algorithm dt_maddpg \
  --checkpoint results/<RUN>/checkpoints/latest.pt --episodes 10 --seed 10000
```

A checkpoint trained with `--num-ues` or `--num-edge` overrides is not compatible with the default paper-sized evaluation command; train the full-scale model to evaluate it here.

## 5. Reproduce communication plots (paper Figs. 3–5)

```bash
python scripts/run_experiments.py --kind throughput --seed 42
python scripts/reproduce_figures.py --input results/<RUN>/raw_metrics/throughput.csv
```

The generated numbers are actual calculations from the implemented assumptions, not digitizations or prefilled paper curve values. NOMA and OMA comparisons depend on the OMA bandwidth/power fairness assumptions specified in `docs/assumptions.md`.

## 6. Reproduce learned-policy comparisons (paper Figs. 6–10)

Train the listed agents and supply checkpoints before running the comparisons:

```bash
python scripts/run_experiments.py --kind algorithms --checkpoints \
  ddpg=results/<DDPG_RUN>/checkpoints/latest.pt \
  d3qn=results/<D3QN_RUN>/checkpoints/latest.pt \
  madqn=results/<MADQN_RUN>/checkpoints/latest.pt \
  maddpg=results/<MADDPG_RUN>/checkpoints/latest.pt \
  dt_maddpg=results/<DT_RUN>/checkpoints/latest.pt

python scripts/run_experiments.py --kind schemes --checkpoints \
  dt_maddpg=results/<DT_RUN>/checkpoints/latest.pt

python scripts/run_experiments.py --kind task_sweep --checkpoints \
  ddpg=results/<DDPG_RUN>/checkpoints/latest.pt \
  d3qn=results/<D3QN_RUN>/checkpoints/latest.pt \
  madqn=results/<MADQN_RUN>/checkpoints/latest.pt \
  maddpg=results/<MADDPG_RUN>/checkpoints/latest.pt \
  dt_maddpg=results/<DT_RUN>/checkpoints/latest.pt
```

Then pass each generated CSV to `scripts/reproduce_figures.py`. The figure scripts intentionally **do not invent results**. Scheme comparisons use the same learned policy while restricting destinations; to reproduce separately trained restricted-mode policies, train with `--mode ...` and evaluate each mode with its own checkpoint.

## 7. Folders

- `configs/`: paper-reported numerical settings, separate assumptions, training choices and experiment definitions.
- `src/rato/environment/`: synthetic physical system, Digital Twin, observations, action projection and unified simulator.
- `src/rato/communication/`: NOMA, OMA, power models, channel and authorization overhead.
- `src/rato/computation/`: device/edge/cloud execution times and energies.
- `src/rato/optimization/`: task costs, constraints and edge-load balancing.
- `src/rato/algorithms/`: DDPG, D3QN, MADQN, MADDPG, DT-MADDPG and shared neural/replay logic.
- `src/rato/utils/`: reproducible seeding, devices, CSV/JSON and model checkpoints.
- `scripts/`: training, dataset creation, analytical experiments, evaluation and figures.
- `tests/`: mathematical, structural and train-update tests.
- `results/`: locally persisted run data (ignored by Git).
- `docs/`: model, assumptions, workflow and caveats.
- `papers/`: citation only; your licensed PDF stays on your machine.
- `.github/workflows/`: CPU unit-test automation for GitHub.

## 8. GitHub push

```bash
git init -b main
git add .
git commit -m "Initial RATO independent reproduction"
git remote add origin git@github.com:YOUR_USERNAME/RATO.git
git push -u origin main
```

Create the empty remote repository first. Do not commit licensed PDFs, passwords or bulky results. Use `git switch -c feature/name` for subsequent work.

## 9. Output integrity

Every run saves `config/resolved_config.yaml`, `config/run_metadata.json`, checkpoints, raw CSV, and JSON metrics. The commit is recorded when Git is initialized. Results are generated at runtime; **no benchmark performance is preclaimed**. Differences from the paper are to be studied, not hidden.

### Overlay reward-convergence curves (Fig. 7)

After training all algorithms, overlay their **actual** episode CSV histories:

```bash
python scripts/reproduce_figures.py --figure fig7 --training-csv \
  DDPG=results/<DDPG_RUN>/raw_metrics/training.csv \
  D3QN=results/<D3QN_RUN>/raw_metrics/training.csv \
  MADQN=results/<MADQN_RUN>/raw_metrics/training.csv \
  MADDPG=results/<MADDPG_RUN>/raw_metrics/training.csv \
  DT-MADDPG=results/<DT_RUN>/raw_metrics/training.csv
```

For Figure 6, train DT-MADDPG separately with `--mode edge_only`, `--mode cloud_only` and `--mode no_dt_deviation`, then supply the four `training.csv` paths with `--figure fig6`. A scheme comparison from the same fully trained policy is provided as an alternative but does not reproduce separate-training convergence curves.

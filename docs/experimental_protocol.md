# Experiment protocol

1. Fix Git commit and Python/CUDA dependency versions.
2. Run unit tests and a small CPU training smoke test.
3. Keep paper parameters unmodified for primary comparison.
4. Run each algorithm with the same workload seeds; repeat independent seeds (e.g., 0–4).
5. Train on specified seeds, evaluate on separate held-out seeds (e.g. 10000–10009).
6. Perform Fig. 3–5 analytical communication sweeps, and compute Fig. 6–10 performance only from trained model rollouts.
7. Report reward, mean latency (s), mean energy (J), QoS ratio (dimensionless), load variance, deadline violations, success rate, and standard deviation across seeds.
8. Never manually rescale or shift curves to match the published graphs. If curves differ, record what assumptions may explain the difference.

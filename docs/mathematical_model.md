# Mathematical model

Use the original paper for the complete derivation. `docs/paper_equations.md` maps major equations to code. Key models:

- Partial offloading: local bytes/cycles `(1-gamma)*(D,C)` and offloaded `gamma*(D,C)`.
- NOMA cluster k: lambda sum=1, `p_k=lambda_k P`, ascending |h|², weaker-user interference `sum_{i<m}|h_i|² p_k`, rate `B log2(1+SINR)`.
- Communication delay equals transmitted bits / link bitrate. Collaboration includes relay delay and authorization overhead.
- CPU execution: `T=C/f_actual`; `E=capacitance*C*f_actual²`, with disclosed units/coefficients.
- Parallel local and offloaded computation: `T=max(T_local,T_remote)`.
- Cost: sum across UEs of `w_time*T + w_energy*E`, with weight construction listed in assumptions.
- Edge utilization: summed task cycles executed on ES k divided by ES effective cycles per slot. Variance across K utilizations quantifies imbalance.
- Reward: negative cost minus deadline and balancing penalties.
- Paper QoS indicator `(T/E)/(T_max/E_max)` reported separately.

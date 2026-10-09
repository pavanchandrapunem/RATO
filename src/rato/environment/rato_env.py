"""Multi-agent synthetic NOMA+DT cloud-edge-end simulator.

API reset(seed)->(M,obs_dim), step((M,act_dim))->obs,reward,done,info.
This is an independent implementation with documented gap-filling assumptions.
"""
import numpy as np
from .physical_system import PhysicalSystem
from .digital_twin import DigitalTwin
from .state_builder import build_observations
from .action_handler import action_dimension, project_actions
from ..communication.noma import noma_rates
from ..communication.oma import oma_rates
from ..communication.transmission import transmit_time, transmit_energy
from ..communication.authentication import AuthenticationModel
from ..computation.local import execution
from ..optimization.cost import dynamic_weights
from ..optimization.load_balancing import utilization, balancing_indicator
from ..optimization.reward import compute_rewards

class RATOEnv:
    def __init__(self, config):
        self.cfg = config
        self.M = config['paper']['network']['num_ues']
        self.K = config['paper']['network']['num_edge_servers']
        self.action_dim = action_dimension(self.K)
        self.max_steps = int(config['environment']['max_steps'])
        self.reset(config['environment']['seed'])

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.physical = PhysicalSystem(self.cfg, self.rng)
        a = self.cfg['assumptions']['network']
        e = self.cfg['environment']
        self.dt = DigitalTwin(self.physical, self.rng,
                              a['digital_twin_relative_deviation_std'],
                              e['use_digital_twin'] and e['mode'] != 'no_dt_deviation')
        self.auth = AuthenticationModel(a['authentication_latency_s'], a['authentication_energy_j'])
        self.step_count = 0
        self.last_utilization = np.zeros(self.K)
        self.last_power_coeffs = np.ones(self.K) / self.K
        obs = self.observations()
        self.observation_dim = obs.shape[-1]
        return obs

    def observations(self):
        return build_observations(self.physical, self.dt, self.last_utilization, self.last_power_coeffs)

    def step(self, actions):
        P = self.cfg['paper']; A = self.cfg['assumptions']; E = self.cfg['environment']
        p, t, dt = self.physical, self.physical.tasks, self.dt
        M, K = self.M, self.K
        act = project_actions(actions, p.associations, K, E['mode'], E['power_policy'])
        gamma, route, dest = act['gamma'], act['routes'], act['collaboration']
        power = act['power_coefficients'] * P['communication']['total_power_w']
        rate_fn = noma_rates if E['access_method'] == 'noma' else oma_rates
        rates = rate_fn(p.channel_gains, p.associations, power,
                        P['communication']['bandwidth_hz'], P['communication']['noise_power_w'])
        off_bytes = gamma * t['data_bytes']
        off_cycles = gamma * t['cycles']
        local_cycles = (1.0 - gamma) * t['cycles']
        local_time, local_energy = execution(local_cycles, dt.ue_actual,
                                            P['computation']['ue_switched_capacitance'])
        radio_delay = transmit_time(off_bytes, rates)
        # Equation (14): share cluster signal power in proportion to |h_m|^2.
        ue_power = np.zeros(M)
        for k in range(K):
            ids = np.flatnonzero(p.associations == k)
            if len(ids):
                ue_power[ids] = power[k] * p.channel_gains[ids] / np.maximum(p.channel_gains[ids].sum(), 1e-15)
        radio_energy = transmit_energy(ue_power, radio_delay)
        radio_delay = np.where(off_bytes > 0, radio_delay, 0.0)
        radio_energy = np.where(off_bytes > 0, radio_energy, 0.0)
        edge_mask = route != 2
        actual_edge = np.where(route == 1, dest, p.associations)
        edge_load_cycles = np.bincount(actual_edge[edge_mask], weights=off_cycles[edge_mask], minlength=K)
        edge_capacity = dt.edge_actual * A['network']['slot_duration_s']
        ratios = utilization(edge_load_cycles, edge_capacity)
        balance = balancing_indicator(ratios)
        overload_count = int(np.sum(ratios > 1.0))
        balance_penalty = (A['objectives']['balance_penalty_coefficient'] * balance +
                           A['objectives']['overload_penalty_coefficient'] * overload_count)
        remote_time = np.zeros(M)
        remote_energy = np.zeros(M)
        auth_ok = np.ones(M, dtype=bool)
        for m in range(M):
            if off_cycles[m] <= 0:
                continue
            base_time = radio_delay[m]
            base_energy = radio_energy[m]
            if route[m] == 0:  # assigned local ES
                x, y = execution(off_cycles[m], dt.edge_actual[p.associations[m]], A['network']['edge_effective_capacitance'])
            elif route[m] == 1: # edge-to-edge with authentication overhead
                ok, auth_t, auth_e = self.auth.authenticate(m, int(dest[m]))
                auth_ok[m] = ok
                mig_t = transmit_time(off_bytes[m], A['network']['edge_edge_rate_bps'])
                base_time += mig_t + auth_t
                base_energy += transmit_energy(A['network']['edge_edge_link_power_w'], mig_t) + auth_e
                x, y = execution(off_cycles[m], dt.edge_actual[dest[m]], A['network']['edge_effective_capacitance'])
            else: # relay through assigned ES to cloud
                mig_t = transmit_time(off_bytes[m], P['communication']['edge_cloud_rate_bps'])
                base_time += mig_t
                base_energy += transmit_energy(A['network']['edge_cloud_link_power_w'], mig_t)
                x, y = execution(off_cycles[m], dt.cloud_actual, A['network']['cloud_effective_capacitance'])
            remote_time[m] = base_time + float(x)
            remote_energy[m] = base_energy + float(y)
        latency = np.maximum(local_time, remote_time)
        energy = local_energy + remote_energy
        wt, we = dynamic_weights(t['deadline_s'], t['energy_budget_j'])
        total_load = np.zeros(M)
        for k in range(K):
            ids = np.flatnonzero(edge_mask & (actual_edge == k) & (off_cycles > 0))
            if len(ids):
                total_load[ids] = ratios[k]
        success = ((latency <= t['deadline_s']) & auth_ok &
                   (total_load <= A['objectives']['capacity_failure_threshold']))
        reward = compute_rewards(latency, energy, wt, we, t['deadline_s'],
                                 balance_penalty,
                                 A['objectives']['deadline_penalty_coefficient'],
                                 A['objectives']['reward_scale'], E['load_balance_penalty'])
        qos = ((latency / np.maximum(energy, 1e-15)) /
               (t['deadline_s'] / np.maximum(t['energy_budget_j'], 1e-15)))
        info = dict(reward=float(np.mean(reward)), latency_s=float(np.mean(latency)),
                    energy_j=float(np.mean(energy)), balance=float(balance),
                    success_rate=float(np.mean(success)), qos=float(np.mean(qos)),
                    deadline_violation_rate=float(np.mean(latency > t['deadline_s'])),
                    throughput_mbps=float(np.sum(rates) / 1e6),
                    total_cost=float(np.sum(wt * latency + we * energy)),
                    edge_loads=ratios.tolist(), power_coefficients=act['power_coefficients'].tolist(),
                    per_ue_latency_s=latency.tolist(), per_ue_energy_j=energy.tolist())
        self.step_count += 1
        done = self.step_count >= self.max_steps
        p.sample_step()
        dt.sync()
        self.last_utilization = ratios
        self.last_power_coeffs = act['power_coefficients']
        return self.observations(), reward.astype(np.float32), done, info

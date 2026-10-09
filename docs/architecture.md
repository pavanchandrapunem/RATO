# Architecture

PhysicalSystem (UEs, ESs, cloud, random tasks/channels) -> DigitalTwin (virtual state and deviations) -> ObservationBuilder -> [DDPG|D3QN|MADQN|MADDPG|DT-MADDPG] -> ActionProjector -> NOMA/OMA UE-to-ES communication -> authentication+ES-to-ES or ES-to-cloud relay -> parallel local/remote computation -> latency/energy/edge occupancy -> reward, new physical and DT state -> replay buffer -> gradient update.

20 agents per-UE for MADDPG/DT-MADDPG; one global controller for DDPG/D3QN; per-UE DQN for MADQN. Observations are dimensionally normalized before neural networks. No GNN or feasibility projection is added to this base-paper reproduction.

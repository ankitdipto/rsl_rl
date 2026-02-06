# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixture-of-Experts Actor-Critic module with Gumbel-Softmax hard gating."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class MoEActorCritic(nn.Module):
    """Mixture-of-Experts Actor-Critic with hard gating via straight-through Gumbel-softmax.
    
    The gate network routes morphologies to specialized expert networks.
    Expert selection is made once per episode based on the morphology vector.
    
    Architecture:
        - Gate: MLP(morphology_dim -> num_experts) with Gumbel-softmax
        - Experts: N separate actor MLPs, each taking [observations, morphology] -> actions
        - Critic: Shared MLP taking privileged observations -> value
    """
    
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        num_morphology_obs: int = 11,
        num_experts: int = 4,
        gate_hidden_dims: list = [64, 32],
        actor_hidden_dims: list = [128, 64, 32],
        critic_hidden_dims: list = [128, 64, 32],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        routing_type: str = "soft",
        tau_initial: float = 1.0,
        tau_min: float = 0.1,
        tau_anneal_rate: float = 0.0001,
        load_balance_coef: float = 0.01,
        # Diversity loss schedule (used by PPO as an auxiliary term)
        diversity_coef_max: float = 1.0e-3,
        diversity_start_iter: int = 1200,
        diversity_ramp_iters: int = 300,
        diversity_eps: float = 1.0e-8,
        **kwargs,
    ):
        """Initialize the MoE Actor-Critic.
        
        Args:
            num_actor_obs: Total dimension of actor observations (INCLUDING morphology).
                The last `num_morphology_obs` dimensions are treated as the morphology vector.
            num_critic_obs: Dimension of critic observations.
            num_actions: Dimension of action space.
            num_morphology_obs: Dimension of morphology vector (default 11 for BALLU).
                Must be <= num_actor_obs.
            num_experts: Number of expert networks.
            gate_hidden_dims: Hidden layer dimensions for gating network.
            actor_hidden_dims: Hidden layer dimensions for each expert.
            critic_hidden_dims: Hidden layer dimensions for critic.
            activation: Activation function name.
            init_noise_std: Initial standard deviation for action noise.
            noise_std_type: Type of noise std ('scalar' or 'log').
            routing_type: Expert routing type ('soft' or 'hard').
                'soft': Temperature-scaled softmax weighted combination of all experts.
                'hard': Straight-through Gumbel-softmax one-hot selection.
            tau_initial: Initial temperature for Gumbel-softmax.
            tau_min: Minimum temperature for Gumbel-softmax.
            tau_anneal_rate: Rate of temperature annealing per step.
            load_balance_coef: Coefficient for load balancing loss.
        """
        if kwargs:
            print(
                "MoEActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        # Store configuration
        # num_actor_obs is the FULL observation dim (includes morphology at the end)
        self.num_actor_obs = num_actor_obs
        self.num_morphology_obs = num_morphology_obs
        self.num_base_obs = num_actor_obs - num_morphology_obs  # Obs without morphology
        self.num_experts = num_experts
        self.num_actions = num_actions
        self.load_balance_coef = load_balance_coef
        self.routing_type = routing_type

        # Diversity loss scheduling (executed outside PPO core loop).
        self.diversity_coef_max = float(diversity_coef_max)
        self.diversity_start_iter = int(diversity_start_iter)
        self.diversity_ramp_iters = int(diversity_ramp_iters)
        self.diversity_eps = float(diversity_eps)
        self.diversity_coef: float = 0.0
        
        assert routing_type in ("soft", "hard"), (
            f"routing_type must be 'soft' or 'hard', got '{routing_type}'"
        )
        assert num_morphology_obs <= num_actor_obs, (
            f"num_morphology_obs ({num_morphology_obs}) must be <= num_actor_obs ({num_actor_obs})"
        )
        
        # Temperature for Gumbel-softmax
        self.tau = tau_initial
        self.tau_initial = tau_initial
        self.tau_min = tau_min
        self.tau_anneal_rate = tau_anneal_rate
        
        # Resolve activation function
        activation_fn = resolve_nn_activation(activation)
        
        # ========== Gating Network ==========
        # Takes morphology vector, outputs expert logits
        gate_layers = []
        gate_input_dim = num_morphology_obs
        for hidden_dim in gate_hidden_dims:
            gate_layers.append(nn.Linear(gate_input_dim, hidden_dim))
            gate_layers.append(activation_fn)
            gate_input_dim = hidden_dim
        gate_layers.append(nn.Linear(gate_input_dim, num_experts))
        self.gate = nn.Sequential(*gate_layers)
        
        # ========== Expert Networks (Actors) ==========
        # Each expert takes [base_observations, morphology] -> actions
        # Since base_obs + morphology = full_obs, expert_input_dim = num_actor_obs
        expert_input_dim = num_actor_obs
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            expert_layers = []
            layer_input_dim = expert_input_dim
            for hidden_dim in actor_hidden_dims:
                expert_layers.append(nn.Linear(layer_input_dim, hidden_dim))
                expert_layers.append(activation_fn)
                layer_input_dim = hidden_dim
            expert_layers.append(nn.Linear(layer_input_dim, num_actions))
            self.experts.append(nn.Sequential(*expert_layers))
        
        # ========== Critic Network ==========
        # Shared critic taking privileged observations
        critic_layers = []
        critic_input_dim = num_critic_obs
        for hidden_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(critic_input_dim, hidden_dim))
            critic_layers.append(activation_fn)
            critic_input_dim = hidden_dim
        critic_layers.append(nn.Linear(critic_input_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # ========== Action Noise ==========
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        # Action distribution (populated in update_distribution)
        self.distribution = None
        Normal.set_default_validate_args(False)
        
        # ========== Per-Environment Expert Tracking ==========
        # These will be initialized when we know num_envs
        self.current_expert_indices: torch.Tensor | None = None
        self.needs_resample: torch.Tensor | None = None
        self._num_envs: int | None = None
        
        # Store gate probabilities for load balancing loss computation
        self._last_gate_probs: torch.Tensor | None = None
        self._last_gate_logits: torch.Tensor | None = None
        
        print(f"MoE Actor-Critic initialized:")
        print(f"  Gate: {self.gate}")
        print(f"  Num experts: {num_experts}")
        print(f"  Full obs dim: {num_actor_obs} (base: {self.num_base_obs}, morph: {num_morphology_obs})")
        print(f"  Expert input dim: {expert_input_dim}")
        print(f"  Expert architecture: {actor_hidden_dims} -> {num_actions}")
        print(f"  Critic: {self.critic}")
        print(f"  Temperature: initial={tau_initial}, min={tau_min}, anneal_rate={tau_anneal_rate}")

    def init_expert_tracking(self, num_envs: int, device: torch.device):
        """Initialize per-environment expert tracking tensors.
        
        Args:
            num_envs: Number of parallel environments.
            device: Device to create tensors on.
        """
        self._num_envs = num_envs
        self._device = device
        self.current_expert_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.committed_expert_indices = None  # Set by sample_and_commit_experts()
        self.committed_gate_weights = None  # For soft routing
        self.experts_committed = False  # Flag to indicate if experts are committed for this rollout
        print(f"MoE expert tracking initialized for {num_envs} environments on {device}")

    def _update_diversity_coef(self, current_iter: int | None):
        """Update diversity coefficient according to the configured schedule."""
        if current_iter is None or self.diversity_coef_max <= 0.0:
            self.diversity_coef = 0.0
            return
        if current_iter < self.diversity_start_iter:
            self.diversity_coef = 0.0
            return
        if self.diversity_ramp_iters <= 0:
            self.diversity_coef = self.diversity_coef_max
            return
        progress = (current_iter - self.diversity_start_iter) / float(self.diversity_ramp_iters)
        progress = max(0.0, min(1.0, progress))
        self.diversity_coef = self.diversity_coef_max * progress

    def sample_and_commit_experts(self, observations: torch.Tensor, current_iter: int | None = None):
        """Sample experts and commit them for the current rollout.
        
        For hard routing: Uses Gumbel-softmax to sample expert indices.
        For soft routing: Computes temperature-scaled softmax weights.
        
        Call this once at the start of each rollout to fix expert selection.
        
        Args:
            observations: Full observations of shape (num_envs, obs_dim).
            current_iter: PPO iteration index (used for scheduling auxiliary losses).
        """
        # Update per-iteration auxiliary-loss schedules.
        self._update_diversity_coef(current_iter)

        # Extract morphology from observations
        morphology = observations[:, -self.num_morphology_obs:]
        
        # Get gate logits
        gate_logits = self.gate(morphology)
        
        if self.routing_type == "hard":
            # Sample using Gumbel-softmax
            _, expert_indices = self.gumbel_softmax_hard(gate_logits)
            self.committed_expert_indices = expert_indices.detach()
            self.committed_gate_weights = None
        else:  # soft
            # Compute temperature-scaled softmax weights
            gate_weights = F.softmax(gate_logits / self.tau, dim=-1)
            self.committed_gate_weights = gate_weights.detach()
            self.committed_expert_indices = gate_weights.argmax(dim=-1).detach()  # For logging
        
        # Update current_expert_indices for logging
        self.current_expert_indices = self.committed_expert_indices.clone()
        
        # Store gate probs for load balancing loss
        self._last_gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Mark experts as committed
        self.experts_committed = True

    def clear_committed_experts(self):
        """Clear committed experts after rollout/update.
        
        Call this after PPO update to allow fresh sampling in next rollout.
        """
        self.committed_expert_indices = None
        self.committed_gate_weights = None
        self.experts_committed = False

    def gumbel_softmax_hard(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Straight-through Gumbel-softmax for hard expert selection.
        
        Args:
            logits: Gate logits of shape (batch_size, num_experts).
            
        Returns:
            Tuple of (one_hot weights, expert indices).
            - one_hot: Shape (batch_size, num_experts), differentiable via straight-through.
            - indices: Shape (batch_size,), selected expert indices.
        """
        # Sample Gumbel noise: g = -log(-log(U)) where U ~ Uniform(0,1)
        u = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        
        # gumbel_noise = torch.zeros_like(logits) # I don't want to use gumbel noise for now

        # Add noise and apply temperature
        y_soft = F.softmax((logits + gumbel_noise) / self.tau, dim=-1)
        
        # Store for load balancing loss
        self._last_gate_probs = F.softmax(logits, dim=-1)
        self._last_gate_logits = logits.detach()
        
        # Hard selection via argmax
        indices = y_soft.argmax(dim=-1)
        y_hard = F.one_hot(indices, num_classes=self.num_experts).float()
        
        # Straight-through estimator: hard in forward, soft gradient in backward
        y = y_hard - y_soft.detach() + y_soft
        
        return y, indices

    def compute_all_expert_means(
        self,
        observations: torch.Tensor,
        morphology: torch.Tensor,
    ) -> torch.Tensor:
        """Compute action means from *all* experts (no routing/mixture).

        This is useful for auxiliary losses that compare expert behaviors (e.g.,
        diversity/repulsion losses). Gradients flow to the expert parameters.

        Args:
            observations: Base observations (without morphology), shape (batch, base_obs_dim).
            morphology: Morphology vectors, shape (batch, morph_dim).

        Returns:
            Tensor of shape (batch, num_experts, action_dim) containing each expert's
            action mean for each sample.
        """
        expert_input = torch.cat([observations, morphology], dim=-1)
        expert_outputs = torch.stack([expert(expert_input) for expert in self.experts], dim=1)
        return expert_outputs

    def forward_all_experts_soft(
        self, 
        observations: torch.Tensor, 
        morphology: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through ALL experts with soft (temperature-scaled) weighting.
        
        This implements soft MoE where all expert outputs are weighted by the 
        temperature-scaled softmax of gate logits. As temperature τ → 0, this
        approaches hard selection. Gradients flow through the softmax to the gate.
        
        If experts are committed (via sample_and_commit_experts), uses the committed
        weights instead of recomputing from gate. This ensures consistent expert
        weighting throughout a rollout.
        
        Args:
            observations: Base observations (without morphology) of shape (batch, base_obs_dim).
            morphology: Morphology vectors of shape (batch, morph_dim).
            
        Returns:
            action_means: Shape (batch, action_dim), soft-weighted expert outputs.
        """
        # Concatenate observations and morphology for expert input
        expert_input = torch.cat([observations, morphology], dim=-1)
        
        # Use committed weights only when gradients are disabled (rollout/inference).
        #
        # Important: PPO update needs gradients to flow through the gate. If we reuse
        # committed (detached) weights during update, the gate will not learn from
        # the PPO objective. Therefore, when grad is enabled we recompute weights.
        if (
            self.experts_committed
            and self.committed_gate_weights is not None
            and not torch.is_grad_enabled()
        ):
            # Rollout/inference path: stable per-rollout weights.
            gate_weights = self.committed_gate_weights
            # Still compute gate probs for logging/load balancing (no grad anyway).
            gate_logits = self.gate(morphology)
        else:
            # Update/training path: recompute weights to allow gate gradients.
            gate_logits = self.gate(morphology)
            gate_weights = F.softmax(gate_logits / self.tau, dim=-1)  # (batch, num_experts)

        # Store gate probs for load balancing loss/logging (no temperature).
        self._last_gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Track which expert has highest weight (for logging purposes)
        expert_indices = gate_weights.argmax(dim=-1)
        if self.current_expert_indices is not None and expert_indices.shape[0] == self.current_expert_indices.shape[0]:
            self.current_expert_indices = expert_indices.detach()
        
        # Run ALL experts and stack outputs: (batch, num_experts, action_dim)
        expert_outputs = torch.stack([
            expert(expert_input) for expert in self.experts
        ], dim=1)
        
        # Soft weighted combination: (batch, num_experts, 1) * (batch, num_experts, action_dim)
        # gate_weights shape: (batch, num_experts) -> (batch, num_experts, 1)
        weighted_output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (batch, action_dim)
        
        return weighted_output

    def forward_all_experts_hard(
        self, 
        observations: torch.Tensor, 
        morphology: torch.Tensor,
        use_gumbel_noise: bool = True
    ) -> torch.Tensor:
        """Forward pass through ALL experts with hard (one-hot) gating via straight-through.
        
        This implements hard MoE where one expert is selected per sample using
        Gumbel-softmax with straight-through estimator. In forward pass, the output
        is from a single expert (one-hot selection). In backward pass, gradients
        flow through the soft Gumbel-softmax to the gate network.
        
        If experts are committed (via sample_and_commit_experts), uses the committed
        indices instead of resampling. This ensures consistent expert selection
        throughout a rollout.
        
        Note: Only the selected expert receives gradients for its parameters.
        Non-selected experts compute forward (for gate gradients) but don't learn.
        
        Args:
            observations: Base observations (without morphology) of shape (batch, base_obs_dim).
            morphology: Morphology vectors of shape (batch, morph_dim).
            use_gumbel_noise: If True, use Gumbel noise for stochastic selection (training).
                             If False, use deterministic argmax (evaluation/update).
            
        Returns:
            action_means: Shape (batch, action_dim), output from selected expert only.
        """
        # Concatenate observations and morphology for expert input
        expert_input = torch.cat([observations, morphology], dim=-1)
        
        # Get gate logits (needed for gradient flow and logging)
        gate_logits = self.gate(morphology)
        
        # Store gate probs for load balancing loss and logging (without temperature)
        self._last_gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Use committed indices if available, otherwise sample/compute
        if self.experts_committed and self.committed_expert_indices is not None:
            # Use committed indices (no Gumbel noise during rollout)
            expert_indices = self.committed_expert_indices
            # Still compute y_soft for straight-through gradient
            y_soft = F.softmax(gate_logits / self.tau, dim=-1)
            y_hard_detached = F.one_hot(expert_indices, num_classes=self.num_experts).float()
            y_hard = y_hard_detached - y_soft.detach() + y_soft
        elif use_gumbel_noise:
            # Gumbel-softmax with straight-through for stochastic hard selection
            y_hard, expert_indices = self.gumbel_softmax_hard(gate_logits)
        else:
            # Deterministic selection (for evaluation or PPO update)
            expert_indices = gate_logits.argmax(dim=-1)
            y_soft = F.softmax(gate_logits / self.tau, dim=-1)
            y_hard_detached = F.one_hot(expert_indices, num_classes=self.num_experts).float()
            # Straight-through: forward uses one-hot, backward uses soft
            y_hard = y_hard_detached - y_soft.detach() + y_soft
        
        # Track expert indices for logging
        if self.current_expert_indices is not None and expert_indices.shape[0] == self.current_expert_indices.shape[0]:
            self.current_expert_indices = expert_indices.detach()
        
        # Run ALL experts and stack outputs: (batch, num_experts, action_dim)
        expert_outputs = torch.stack([
            expert(expert_input) for expert in self.experts
        ], dim=1)
        
        # Hard weighted combination using one-hot y_hard
        # In forward: only selected expert contributes (one-hot multiplication)
        # In backward: gradients flow through y_soft to gate
        weighted_output = (y_hard.unsqueeze(-1) * expert_outputs).sum(dim=1)  # (batch, action_dim)
        
        return weighted_output

    def forward_single_expert(
        self, 
        observations: torch.Tensor, 
        morphology: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through a single expert per sample (efficient, no gradient to gate).
        
        Use this for inference or when you don't need gate gradients.
        
        Args:
            observations: Base observations (without morphology) of shape (batch, base_obs_dim).
            morphology: Morphology vectors of shape (batch, morph_dim).
            expert_indices: Which expert to use for each sample, shape (batch,).
            
        Returns:
            Action means of shape (batch, action_dim).
        """
        batch_size = observations.size(0)
        device = observations.device
        
        # Concatenate observations and morphology for expert input
        expert_input = torch.cat([observations, morphology], dim=-1)
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, self.num_actions, device=device)
        
        # Forward through each expert for its assigned samples
        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx)
            if mask.any():
                outputs[mask] = self.experts[expert_idx](expert_input[mask])
        
        return outputs

    def update_distribution(self, observations: torch.Tensor, morphology: torch.Tensor | None = None):
        """Update action distribution based on observations.
        
        Routing behavior depends on `self.routing_type`:
        - 'soft': Temperature-scaled softmax weighted combination of all experts.
        - 'hard': Straight-through Gumbel-softmax one-hot selection.
        
        Args:
            observations: Full observations (including morphology at the end).
            morphology: Morphology vector. If None, extracts from last num_morphology_obs dims.
        """
        # Extract morphology if not provided separately
        if morphology is None:
            morphology = observations[:, -self.num_morphology_obs:]
            actor_obs = observations[:, :-self.num_morphology_obs]
        else:
            actor_obs = observations
        
        # Forward through experts based on routing type
        if self.routing_type == "soft":
            mean = self.forward_all_experts_soft(actor_obs, morphology)
        else:  # hard
            mean = self.forward_all_experts_hard(actor_obs, morphology, use_gumbel_noise=True)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample actions from the policy.
        
        Args:
            observations: Actor observations of shape (num_envs, obs_dim).
            
        Returns:
            Sampled actions of shape (num_envs, action_dim).
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass for inference (evaluation/deployment).
        
        Routing behavior depends on `self.routing_type`:
        - 'soft': Temperature-scaled softmax weighted combination (same as training).
        - 'hard': Deterministic argmax selection, runs only the selected expert.
        
        Args:
            observations: Actor observations of shape (batch, obs_dim).
            
        Returns:
            Action means of shape (batch, action_dim).
        """
        # Extract morphology from observations
        morphology = observations[:, -self.num_morphology_obs:]
        actor_obs = observations[:, :-self.num_morphology_obs]
        
        if self.routing_type == "soft":
            # Soft weighted combination (same as training)
            return self.forward_all_experts_soft(actor_obs, morphology)
        else:  # hard
            # Deterministic selection via argmax (efficient - only runs selected expert)
            gate_logits = self.gate(morphology)
            expert_indices = gate_logits.argmax(dim=-1)
            
            # Track expert indices for logging
            if self.current_expert_indices is not None and expert_indices.shape[0] == self.current_expert_indices.shape[0]:
                self.current_expert_indices = expert_indices.detach()
            
            # Store gate probs for logging
            self._last_gate_probs = F.softmax(gate_logits, dim=-1)
            
            return self.forward_single_expert(actor_obs, morphology, expert_indices)

    def act_for_update(
        self, 
        observations: torch.Tensor, 
        morphology: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for PPO update - same routing as rollout for consistency.
        
        Routing behavior depends on `self.routing_type`:
        - 'soft': Temperature-scaled softmax weighted combination.
        - 'hard': Deterministic argmax selection with straight-through for gradients.
        
        For hard routing, we use deterministic argmax (no Gumbel noise) during update
        to match the rollout behavior, but still run all experts and use straight-through
        to allow gradient flow to the gate network.
        
        Args:
            observations: Actor observations (without morphology).
            morphology: Morphology vectors.
            
        Returns:
            Action means.
        """
        # Forward through experts based on routing type
        if self.routing_type == "soft":
            mean = self.forward_all_experts_soft(observations, morphology)
        else:  # hard
            # Use deterministic selection (no Gumbel noise) but still straight-through
            # This ensures consistency with rollout while allowing gate gradients
            mean = self.forward_all_experts_hard(observations, morphology, use_gumbel_noise=False)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        # Create distribution
        self.distribution = Normal(mean, std)
        return mean

    def act_for_update_hard(
        self, 
        observations: torch.Tensor, 
        morphology: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for PPO update with hard routing using stored expert indices.
        
        Uses the expert indices that were selected during rollout to ensure
        consistency in the importance sampling ratio. Gradients still flow to 
        the gate network via straight-through estimator.
        
        Args:
            observations: Actor observations (without morphology).
            morphology: Morphology vectors.
            expert_indices: Expert indices selected during rollout.
            
        Returns:
            Action means.
        """
        mean = self.forward_with_stored_indices(observations, morphology, expert_indices)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        # Create distribution
        self.distribution = Normal(mean, std)
        return mean

    def forward_with_stored_indices(
        self,
        observations: torch.Tensor,
        morphology: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using stored expert indices with straight-through gradient.
        
        This ensures consistency between rollout and update for hard routing.
        The forward pass uses the exact expert that was selected during rollout,
        but gradients flow through the soft gate outputs via straight-through.
        
        Args:
            observations: Base observations (without morphology).
            morphology: Morphology vectors.
            expert_indices: Expert indices selected during rollout.
            
        Returns:
            Action means from the stored expert selection.
        """
        # Concatenate observations and morphology for expert input
        expert_input = torch.cat([observations, morphology], dim=-1)
        
        # Get gate logits and compute soft probabilities for gradient flow
        gate_logits = self.gate(morphology)
        y_soft = F.softmax(gate_logits / self.tau, dim=-1)
        
        # Store gate probs for load balancing loss and logging
        self._last_gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Create one-hot from stored expert indices
        y_hard_detached = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        
        # Straight-through: forward uses stored indices, backward uses soft
        y_hard = y_hard_detached - y_soft.detach() + y_soft
        
        # Run ALL experts and stack outputs
        expert_outputs = torch.stack([
            expert(expert_input) for expert in self.experts
        ], dim=1)
        
        # Weighted combination using stored indices (hard in forward, soft gradient)
        weighted_output = (y_hard.unsqueeze(-1) * expert_outputs).sum(dim=1)
        
        return weighted_output

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Get log probability of actions under current distribution.
        
        Args:
            actions: Actions of shape (batch, action_dim).
            
        Returns:
            Log probabilities of shape (batch,).
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate value of observations using the critic.
        
        Args:
            critic_observations: Critic observations of shape (batch, critic_obs_dim).
            
        Returns:
            Value estimates of shape (batch, 1).
        """
        return self.critic(critic_observations)

    def reset(self, dones: torch.Tensor | None = None):
        """Handle episode resets.
        
        Note: With the all-experts weighted approach, expert selection is recomputed
        on every forward pass based on current morphology, so no explicit reset
        tracking is needed. This method is kept for API compatibility.
        
        Args:
            dones: Done flags of shape (num_envs,) indicating which envs reset.
        """
        # No action needed - expert selection is recomputed each forward pass
        pass

    def anneal_temperature(self):
        """Anneal Gumbel-softmax temperature."""
        self.tau = max(self.tau_min, self.tau * (1 - self.tau_anneal_rate))

    def reset_temperature(self):
        """Reset temperature to initial value."""
        self.tau = self.tau_initial

    def get_expert_utilization(self) -> torch.Tensor:
        """Get count of environments assigned to each expert.
        
        Returns:
            Tensor of shape (num_experts,) with counts.
        """
        if self.current_expert_indices is None:
            return torch.zeros(self.num_experts)
        return torch.bincount(
            self.current_expert_indices, 
            minlength=self.num_experts
        ).float()

    def compute_load_balance_loss(self, gate_probs: torch.Tensor | None = None) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert utilization.
        
        Args:
            gate_probs: Gate probabilities of shape (batch, num_experts).
                       If None, uses last stored gate probs.
                       
        Returns:
            Load balance loss scalar.
        """
        if gate_probs is None:
            gate_probs = self._last_gate_probs
            
        if gate_probs is None:
            return torch.tensor(0.0)
        
        # f_i = mean probability assigned to expert i
        f = gate_probs.mean(dim=0)
        
        # p_i = fraction of samples where expert i is top choice
        top_expert = gate_probs.argmax(dim=-1)
        p = torch.zeros(self.num_experts, device=gate_probs.device)
        for i in range(self.num_experts):
            p[i] = (top_expert == i).float().mean()
        
        # Load balance loss: encourages uniform f and p
        return self.num_experts * (f * p).sum()

    @property
    def action_mean(self) -> torch.Tensor:
        """Mean of the action distribution."""
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        """Standard deviation of the action distribution."""
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Entropy of the action distribution."""
        return self.distribution.entropy().sum(dim=-1)

    def forward(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        """Load model state dict.
        
        Args:
            state_dict: State dictionary to load.
            strict: Whether to strictly enforce key matching.
            
        Returns:
            True indicating this is a resumed training.
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

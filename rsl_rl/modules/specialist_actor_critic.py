from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class SpecialistActorCritic(nn.Module):
    """Mixture-of-Specialists actor-critic with fixed hard routing.

    Routing is controlled by `forced_expert_indices` provided externally
    (for example by `play_specialist.py` from runtime GCR/SPCF values).
    This module does not read cluster IDs from the observation tensor.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[128, 64, 32],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        drop_last_obs_dim: bool = False,
        gcr_threshold: float = 0.82,
        spcf_threshold: float = 0.006,
        specialist_ckpt_by0spc0: str | None = None,
        specialist_ckpt_by1spc0: str | None = None,
        specialist_ckpt_by0spc1: str | None = None,
        specialist_ckpt_by1spc1: str | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "SpecialistActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation_fn = resolve_nn_activation(activation)

        self.num_actions = num_actions
        self.num_experts = 4
        self.drop_last_obs_dim = bool(drop_last_obs_dim)
        self.gcr_threshold = float(gcr_threshold)
        self.spcf_threshold = float(spcf_threshold)

        if self.drop_last_obs_dim:
            print(
                "[WARN] drop_last_obs_dim=True is deprecated for SpecialistActorCritic. "
                "Observations are no longer expected to include cluster_id; using full observation."
            )
        self.actor_input_dim = num_actor_obs
        self.critic_input_dim = num_critic_obs
        if self.actor_input_dim <= 0:
            raise ValueError(
                f"Invalid actor input dim: {self.actor_input_dim} (num_actor_obs={num_actor_obs}, "
                f"drop_last_obs_dim={self.drop_last_obs_dim})"
            )
        if self.critic_input_dim <= 0:
            raise ValueError(
                f"Invalid critic input dim: {self.critic_input_dim} (num_critic_obs={num_critic_obs}, "
                f"drop_last_obs_dim={self.drop_last_obs_dim})"
            )

        # Four expert actors:
        # 0 -> gcr low,  spcf low
        # 1 -> gcr low,  spcf high
        # 2 -> gcr high, spcf low
        # 3 -> gcr high, spcf high
        self.experts = nn.ModuleList(
            [
                self._build_actor(self.actor_input_dim, num_actions, actor_hidden_dims, activation_fn)
                for _ in range(self.num_experts)
            ]
        )

        self.critic = self._build_critic(self.critic_input_dim, critic_hidden_dims, activation_fn)

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)
        self.current_expert_indices: torch.Tensor | None = None
        self.forced_expert_indices: torch.Tensor | None = None
        self.norm_eps = 1.0e-2

        # Per-expert actor observation normalization loaded from each specialist checkpoint.
        self.register_buffer("expert_obs_mean", torch.zeros(self.num_experts, self.actor_input_dim))
        self.register_buffer("expert_obs_std", torch.ones(self.num_experts, self.actor_input_dim))
        self.register_buffer("expert_obs_norm_loaded", torch.zeros(self.num_experts, dtype=torch.bool))

        # Per-expert action noise loaded from each specialist checkpoint.
        self.register_buffer("expert_std_table", torch.ones(self.num_experts, num_actions))
        self.register_buffer("expert_noise_loaded", torch.zeros(self.num_experts, dtype=torch.bool))

        ckpt_map = {
            0: specialist_ckpt_by0spc0,
            2: specialist_ckpt_by1spc0,
            1: specialist_ckpt_by0spc1,
            3: specialist_ckpt_by1spc1,
        }
        self._load_specialists_from_checkpoints(ckpt_map)

        print("SpecialistActorCritic initialized:")
        print(f"  Num experts: {self.num_experts}")
        print(f"  Actor input dim: {self.actor_input_dim}")
        print(f"  Critic input dim: {self.critic_input_dim}")
        print(f"  Cluster thresholds: gcr={self.gcr_threshold}, spcf={self.spcf_threshold}")
        if self.expert_obs_norm_loaded.any():
            loaded_ids = torch.nonzero(self.expert_obs_norm_loaded, as_tuple=False).view(-1).tolist()
            print(f"  Loaded per-expert obs normalizers: {loaded_ids}")
        if self.expert_noise_loaded.any():
            loaded_ids = torch.nonzero(self.expert_noise_loaded, as_tuple=False).view(-1).tolist()
            print(f"  Loaded per-expert action std: {loaded_ids}")

    @staticmethod
    def _build_actor(input_dim: int, num_actions: int, hidden_dims: list[int], activation_fn: nn.Module) -> nn.Sequential:
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation_fn)
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], num_actions))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation_fn)
        return nn.Sequential(*layers)

    @staticmethod
    def _build_critic(input_dim: int, hidden_dims: list[int], activation_fn: nn.Module) -> nn.Sequential:
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation_fn)
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], 1))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation_fn)
        return nn.Sequential(*layers)

    @staticmethod
    def _default_checkpoint_paths() -> dict[int, str]:
        repo_root = Path(__file__).resolve().parents[3]
        base = repo_root / "ballu_isclb_extension" / "logs" / "rsl_rl" / "lab_02.10.2026"
        return {
            0: str(base / "2026-02-08_14-46-32_mlp_sub_controller_by0spc0" / "model_best.pt"),
            2: str(base / "2026-02-08_15-57-03_mlp_sub_controller_by1spc0" / "model_best.pt"),
            1: str(base / "2026-02-08_16-13-28_mlp_sub_controller_by0spc1" / "model_best.pt"),
            3: str(base / "2026-02-08_17-41-26_mlp_sub_controller_by1spc1" / "model_best.pt"),
        }

    @staticmethod
    def _env_override_for_expert(expert_idx: int) -> str:
        if expert_idx == 0:
            return "BALLU_SPECIALIST_CKPT_BY0SPC0"
        if expert_idx == 1:
            return "BALLU_SPECIALIST_CKPT_BY0SPC1"
        if expert_idx == 2:
            return "BALLU_SPECIALIST_CKPT_BY1SPC0"
        return "BALLU_SPECIALIST_CKPT_BY1SPC1"

    def _load_specialists_from_checkpoints(self, ckpt_map: dict[int, str | None]) -> None:
        default_paths = self._default_checkpoint_paths()
        loaded_any = False
        self.loaded_specialist_paths: dict[int, str] = {}
        self.specialist_mapping = {
            0: "by0spc0 (GCR low, SPCF low)",
            1: "by0spc1 (GCR low, SPCF high)",
            2: "by1spc0 (GCR high, SPCF low)",
            3: "by1spc1 (GCR high, SPCF high)",
        }

        for expert_idx in range(self.num_experts):
            env_var = self._env_override_for_expert(expert_idx)
            path = os.getenv(env_var, ckpt_map.get(expert_idx) or default_paths[expert_idx])
            if not path:
                continue
            if not os.path.exists(path):
                print(f"[WARN] Specialist checkpoint not found for expert {expert_idx}: {path}")
                continue

            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model_sd = ckpt.get("model_state_dict", ckpt)
            actor_sd = {}
            for key, value in model_sd.items():
                if key.startswith("actor."):
                    actor_sd[key[len("actor."):]] = value
            if not actor_sd:
                print(f"[WARN] No actor weights found in specialist checkpoint: {path}")
                continue

            self.experts[expert_idx].load_state_dict(actor_sd, strict=True)
            loaded_any = True
            self.loaded_specialist_paths[expert_idx] = path
            print(f"[INFO] Loaded specialist expert {expert_idx} from: {path}")

            # Load per-expert action noise if present.
            if self.noise_std_type == "scalar" and "std" in model_sd:
                std_vec = model_sd["std"].detach().to(torch.float32).view(-1)
                if std_vec.numel() == self.num_actions:
                    self.expert_std_table[expert_idx] = std_vec
                    self.expert_noise_loaded[expert_idx] = True
                    with torch.no_grad():
                        self.std.copy_(std_vec)
                else:
                    print(
                        f"[WARN] Ignoring std for expert {expert_idx}: expected {self.num_actions}, got {std_vec.numel()}"
                    )
            elif self.noise_std_type == "log" and "log_std" in model_sd:
                log_std_vec = model_sd["log_std"].detach().to(torch.float32).view(-1)
                if log_std_vec.numel() == self.num_actions:
                    self.expert_std_table[expert_idx] = torch.exp(log_std_vec)
                    self.expert_noise_loaded[expert_idx] = True
                    with torch.no_grad():
                        self.log_std.copy_(log_std_vec)
                else:
                    print(
                        f"[WARN] Ignoring log_std for expert {expert_idx}: expected {self.num_actions}, got {log_std_vec.numel()}"
                    )

            # Load per-expert actor observation normalization if present.
            obs_norm_sd = ckpt.get("obs_norm_state_dict", None)
            if obs_norm_sd is not None:
                mean = obs_norm_sd.get("_mean", None)
                std = obs_norm_sd.get("_std", None)
                if mean is not None and std is not None:
                    mean = mean.detach().to(torch.float32).view(-1)
                    std = std.detach().to(torch.float32).view(-1)
                    if mean.numel() == self.actor_input_dim and std.numel() == self.actor_input_dim:
                        self.expert_obs_mean[expert_idx] = mean
                        self.expert_obs_std[expert_idx] = std
                        self.expert_obs_norm_loaded[expert_idx] = True
                    else:
                        print(
                            f"[WARN] Ignoring obs norm for expert {expert_idx}: expected dim {self.actor_input_dim}, "
                            f"got mean={mean.numel()}, std={std.numel()}"
                        )
                else:
                    print(f"[WARN] obs_norm_state_dict missing _mean/_std for expert {expert_idx}: {path}")

        if not loaded_any:
            print("[WARN] No specialist checkpoints loaded. Using randomly initialized experts.")
            return

        print("[INFO] Specialist routing map:")
        for expert_idx in range(self.num_experts):
            desc = self.specialist_mapping[expert_idx]
            loaded_path = self.loaded_specialist_paths.get(expert_idx, "<not loaded>")
            print(f"  - specialist {expert_idx}: {desc}")
            print(f"      checkpoint: {loaded_path}")

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _split_actor_input(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor_obs = observations
        if self.forced_expert_indices is not None and self.forced_expert_indices.shape[0] == observations.shape[0]:
            cluster_ids = self.forced_expert_indices.to(observations.device).view(-1).long()
        elif self.training:
            # Keep training path functional even when no external routing is provided.
            cluster_ids = torch.zeros(observations.shape[0], device=observations.device, dtype=torch.long)
        else:
            raise RuntimeError(
                "SpecialistActorCritic requires `forced_expert_indices` during inference/execution. "
                "Set it from runtime morphology routing before calling policy(observations)."
            )
        cluster_ids = torch.clamp(cluster_ids, min=0, max=self.num_experts - 1)
        return actor_obs, cluster_ids

    def _normalize_actor_obs_by_expert(self, actor_obs: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        """Apply specialist-specific empirical normalization to actor observations."""
        normalized = actor_obs.clone()
        for expert_idx in range(self.num_experts):
            mask = cluster_ids == expert_idx
            if not mask.any():
                continue
            mean = self.expert_obs_mean[expert_idx]
            std = self.expert_obs_std[expert_idx]
            normalized[mask] = (normalized[mask] - mean) / (std + self.norm_eps)
        return normalized

    def _forward_by_expert(self, actor_obs: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros(actor_obs.shape[0], self.num_actions, device=actor_obs.device, dtype=actor_obs.dtype)
        for expert_idx in range(self.num_experts):
            mask = cluster_ids == expert_idx
            if mask.any():
                outputs[mask] = self.experts[expert_idx](actor_obs[mask])
        self.current_expert_indices = cluster_ids.detach()
        return outputs

    def update_distribution(self, observations: torch.Tensor):
        actor_obs, cluster_ids = self._split_actor_input(observations)
        actor_obs = self._normalize_actor_obs_by_expert(actor_obs, cluster_ids)
        mean = self._forward_by_expert(actor_obs, cluster_ids)
        if self.noise_std_type == "scalar":
            std = self.expert_std_table[cluster_ids] if self.expert_noise_loaded.any() else self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            if self.expert_noise_loaded.any():
                std = self.expert_std_table[cluster_ids]
            else:
                std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_obs, cluster_ids = self._split_actor_input(observations)
        actor_obs = self._normalize_actor_obs_by_expert(actor_obs, cluster_ids)
        return self._forward_by_expert(actor_obs, cluster_ids)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def load_state_dict(self, state_dict, strict=True):
        """Load specialist state dict if compatible.

        For standard ActorCritic checkpoints, ignore model weights here (experts are
        loaded separately from dedicated specialist checkpoints) and return False so
        optimizer state is not loaded by OnPolicyRunner.
        """
        has_specialist_weights = any(key.startswith("experts.") for key in state_dict.keys())
        if has_specialist_weights:
            super().load_state_dict(state_dict, strict=False)
            return True

        print("[INFO] Ignoring non-specialist policy weights during load; using configured specialist checkpoints.")
        return False

    def forward(self):
        raise NotImplementedError

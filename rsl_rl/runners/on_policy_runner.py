# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
import statistics
import time
import torch
import csv
from collections import deque
from tqdm import tqdm
import copy

import rsl_rl
from rsl_rl.algorithms import PPO, Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    MoEActorCritic,
    SpecialistActorCritic,
    StudentTeacher,
    StudentTeacherRecurrent,
)
from rsl_rl.utils import store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # resolve training type depending on the algorithm
        if self.alg_cfg["class_name"] == "PPO":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "Distillation":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        # resolve dimensions of observations
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        # resolve type of privileged observations
        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        # resolve dimensions of privileged observations
        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        # evaluate the policy class
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: (
            ActorCritic
            | ActorCriticRecurrent
            | MoEActorCritic
            | SpecialistActorCritic
            | StudentTeacher
            | StudentTeacherRecurrent
        ) = policy_class(
            num_obs, num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        
        # Check if using MoE policy
        self.use_moe = isinstance(policy, MoEActorCritic)

        # resolve dimension of rnd gated state
        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO | Distillation = alg_class(
            policy, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

        # init storage and model
        self.alg.init_storage(
            self.training_type,
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_privileged_obs],
            [self.env.num_actions],
        )
        
        # Initialize MoE expert tracking if using MoE policy
        if self.use_moe:
            self.alg.policy.init_expert_tracking(self.env.num_envs, self.device)
            # Gate probs logging configuration (read from moe_cfg if available)
            moe_cfg = self.alg_cfg.get("moe_cfg", {}) or {}
            self.gate_probs_log_interval = moe_cfg.get("gate_probs_log_interval", 3)
            self.gate_probs_num_envs = min(
                moe_cfg.get("gate_probs_num_envs", 4),
                self.env.num_envs
            )
            self.gate_probs_log_file = None  # Will be set when log_dir is available

        # Decide whether to disable logging
        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        # Initialize CSV logging for aggregate actions
        self._init_action_csv_logging()

    def _init_action_csv_logging(self):
        """Initialize CSV file for logging aggregate actions."""
        # Create CSV file in the root directory (where the script is run from)
        self.action_csv_path = "aggregate_actions.csv"
        
        # Only initialize CSV if logging is enabled and we're the main process
        if not self.disable_logs:
            # Create CSV file with headers
            with open(self.action_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header based on number of actions
                header = ['iteration', 'rollout_step'] + [f'mean_action_{i}' for i in range(self.env.num_actions)] + \
                    [f'std_action_{i}' for i in range(self.env.num_actions)]
                writer.writerow(header)
            print(f"[INFO] Initialized action logging CSV at: {self.action_csv_path}")

    def _log_aggregate_actions(self, actions: torch.Tensor, iteration: int, rollout_step: int):
        """Log aggregate actions (averaged over all environments) to CSV.
        
        Args:
            actions: Action tensor of shape (num_envs, num_actions)
            iteration: Current learning iteration
            rollout_step: Current step within the rollout
        """
        if not self.disable_logs:
            # Compute mean actions across all environments
            mean_actions = actions.mean(dim=0).cpu().numpy()
            std_actions = actions.std(dim=0).cpu().numpy()
            # Write to CSV
            with open(self.action_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                row = [iteration, rollout_step] + mean_actions.tolist() + std_actions.tolist()
                writer.writerow(row)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # check if teacher is loaded
        if self.training_type == "distillation" and not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        # randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # start learning
        obs, extras = self.env.get_observations()
        privileged_obs = extras["observations"].get(self.privileged_obs_type, obs)
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Create variables to store the best checkpoint and the best reward as the training progresses
        best_reward = -float('inf')
        best_crclm_level = -1.0
        best_checkpoint = {
            "model_state_dict": copy.deepcopy(self.alg.policy.state_dict()),
            "optimizer_state_dict": copy.deepcopy(self.alg.optimizer.state_dict()),
            "iter": -1,
            "infos": None,
            "best_crclm_level": best_crclm_level
        }
        if self.empirical_normalization:
            best_checkpoint.update({
                    "obs_norm_state_dict": copy.deepcopy(self.obs_normalizer.state_dict()),
                    "privileged_obs_norm_state_dict": copy.deepcopy(self.privileged_obs_normalizer.state_dict()),
                }
            )
            
        torch.save(best_checkpoint, os.path.join(self.log_dir, "model_best.pt"))
        
        # create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
            # TODO: Do we need to synchronize empirical normalizers?
            #   Right now: No, because they all should converge to the same values "asymptotically".

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # Extract the pure IsaacLab environment
        pure_env = self.env.unwrapped.unwrapped

        for it in range(start_iter, tot_iter):
            pure_env.rsl_rl_iteration = it
            # if it >= 0.3 * tot_iter and self.alg.mirror_symmetry['weight'] != 0.1: # Setting the mirror symmetry weight to 1.0 after 70% of the training
            #     print(f"Trying to set mirror symmetry weight to 1.0 at iteration {it}")
            #     self.alg.mirror_symmetry['weight'] = 0.1
            #     print(self.alg.mirror_symmetry)

            start = time.time()
            # Rollout
            with torch.inference_mode():
                # Commit experts for this rollout (MoE only)
                if self.use_moe:
                    self.alg.policy.sample_and_commit_experts(obs, current_iter=it)
                
                try:
                    for rollout_step in range(self.num_steps_per_env):
                        # Sample actions
                        actions = self.alg.act(obs, privileged_obs)
                        
                        # Log aggregate actions to CSV
                        # self._log_aggregate_actions(actions, it, rollout_step)
                        
                        # Step the environment
                        obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                        # Move to device
                        obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                        # perform normalization
                        obs = self.obs_normalizer(obs)
                        if self.privileged_obs_type is not None:
                            privileged_obs = self.privileged_obs_normalizer(
                                infos["observations"][self.privileged_obs_type].to(self.device)
                            )
                        else:
                            privileged_obs = obs

                        # process the step
                        self.alg.process_env_step(rewards, dones, infos)

                        # Extract intrinsic rewards (only for logging)
                        intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                        # book keeping
                        if self.log_dir is not None:
                            if "episode" in infos:
                                ep_infos.append(infos["episode"])
                            elif "log" in infos:
                                ep_infos.append(infos["log"])
                            # Update rewards
                            if self.alg.rnd:
                                cur_ereward_sum += rewards
                                cur_ireward_sum += intrinsic_rewards  # type: ignore
                                cur_reward_sum += rewards + intrinsic_rewards
                            else:
                                cur_reward_sum += rewards
                            # Update episode length
                            cur_episode_length += 1
                            # Clear data for completed episodes
                            # -- common
                            new_ids = (dones > 0).nonzero(as_tuple=False)
                            # print("new_ids: ", new_ids)
                            # Terminate/exit the program if new_ids is not empty
                            # if new_ids.numel() > 0:
                            #     print(f"Terminating program: Found {new_ids.numel()} completed environments at iteration {it}")
                            #     exit()

                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0
                            # -- intrinsic and extrinsic rewards
                            if self.alg.rnd:
                                erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                                cur_ereward_sum[new_ids] = 0
                                cur_ireward_sum[new_ids] = 0
                except Exception as e:
                    torch.save(best_checkpoint, os.path.join(self.log_dir, "model_best.pt"))
                    raise
                
                stop = time.time()
                collection_time = stop - start
                start = stop

                # compute returns
                if self.training_type == "rl":
                    self.alg.compute_returns(privileged_obs)

            # Compare and save the best checkpoint so far
            try:
                # curr_reward = statistics.mean(rewbuffer)
                curr_crclm_level = infos["log"].get("Curriculum/obstacle_height_levels_custom", -1)
                if curr_crclm_level > best_crclm_level:
                    best_crclm_level = curr_crclm_level
                    best_checkpoint = {
                        "model_state_dict": copy.deepcopy(self.alg.policy.state_dict()),
                        "optimizer_state_dict": copy.deepcopy(self.alg.optimizer.state_dict()),
                        "iter": it,
                        "infos": infos,
                        "best_crclm_level": curr_crclm_level
                    }
                    if self.empirical_normalization:
                        best_checkpoint.update({
                                "obs_norm_state_dict": copy.deepcopy(self.obs_normalizer.state_dict()),
                                "privileged_obs_norm_state_dict": copy.deepcopy(self.privileged_obs_normalizer.state_dict()),
                            }
                        )
            except:
                # print(f"Error computing mean reward at iteration {it}")
                # curr_reward = 0
                print(f"Error extracting current curriculum level at iteration {it}")
            
            # update policy
            loss_dict = self.alg.update()
            
            # Anneal MoE temperature and clear committed experts
            if self.use_moe:
                self.alg.policy.anneal_temperature()
                self.alg.policy.clear_committed_experts()

            stop = time.time()
            learn_time = stop - start

            self.current_learning_iteration = it
            # log info
            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Log gate probabilities for MoE visualization
                if self.use_moe and it % self.gate_probs_log_interval == 0:
                    self._log_gate_probs(it, obs)
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                    torch.save(best_checkpoint, os.path.join(self.log_dir, "model_best.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # if possible store them to wandb
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        # Save the best checkpoint after training
        torch.save(best_checkpoint, os.path.join(self.log_dir, "model_best.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # -- Episode info
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.8f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.8f}\n"""

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # -- Losses
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # -- Policy
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # -- Performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # -- Training
        if len(locs["rewbuffer"]) > 0:
            # separate logging for intrinsic and extrinsic rewards
            if self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            # MoE logging
            if self.use_moe:
                self.writer.add_scalar("MoE/temperature", self.alg.policy.tau, locs["it"])
                expert_utilization = self.alg.policy.get_expert_utilization()
                for i, count in enumerate(expert_utilization):
                    self.writer.add_scalar(f"MoE/expert_{i}_count", count.item(), locs["it"])

                # Per-expert curriculum progress (mean obstacle height)
                # We compute this globally across all envs each PPO iteration.
                try:
                    pure_env = locs.get("pure_env", None)
                    if pure_env is not None and hasattr(pure_env, "obstacle_height_list"):
                        # Inter-obstacle spacing along -Y used to encode curriculum level.
                        inter_obstacle_spacing_y = 2.0
                        env_origins = pure_env.scene.env_origins  # (num_envs, 3)
                        level = (-env_origins[:, 1] / inter_obstacle_spacing_y).clamp(min=0.0)
                        level_idx = level.long()

                        all_heights = torch.as_tensor(
                            pure_env.obstacle_height_list, device=env_origins.device, dtype=torch.float32
                        )
                        level_idx = level_idx.clamp(min=0, max=all_heights.numel() - 1)
                        obstacle_heights = all_heights[level_idx]  # (num_envs,)

                        # Expert assignment per env:
                        expert_ids = self.alg.policy.current_expert_indices

                        if expert_ids is not None:
                            expert_ids = expert_ids.to(env_origins.device).long()
                            for expert_id in range(self.alg.policy.num_experts):
                                mask = expert_ids == expert_id
                                if mask.any():
                                    mean_h = obstacle_heights[mask].mean()
                                    self.writer.add_scalar(
                                        f"MoE/curriculum_mean_obstacle_height/expert_{expert_id}",
                                        mean_h.item(),
                                        locs["it"],
                                    )
                except Exception as e:
                    # Never break training due to logging; keep it quiet unless debugging.
                    pass
            # everything else
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            # log the mean median velocity if it exists
            if "mean_median_vel" in locs:
                print(f"Logging mean median velocity: {locs['mean_median_vel']}")
                self.writer.add_scalar("Train/mean_median_velocity", locs["mean_median_vel"], locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            # -- Losses
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            # -- Rewards
            if self.alg.rnd:
                log_string += (
                    f"""{'Mean extrinsic reward:':>{pad}} {statistics.mean(locs['erewbuffer']):.2f}\n"""
                    f"""{'Mean intrinsic reward:':>{pad}} {statistics.mean(locs['irewbuffer']):.2f}\n"""
                )
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            # -- episode info
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    self.tot_time / (locs['it'] - locs['start_iter'] + 1)
                    * (locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])
                )
            )}\n"""
        )
        print(log_string)

    def _log_gate_probs(self, iteration: int, obs: torch.Tensor):
        """Log gate probabilities for visualization.
        
        Logs gate probs and morphology vectors for the first N environments
        to a JSONL file for real-time visualization.
        
        Args:
            iteration: Current training iteration.
            obs: Current observations tensor of shape (num_envs, obs_dim).
        """
        # Initialize log file if not done yet
        if self.gate_probs_log_file is None:
            self.gate_probs_log_file = os.path.join(self.log_dir, "gate_probs.jsonl")
            # Write header info (number of experts, morphology dim, etc.)
            header = {
                "type": "header",
                "num_experts": self.alg.policy.num_experts,
                "num_morphology_obs": self.alg.policy.num_morphology_obs,
                "num_envs_logged": self.gate_probs_num_envs,
                "routing_type": self.alg.policy.routing_type,
            }
            with open(self.gate_probs_log_file, "w") as f:
                f.write(json.dumps(header) + "\n")
        
        # Get gate probabilities for the first N envs
        policy = self.alg.policy
        num_envs = self.gate_probs_num_envs
        num_morph = policy.num_morphology_obs
        
        # Extract morphology from observations (last num_morph dims)
        morphology = obs[:num_envs, -num_morph:].detach().cpu()
        
        # Compute gate logits and probabilities
        with torch.no_grad():
            gate_logits = policy.gate(obs[:num_envs, -num_morph:]).cpu()
            gate_probs = torch.softmax(gate_logits, dim=-1)
            u = torch.rand_like(gate_logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            
        # Build log entry
        entry = {
            "type": "data",
            "iteration": iteration,
            "temperature": policy.tau,
        }
        
        # Add per-environment data
        for env_idx in range(num_envs):
            entry[f"env_{env_idx}_gate_logits"] = gate_logits[env_idx].tolist()
            entry[f"env_{env_idx}_gate_probs"] = gate_probs[env_idx].tolist()
            entry[f"env_{env_idx}_morphology"] = morphology[env_idx].tolist()
            entry[f"env_{env_idx}_gumbel_noise"] = gumbel_noise[env_idx].tolist()
        
        # Append to log file
        with open(self.gate_probs_log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save(self, path: str, infos=None):
        # -- Save model
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # -- Save RND model if used
        if self.alg.rnd:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        # -- Save observation normalizer if used
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()

        # save model
        torch.save(saved_dict, path)

        # upload model to external logging service
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        # -- Load model
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        # -- Load RND model if used
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        # -- Load observation normalizer if used
        if self.empirical_normalization:
            if resumed_training:
                # if a previous training is resumed, the actor/student normalizer is loaded for the actor/student
                # and the critic/teacher normalizer is loaded for the critic/teacher
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                # if the training is not resumed but a model is loaded, this run must be distillation training following
                # an rl training. Thus the actor normalizer is loaded for the teacher model. The student's normalizer
                # is not loaded, as the observation space could differ from the previous rl training.
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        # -- load optimizer if used
        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        # -- PPO
        self.alg.policy.train()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.train()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    """
    Helper functions.
    """

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

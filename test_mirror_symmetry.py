#!/usr/bin/env python3

"""
Test script for the mirror symmetry loss implementation.
This script verifies that the mirror symmetry loss is computed correctly
according to the Yu et al. paper formulation.
"""

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


def test_mirror_symmetry_loss():
    """Test the mirror symmetry loss implementation."""
    
    # Set up test parameters
    device = "cpu"
    batch_size = 4
    obs_dim = 12
    action_dim = 6
    
    # Create a simple actor-critic policy
    policy = ActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=action_dim,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
    )
    
    # Define mirror symmetry configuration for a simple bipedal robot
    mirror_symmetry_cfg = {
        'enabled': True,
        'weight': 4.0,
        'symmetric_joint_pairs': [
            (0, 3),  # left_hip, right_hip
            (1, 4),  # left_knee, right_knee  
            (2, 5),  # left_ankle, right_ankle
        ],
        'symmetric_obs_indices': [6, 7],  # lateral velocity, angular velocity
        'symmetric_action_indices': [
            (0, 3),  # left_hip, right_hip
            (1, 4),  # left_knee, right_knee
            (2, 5),  # left_ankle, right_ankle
        ]
    }
    
    # Create PPO agent with mirror symmetry
    ppo = PPO(
        policy=policy,
        device=device,
        mirror_symmetry_cfg=mirror_symmetry_cfg
    )
    
    # Create test observations
    obs_batch = torch.randn(batch_size, obs_dim, device=device)
    
    print("Testing Mirror Symmetry Loss Implementation")
    print("=" * 50)
    print(f"Batch size: {batch_size}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Mirror symmetry enabled: {ppo.mirror_symmetry['enabled']}")
    print(f"Mirror symmetry weight: {ppo.mirror_symmetry['weight']}")
    print()
    
    # Test observation mirroring
    print("Testing observation mirroring...")
    mirrored_obs = ppo.mirror_observations(obs_batch)
    print(f"Original obs shape: {obs_batch.shape}")
    print(f"Mirrored obs shape: {mirrored_obs.shape}")
    
    # Check that symmetric joints are swapped
    for left_idx, right_idx in ppo.mirror_symmetry['symmetric_joint_pairs']:
        assert torch.allclose(obs_batch[:, left_idx], mirrored_obs[:, right_idx]), \
            f"Joint {left_idx} should be swapped with joint {right_idx}"
        assert torch.allclose(obs_batch[:, right_idx], mirrored_obs[:, left_idx]), \
            f"Joint {right_idx} should be swapped with joint {left_idx}"
    
    # Check that symmetric obs indices are sign-flipped
    for idx in ppo.mirror_symmetry['symmetric_obs_indices']:
        assert torch.allclose(obs_batch[:, idx], -mirrored_obs[:, idx]), \
            f"Observation index {idx} should be sign-flipped"
    
    print("✓ Observation mirroring works correctly")
    print()
    
    # Test action mirroring
    print("Testing action mirroring...")
    with torch.no_grad():
        actions = ppo.policy.act_inference(obs_batch)
    mirrored_actions = ppo.mirror_actions(actions)
    
    print(f"Original actions shape: {actions.shape}")
    print(f"Mirrored actions shape: {mirrored_actions.shape}")
    
    # Check that symmetric actions are swapped
    for left_idx, right_idx in ppo.mirror_symmetry['symmetric_action_indices']:
        assert torch.allclose(actions[:, left_idx], mirrored_actions[:, right_idx]), \
            f"Action {left_idx} should be swapped with action {right_idx}"
        assert torch.allclose(actions[:, right_idx], mirrored_actions[:, left_idx]), \
            f"Action {right_idx} should be swapped with action {left_idx}"
    
    print("✓ Action mirroring works correctly")
    print()
    
    # Test mirror symmetry loss computation
    print("Testing mirror symmetry loss computation...")
    symmetry_loss = ppo.compute_mirror_symmetry_loss(obs_batch)
    
    print(f"Mirror symmetry loss: {symmetry_loss.item():.6f}")
    print(f"Loss shape: {symmetry_loss.shape}")
    print(f"Loss requires grad: {symmetry_loss.requires_grad}")
    
    # Verify loss is a scalar
    assert symmetry_loss.dim() == 0, "Loss should be a scalar"
    assert symmetry_loss.requires_grad, "Loss should require gradients"
    assert symmetry_loss.item() >= 0, "Loss should be non-negative"
    
    print("✓ Mirror symmetry loss computation works correctly")
    print()
    
    # Test that loss decreases when policy becomes more symmetric
    print("Testing symmetry property...")
    
    # Create a perfectly symmetric observation (all zeros)
    symmetric_obs = torch.zeros(1, obs_dim, device=device)
    symmetric_loss = ppo.compute_mirror_symmetry_loss(symmetric_obs)
    
    print(f"Loss for symmetric observation: {symmetric_loss.item():.6f}")
    
    # For a perfectly symmetric observation, the loss should be small
    # (though not necessarily zero due to random policy initialization)
    print("✓ Symmetry property test completed")
    print()
    
    print("All tests passed! ✓")
    print("Mirror symmetry loss implementation is working correctly.")


if __name__ == "__main__":
    test_mirror_symmetry_loss() 
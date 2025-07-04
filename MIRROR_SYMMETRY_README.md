# Mirror Symmetry Loss Implementation

This document describes the implementation of the mirror symmetry loss from the paper "Learning Symmetric and Low-Energy Locomotion" by Yu et al. (SIGGRAPH 2018).

## Overview

The mirror symmetry loss encourages symmetric locomotion gaits by penalizing asymmetric actions for symmetric morphologies. Unlike traditional approaches that enforce symmetry on states (which requires observing full gait cycles), this method enforces symmetry on actions, avoiding delayed reward issues.

## Mathematical Formulation

The mirror symmetry loss is defined as:

```
L_sym(θ) = Σ(i=0 to B) ||π_θ(s_i) - Ψ_a(π_θ(Ψ_o(s_i)))||²
```

Where:
- `π_θ(s_i)`: Policy output (mean action) for original state s_i
- `Ψ_o(s_i)`: Mirror transformation of state s_i (swap left/right components)
- `π_θ(Ψ_o(s_i))`: Policy output for the mirrored state
- `Ψ_a(π_θ(Ψ_o(s_i)))`: Mirror transformation of policy output from mirrored state

## Key Insight

If a character has symmetric morphology, then the action it takes in some pose should be the mirrored version of the action taken when the character is in the mirrored pose.

**Example**: If a humanoid is in state "left foot forward" and the policy says "lift right foot", then when the humanoid is in the mirrored state "right foot forward", the policy should say "lift left foot".

## Configuration

Add the following to your algorithm configuration:

```yaml
algorithm:
  mirror_symmetry_cfg:
    enabled: true          # Enable mirror symmetry loss
    weight: 4.0            # Loss coefficient (default from Yu et al. paper)
    
    # Define symmetric joint pairs [left_joint_idx, right_joint_idx]
    symmetric_joint_pairs:
      - [0, 3]   # left_hip_yaw, right_hip_yaw
      - [1, 4]   # left_hip_roll, right_hip_roll  
      - [2, 5]   # left_hip_pitch, right_hip_pitch
      - [6, 9]   # left_knee, right_knee
      - [7, 10]  # left_ankle_pitch, right_ankle_pitch
      - [8, 11]  # left_ankle_roll, right_ankle_roll
    
    # Observation indices that should be sign-flipped when mirrored
    symmetric_obs_indices:
      - 13  # lateral velocity
      - 14  # angular velocity around z-axis
    
    # Action indices that should be swapped when mirrored
    symmetric_action_indices:
      - [0, 3]   # left_hip_yaw, right_hip_yaw
      - [1, 4]   # left_hip_roll, right_hip_roll  
      - [2, 5]   # left_hip_pitch, right_hip_pitch
      - [6, 9]   # left_knee, right_knee
      - [7, 10]  # left_ankle_pitch, right_ankle_pitch
      - [8, 11]  # left_ankle_roll, right_ankle_roll
```

## Usage Example

```python
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

# Define mirror symmetry configuration
mirror_symmetry_cfg = {
    'enabled': True,
    'weight': 4.0,
    'symmetric_joint_pairs': [(0, 3), (1, 4), (2, 5)],
    'symmetric_obs_indices': [6, 7],
    'symmetric_action_indices': [(0, 3), (1, 4), (2, 5)]
}

# Create PPO agent with mirror symmetry
ppo = PPO(
    policy=policy,
    mirror_symmetry_cfg=mirror_symmetry_cfg
)

# The mirror symmetry loss will be automatically computed and added to the total loss
loss_dict = ppo.update()
print(f"Mirror symmetry loss: {loss_dict['mirror_symmetry']}")
```

## Implementation Details

### Core Methods

1. **`mirror_observations(obs_batch)`**: Mirrors observations by swapping symmetric joint pairs and flipping signs of specified indices.

2. **`mirror_actions(actions_batch)`**: Mirrors actions by swapping symmetric action pairs.

3. **`compute_mirror_symmetry_loss(obs_batch)`**: Computes the full mirror symmetry loss according to Yu et al.'s formulation.

### Integration with PPO

The mirror symmetry loss is seamlessly integrated into the PPO update loop:

1. **Initialization**: Mirror symmetry configuration is processed during PPO initialization
2. **Loss Computation**: Mirror symmetry loss is computed for each mini-batch during training
3. **Gradient Flow**: The loss is added to the total PPO loss and gradients flow through the policy network
4. **Logging**: Mirror symmetry loss is tracked and returned in the loss dictionary

### Performance Considerations

- **Computational Overhead**: The mirror symmetry loss requires two additional forward passes through the policy network per mini-batch
- **Memory Usage**: Minimal additional memory overhead (only for mirrored observations/actions)
- **Gradient Computation**: Uses `.detach()` on the target to prevent double gradients

## Configuration Guidelines

### Symmetric Joint Pairs
- Identify left-right joint pairs in your robot's kinematic structure
- Use joint indices from your observation/action space
- Example for humanoid: hip, knee, ankle joints

### Symmetric Observation Indices
- Include observations that should be sign-flipped when mirrored
- Common examples: lateral velocity, angular velocity around vertical axis
- Do NOT include joint positions/velocities (these are handled by joint pairs)

### Symmetric Action Indices
- Should match the symmetric joint pairs
- Use the same indices as in `symmetric_joint_pairs`

### Weight Selection
- Default: 4.0 (from Yu et al. paper)
- Higher values: Stronger symmetry enforcement
- Lower values: Weaker symmetry enforcement
- Tune based on your specific task and robot

## Benefits

1. **Natural Gaits**: Produces more natural, symmetric locomotion patterns
2. **Energy Efficiency**: Symmetric gaits are typically more energy-efficient
3. **Stability**: Symmetric locomotion reduces risk of falling
4. **Generalization**: Works across different robot morphologies
5. **No Motion Capture**: Doesn't require reference motion data

## Comparison with Existing Symmetry Methods

| Feature | Mirror Symmetry Loss (Yu et al.) | Existing RSL-RL Symmetry |
|---------|----------------------------------|---------------------------|
| **Approach** | Action-level symmetry enforcement | Data augmentation + state symmetry |
| **Complexity** | Simple configuration | Requires environment-specific functions |
| **Delayed Rewards** | No (immediate action-level loss) | Yes (trajectory-level metrics) |
| **Implementation** | Standalone, morphology-agnostic | Environment-dependent |
| **Computational Cost** | 2 extra forward passes | Variable (depends on augmentation) |

## Testing

Run the test script to verify the implementation:

```bash
cd rsl_rl
python test_mirror_symmetry.py
```

The test verifies:
- Observation mirroring correctness
- Action mirroring correctness  
- Loss computation functionality
- Gradient flow
- Basic symmetry properties

## Troubleshooting

### Common Issues

1. **Index Errors**: Ensure joint indices are within observation/action dimensions
2. **Mismatched Pairs**: Verify symmetric joint pairs are correctly identified
3. **No Symmetry Effect**: Check that weight is non-zero and enabled=True
4. **High Loss Values**: May indicate incorrect joint pair mappings

### Debugging Tips

1. **Visualize Mirroring**: Print original vs mirrored observations to verify correctness
2. **Check Gradients**: Ensure `requires_grad=True` for the symmetry loss
3. **Monitor Loss**: Track mirror symmetry loss during training to see if it decreases
4. **Test Configuration**: Use the provided test script with your specific configuration

## References

- Yu, W., Turk, G., & Liu, C. K. (2018). Learning symmetric and low-energy locomotion. ACM Transactions on Graphics, 37(4), 1-12.
- Paper URL: https://arxiv.org/abs/1801.08093
- Video: https://www.youtube.com/watch?v=zkH90rU-uew 
#!/usr/bin/env python3
"""
Compositional Neural Networks for Multitask RL.

This module contains compositional network implementations adapted from
offline_compositional_rl_datasets/algos/cp_iql.py, with PyTorch 2.5+ compatibility fixes.

The compositional architecture uses hierarchical modules:
- Obstacle module → Object module → Subtask module → Robot module

Can be imported for use in training scripts or run directly as a test suite.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Optional

# Device configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================================
# COPIED NETWORK IMPLEMENTATIONS FROM cp_iql.py
# ============================================================================

def fanin_init(tensor):
    """Initialize the weights of a layer with fan-in initialization."""
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


class CompositionalMlp(nn.Module):
    """Compositional MLP module."""

    def __init__(
        self,
        sizes: Sequence[Sequence[int]],
        num_modules: Sequence[int],
        module_assignment_positions: Sequence[int],
        module_inputs: Sequence[str],
        interface_depths: Sequence[int],
        graph_structure: Sequence[Sequence[int]],
        init_w: float = 3e-3,
        hidden_activation: nn.Module = nn.ReLU,
        output_activation: nn.Module = nn.Identity,
        hidden_init: Optional[nn.Module] = fanin_init,
        b_init_value: float = 0.1,
        layer_norm: bool = False,
        layer_norm_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.sizes = sizes
        self.num_modules = num_modules
        self.module_assignment_positions = module_assignment_positions
        self.module_inputs = module_inputs  # keys in a dict
        self.interface_depths = interface_depths
        self.graph_structure = graph_structure
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        self.count = 0

        # Pre-allocate module_list to support any graph_structure order
        # We need module_list[j] to work for any j, regardless of processing order
        max_module_idx = max(max(level) for level in graph_structure) if graph_structure else 0
        self.module_list = nn.ModuleList([None] * (max_module_idx + 1))

        for graph_depth in range(len(graph_structure)):
            for j in graph_structure[graph_depth]:
                self.module_list[j] = nn.ModuleDict()
                self.module_list[j]["pre_interface"] = nn.ModuleList()
                self.module_list[j]["post_interface"] = nn.ModuleList()
                for k in range(num_modules[j]):
                    layers_pre = []
                    layers_post = []
                    for i in range(len(sizes[j]) - 1):
                        if i == interface_depths[j]:
                            input_size = sum(
                                sizes[j_prev][-1]
                                for j_prev in graph_structure[graph_depth - 1]
                            )
                            input_size += sizes[j][i]
                        else:
                            input_size = sizes[j][i]

                        fc = nn.Linear(input_size, sizes[j][i + 1])
                        if (
                            graph_depth < len(graph_structure) - 1
                            or i < len(sizes[j]) - 2
                        ):
                            hidden_init(fc.weight)
                            fc.bias.data.fill_(b_init_value)
                            act = hidden_activation
                            layer_norm_this = layer_norm
                        else:
                            fc.weight.data.uniform_(-init_w, init_w)
                            fc.bias.data.uniform_(-init_w, init_w)
                            act = output_activation
                            layer_norm_this = None

                        if layer_norm_this is not None:
                            new_layer = [fc, nn.LayerNorm(sizes[j][i + 1]), act()]
                        else:
                            new_layer = [fc, act()]

                        if i < interface_depths[j]:
                            layers_pre += new_layer
                        else:
                            layers_post += new_layer
                    if layers_pre:
                        self.module_list[j]["pre_interface"].append(
                            nn.Sequential(*layers_pre)
                        )
                    else:
                        self.module_list[j]["pre_interface"].append(nn.Identity())
                    self.module_list[j]["post_interface"].append(
                        nn.Sequential(*layers_post)
                    )

    def forward(self, input_val: torch.Tensor, return_preactivations: bool = False):
        """Forward pass."""
        if len(input_val.shape) > 2:
            input_val = input_val.squeeze(0)

        if return_preactivations:
            raise NotImplementedError("TODO: implement return preactivations")
        
        x = None
        for graph_depth in range(len(self.graph_structure)):
            x_post = []
            for j in self.graph_structure[graph_depth]:
                if len(input_val.shape) == 1:
                    # Single sample case
                    x_pre = input_val[self.module_inputs[j]]
                    onehot = input_val[self.module_assignment_positions[j]]
                    module_index = onehot.nonzero()[0]
                    x_pre = self.module_list[j]["pre_interface"][module_index](x_pre)
                    if x is not None:
                        x_pre = torch.cat((x, x_pre), dim=-1)
                    x_post.append(
                        self.module_list[j]["post_interface"][module_index](x_pre)
                    )
                else:
                    # Batch case
                    x_post_tmp = torch.empty(input_val.shape[0], self.sizes[j][-1]).to(
                        DEVICE
                    )
                    x_pre = input_val[:, self.module_inputs[j]]
                    onehot = input_val[:, self.module_assignment_positions[j]]
                    
                    # FIX: Updated for newer PyTorch versions
                    module_indices = onehot.nonzero(as_tuple=True)
                    assert (
                        module_indices[0]
                        == torch.arange(module_indices[0].shape[0]).to(DEVICE)
                    ).all()
                    module_indices_1 = module_indices[1]
                    
                    for module_idx in range(self.num_modules[j]):
                        mask_inputs_for_this_module = module_indices_1 == module_idx
                        mask_to_input_idx = mask_inputs_for_this_module.nonzero(as_tuple=False)
                        x_pre_this_module = self.module_list[j]["pre_interface"][
                            module_idx
                        ](x_pre[mask_inputs_for_this_module])
                        if x is not None:
                            x_pre_this_module = torch.cat(
                                (x[mask_inputs_for_this_module], x_pre_this_module),
                                dim=-1,
                            )
                        x_post_this_module = self.module_list[j]["post_interface"][
                            module_idx
                        ](x_pre_this_module)
                        mask_to_input_idx = mask_to_input_idx.expand(
                            mask_to_input_idx.shape[0], x_post_this_module.shape[1]
                        )
                        x_post_tmp.scatter_(0, mask_to_input_idx, x_post_this_module)
                    x_post.append(x_post_tmp)
            x = torch.cat(x_post, dim=-1)
        return x


class CompositionalEncoder(nn.Module):
    """Compositional encoder for state inputs."""
    
    def __init__(
        self,
        encoder_kwargs: dict,
        observation_shape: Sequence[int],
        init_w: float = 3e-3,
    ):
        super().__init__()
        
        self._observation_shape = observation_shape
        self.encoder_kwargs = encoder_kwargs
        sizes = encoder_kwargs["sizes"]
        output_dim = encoder_kwargs["output_dim"]
        num_modules = encoder_kwargs["num_modules"]
        module_assignment_positions = encoder_kwargs["module_assignment_positions"]
        module_inputs = encoder_kwargs["module_inputs"]
        interface_depths = encoder_kwargs["interface_depths"]
        graph_structure = encoder_kwargs["graph_structure"]
        
        sizes = list(sizes)
        for j in range(len(sizes)):
            input_size = len(module_inputs[j])
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [output_dim]

        self._feature_size = sizes[-1][-1]

        self.comp_mlp = CompositionalMlp(
            sizes=sizes,
            num_modules=num_modules,
            module_assignment_positions=module_assignment_positions,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure,
            init_w=init_w,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.comp_mlp.forward(x)
    
    @property
    def feature_size(self):
        return self._feature_size


class CompositionalEncoderWithAction(nn.Module):
    """Compositional encoder that takes both state and action as input."""
    
    def __init__(
        self,
        encoder_kwargs: dict,
        observation_shape: Sequence[int],
        action_size: int,
        init_w: float = 3e-3,
    ):
        super().__init__()
        
        self._observation_shape = observation_shape
        self._action_size = action_size
        self.encoder_kwargs = encoder_kwargs
        sizes = encoder_kwargs["sizes"]
        output_dim = encoder_kwargs["output_dim"]
        num_modules = encoder_kwargs["num_modules"]
        module_assignment_positions = encoder_kwargs["module_assignment_positions"]
        module_inputs = encoder_kwargs["module_inputs"]
        interface_depths = encoder_kwargs["interface_depths"]
        graph_structure = encoder_kwargs["graph_structure"]
        
        sizes = list(sizes)
        for j in range(len(sizes)):
            input_size = len(module_inputs[j])
            sizes[j] = [input_size] + list(sizes[j])
            if j in graph_structure[-1]:
                sizes[j] = sizes[j] + [output_dim]

        self._feature_size = sizes[-1][-1]

        self.comp_mlp = CompositionalMlp(
            sizes=sizes,
            num_modules=num_modules,
            module_assignment_positions=module_assignment_positions,
            module_inputs=module_inputs,
            interface_depths=interface_depths,
            graph_structure=graph_structure,
            init_w=init_w,
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Concatenate state and action
        x = torch.cat([x, action], dim=1)
        return self.comp_mlp.forward(x)
    
    @property
    def feature_size(self):
        return self._feature_size


def create_cp_encoderfactory_params(with_action=False, output_dim=None):
    """Create encoder factory parameters for compositional networks.
    
    This is based on the CompoSuite environment structure.
    """
    obs_dim = 93
    act_dim = 8
    
    # Observation space structure from CompoSuite
    observation_positions = {
        'object-state': np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]), 
        'obstacle-state': np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]), 
        'goal-state': np.array([28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]), 
        'object_id': np.array([45, 46, 47, 48]), 
        'robot_id': np.array([49, 50, 51, 52]), 
        'obstacle_id': np.array([53, 54, 55, 56]), 
        'subtask_id': np.array([57, 58, 59, 60]), 
        'robot0_proprio-state': np.array([61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92])
    }
    
    if with_action:
        observation_positions["action"] = np.array([93, 94, 95, 96, 97, 98, 99, 100])

    # Network architecture
    # Configured for reversed hierarchy [[3], [2], [1], [0]]: Robot → Subtask → Object → Obstacle
    # Robot: 32→32, Subtask: 17→32 pre + 64→32 post, Object: 14→64→64 pre + 96→64 post, Obstacle: 14→64→64→64 pre + 128→8 post
    # sizes = ((64, 64, 64), (64, 64, 64), (32, 32), (32,))
    sizes = ((38, 38, 38), (38, 38, 38), (76, 76), (76,))
    module_names = ["obstacle_id", "object_id", "subtask_id", "robot_id"]
    module_input_names = [
        "obstacle-state",
        "object-state",
        "goal-state",
    ]
    if with_action:
        module_input_names.append(["robot0_proprio-state", "action"])
    else:
        module_input_names.append("robot0_proprio-state")

    module_assignment_positions = [observation_positions[key] for key in module_names]
    # interface_depths: Robot(-1=all post), Subtask(1=after 1st), Object(2=after 2nd), Obstacle(3=after 3rd)
    interface_depths = [3, 2, 1, -1]  # Obstacle, Object, Subtask, Robot
    graph_structure = [[3], [2], [1], [0]]  # Reversed: Robot → Subtask → Object → Obstacle
    num_modules = [len(onehot_pos) for onehot_pos in module_assignment_positions]

    module_inputs = []
    for key in module_input_names:
        if isinstance(key, list):
            module_inputs.append(
                np.concatenate([observation_positions[k] for k in key], axis=0)
            )
        else:
            module_inputs.append(observation_positions[key])

    encoder_kwargs = {
        "sizes": sizes,
        "obs_dim": obs_dim,
        "output_dim": output_dim if output_dim is not None else act_dim,
        "num_modules": num_modules,
        "module_assignment_positions": module_assignment_positions,
        "module_inputs": module_inputs,
        "interface_depths": interface_depths,
        "graph_structure": graph_structure,
    }

    return encoder_kwargs


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_basic_instantiation():
    """Test 1: Can we instantiate the networks?"""
    print("\n" + "="*60)
    print("TEST 1: Basic Network Instantiation")
    print("="*60)
    
    try:
        # Create encoder parameters
        encoder_kwargs = create_cp_encoderfactory_params(with_action=False, output_dim=8)
        print("✓ Created encoder parameters")
        
        # Instantiate state encoder
        state_encoder = CompositionalEncoder(
            encoder_kwargs=encoder_kwargs,
            observation_shape=(93,)
        ).to(DEVICE)
        print(f"✓ Created state encoder with feature size: {state_encoder.feature_size}")
        
        # Instantiate state-action encoder (for Q-function)
        encoder_kwargs_q = create_cp_encoderfactory_params(with_action=True, output_dim=1)
        q_encoder = CompositionalEncoderWithAction(
            encoder_kwargs=encoder_kwargs_q,
            observation_shape=(93,),
            action_size=8
        ).to(DEVICE)
        print(f"✓ Created Q encoder with feature size: {q_encoder.feature_size}")
        
        return True
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_single():
    """Test 2: Forward pass with single sample"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass - Single Sample")
    print("="*60)
    
    try:
        # Create a dummy observation (93-dim)
        # Structure: [object_state(14), obstacle_state(14), goal_state(17), 
        #             object_id(4), robot_id(4), obstacle_id(4), subtask_id(4), 
        #             robot_proprio(32)]
        obs = torch.zeros(93).to(DEVICE)
        
        # Set one-hot encodings for task components
        obs[45] = 1.0  # object_id = 0 (e.g., Plate)
        obs[49] = 1.0  # robot_id = 0 (e.g., IIWA)
        obs[53] = 1.0  # obstacle_id = 0 (e.g., ObjectDoor)
        obs[57] = 1.0  # subtask_id = 0 (e.g., Trashcan)
        
        # Random values for continuous states
        obs[:14] = torch.randn(14).to(DEVICE)  # object state
        obs[14:28] = torch.randn(14).to(DEVICE)  # obstacle state
        obs[28:45] = torch.randn(17).to(DEVICE)  # goal state
        obs[61:] = torch.randn(32).to(DEVICE)  # robot proprio
        
        # Create encoder and forward pass
        encoder_kwargs = create_cp_encoderfactory_params(with_action=False, output_dim=8)
        state_encoder = CompositionalEncoder(
            encoder_kwargs=encoder_kwargs,
            observation_shape=(93,)
        ).to(DEVICE)
        
        output = state_encoder(obs)
        print(f"✓ Single sample forward pass successful")
        print(f"  Input shape: {obs.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: torch.Size([8])")
        
        assert output.shape == (8,), f"Expected shape (8,), got {output.shape}"
        print(f"✓ Output shape correct!")
        
        return True
    except Exception as e:
        print(f"✗ Single forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass_batch():
    """Test 3: Forward pass with batch"""
    print("\n" + "="*60)
    print("TEST 3: Forward Pass - Batch")
    print("="*60)
    
    try:
        batch_size = 256
        
        # Create dummy batch observations
        obs = torch.zeros(batch_size, 93).to(DEVICE)
        
        # For variety, set different task combinations
        for i in range(batch_size):
            # Cycle through different tasks
            obs[i, 45 + (i % 4)] = 1.0  # object_id
            obs[i, 49 + (i % 4)] = 1.0  # robot_id
            obs[i, 53 + (i % 4)] = 1.0  # obstacle_id
            obs[i, 57 + (i % 4)] = 1.0  # subtask_id
            
            # Random continuous states
            obs[i, :14] = torch.randn(14).to(DEVICE)
            obs[i, 14:28] = torch.randn(14).to(DEVICE)
            obs[i, 28:45] = torch.randn(17).to(DEVICE)
            obs[i, 61:] = torch.randn(32).to(DEVICE)
        
        # Test state encoder
        encoder_kwargs = create_cp_encoderfactory_params(with_action=False, output_dim=8)
        state_encoder = CompositionalEncoder(
            encoder_kwargs=encoder_kwargs,
            observation_shape=(93,)
        ).to(DEVICE)
        
        output = state_encoder(obs)
        print(f"✓ Batch forward pass successful")
        print(f"  Input shape: {obs.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: torch.Size([{batch_size}, 8])")
        
        assert output.shape == (batch_size, 8), f"Expected shape ({batch_size}, 8), got {output.shape}"
        print(f"✓ Output shape correct!")
        
        return True
    except Exception as e:
        print(f"✗ Batch forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_q_function_forward():
    """Test 4: Q-function encoder with state + action"""
    print("\n" + "="*60)
    print("TEST 4: Q-Function Forward Pass")
    print("="*60)
    
    try:
        batch_size = 256
        
        # Create dummy observations and actions
        obs = torch.zeros(batch_size, 93).to(DEVICE)
        actions = torch.randn(batch_size, 8).to(DEVICE)
        
        for i in range(batch_size):
            obs[i, 45 + (i % 4)] = 1.0  # object_id
            obs[i, 49 + (i % 4)] = 1.0  # robot_id
            obs[i, 53 + (i % 4)] = 1.0  # obstacle_id
            obs[i, 57 + (i % 4)] = 1.0  # subtask_id
            
            obs[i, :14] = torch.randn(14).to(DEVICE)
            obs[i, 14:28] = torch.randn(14).to(DEVICE)
            obs[i, 28:45] = torch.randn(17).to(DEVICE)
            obs[i, 61:] = torch.randn(32).to(DEVICE)
        
        # Test Q encoder
        encoder_kwargs_q = create_cp_encoderfactory_params(with_action=True, output_dim=1)
        q_encoder = CompositionalEncoderWithAction(
            encoder_kwargs=encoder_kwargs_q,
            observation_shape=(93,),
            action_size=8
        ).to(DEVICE)
        
        output = q_encoder(obs, actions)
        print(f"✓ Q-function forward pass successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action shape: {actions.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: torch.Size([{batch_size}, 1])")
        
        assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
        print(f"✓ Output shape correct!")
        
        return True
    except Exception as e:
        print(f"✗ Q-function forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_td3bc_integration():
    """Test 5: Integration with TD3+BC style networks"""
    print("\n" + "="*60)
    print("TEST 5: TD3+BC Integration Test")
    print("="*60)
    
    try:
        state_dim = 93
        action_dim = 8
        max_action = 1.0
        
        # Create compositional actor (deterministic policy)
        encoder_kwargs = create_cp_encoderfactory_params(with_action=False, output_dim=action_dim)
        actor = CompositionalEncoder(
            encoder_kwargs=encoder_kwargs,
            observation_shape=(state_dim,)
        ).to(DEVICE)
        
        # Wrap with Tanh for bounded actions
        class CompositionalActor(nn.Module):
            def __init__(self, encoder, max_action):
                super().__init__()
                self.encoder = encoder
                self.max_action = max_action
                self.tanh = nn.Tanh()
            
            def forward(self, state):
                return self.max_action * self.tanh(self.encoder(state))
            
            def act(self, state, device):
                with torch.no_grad():
                    state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
                    return self(state_tensor).cpu().data.numpy().flatten()
        
        comp_actor = CompositionalActor(actor, max_action).to(DEVICE)
        
        # Create compositional critics
        encoder_kwargs_q = create_cp_encoderfactory_params(with_action=True, output_dim=1)
        critic_1 = CompositionalEncoderWithAction(
            encoder_kwargs=encoder_kwargs_q,
            observation_shape=(state_dim,),
            action_size=action_dim
        ).to(DEVICE)
        critic_2 = CompositionalEncoderWithAction(
            encoder_kwargs=encoder_kwargs_q,
            observation_shape=(state_dim,),
            action_size=action_dim
        ).to(DEVICE)
        
        # Test forward passes
        batch_size = 256
        obs = torch.zeros(batch_size, 93).to(DEVICE)
        
        for i in range(batch_size):
            obs[i, 45 + (i % 4)] = 1.0
            obs[i, 49 + (i % 4)] = 1.0
            obs[i, 53 + (i % 4)] = 1.0
            obs[i, 57 + (i % 4)] = 1.0
            obs[i, :14] = torch.randn(14).to(DEVICE)
            obs[i, 14:28] = torch.randn(14).to(DEVICE)
            obs[i, 28:45] = torch.randn(17).to(DEVICE)
            obs[i, 61:] = torch.randn(32).to(DEVICE)
        
        # Actor forward pass
        actions = comp_actor(obs)
        print(f"✓ Actor forward pass: {obs.shape} -> {actions.shape}")
        assert actions.shape == (batch_size, action_dim)
        assert (actions.abs() <= max_action).all(), "Actions not properly bounded!"
        
        # Critic forward passes
        q1 = critic_1(obs, actions)
        q2 = critic_2(obs, actions)
        print(f"✓ Critic 1 forward pass: ({obs.shape}, {actions.shape}) -> {q1.shape}")
        print(f"✓ Critic 2 forward pass: ({obs.shape}, {actions.shape}) -> {q2.shape}")
        assert q1.shape == (batch_size, 1)
        assert q2.shape == (batch_size, 1)
        
        # Test single sample action (for evaluation)
        single_obs = obs[0].cpu().numpy()
        single_action = comp_actor.act(single_obs, DEVICE)
        print(f"✓ Single action generation: {single_obs.shape} -> {single_action.shape}")
        assert single_action.shape == (action_dim,)
        
        print(f"\n✓ All TD3+BC integration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ TD3+BC integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test 6: Verify gradients flow through the network"""
    print("\n" + "="*60)
    print("TEST 6: Gradient Flow Test")
    print("="*60)
    
    try:
        # Create actor and optimizer
        encoder_kwargs = create_cp_encoderfactory_params(with_action=False, output_dim=8)
        actor = CompositionalEncoder(
            encoder_kwargs=encoder_kwargs,
            observation_shape=(93,)
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        
        # Create dummy batch
        batch_size = 32
        obs = torch.zeros(batch_size, 93).to(DEVICE)
        
        for i in range(batch_size):
            obs[i, 45] = 1.0  # object_id
            obs[i, 49] = 1.0  # robot_id
            obs[i, 53] = 1.0  # obstacle_id
            obs[i, 57] = 1.0  # subtask_id
            obs[i, :14] = torch.randn(14).to(DEVICE)
            obs[i, 14:28] = torch.randn(14).to(DEVICE)
            obs[i, 28:45] = torch.randn(17).to(DEVICE)
            obs[i, 61:] = torch.randn(32).to(DEVICE)
        
        # Forward pass
        output = actor(obs)
        
        # Compute dummy loss and backward
        target = torch.randn_like(output).to(DEVICE)
        loss = ((output - target) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check if gradients were computed
        has_grad = False
        for name, param in actor.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No gradients computed!"
        print(f"✓ Gradients computed successfully")
        print(f"✓ Loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_task_batch():
    """Test 7: Mixed batch with all 4 robot/object/obstacle/subtask combinations"""
    print("\n" + "="*60)
    print("TEST 7: Mixed Task Batch (Stress Test)")
    print("="*60)
    
    try:
        batch_size = 1024  # Larger batch
        
        # Create encoder
        encoder_kwargs = create_cp_encoderfactory_params(with_action=False, output_dim=8)
        state_encoder = CompositionalEncoder(
            encoder_kwargs=encoder_kwargs,
            observation_shape=(93,)
        ).to(DEVICE)
        
        # Create batch with ALL possible task combinations (repeated to fill batch)
        obs = torch.zeros(batch_size, 93).to(DEVICE)
        
        # Generate all 256 unique task combinations
        task_combinations = []
        for robot in range(4):
            for obj in range(4):
                for obst in range(4):
                    for subtask in range(4):
                        task_combinations.append((robot, obj, obst, subtask))
        
        # Fill batch by repeating task combinations
        for i in range(batch_size):
            robot, obj, obst, subtask = task_combinations[i % 256]
            
            # Set one-hot encodings
            obs[i, 45 + obj] = 1.0
            obs[i, 49 + robot] = 1.0
            obs[i, 53 + obst] = 1.0
            obs[i, 57 + subtask] = 1.0
            
            # Random continuous states
            obs[i, :14] = torch.randn(14).to(DEVICE)
            obs[i, 14:28] = torch.randn(14).to(DEVICE)
            obs[i, 28:45] = torch.randn(17).to(DEVICE)
            obs[i, 61:] = torch.randn(32).to(DEVICE)
        
        print(f"  Created {batch_size} samples with 256 unique task combinations (repeated)")
        
        # Forward pass
        output = state_encoder(obs)
        
        print(f"✓ Forward pass with {batch_size} samples successful")
        print(f"  Output shape: {output.shape}")
        assert output.shape == (batch_size, 8)
        
        # Check for NaN or Inf
        assert not torch.isnan(output).any(), "NaN values in output!"
        assert not torch.isinf(output).any(), "Inf values in output!"
        print(f"✓ No NaN or Inf values")
        
        # Check output statistics
        print(f"  Output mean: {output.mean().item():.4f}")
        print(f"  Output std: {output.std().item():.4f}")
        print(f"  Output min: {output.min().item():.4f}")
        print(f"  Output max: {output.max().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mixed task batch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop_simulation():
    """Test 8: Simulate actual TD3+BC training loop"""
    print("\n" + "="*60)
    print("TEST 8: TD3+BC Training Loop Simulation")
    print("="*60)
    
    try:
        state_dim = 93
        action_dim = 8
        max_action = 1.0
        batch_size = 256
        
        # Create networks
        encoder_kwargs = create_cp_encoderfactory_params(with_action=False, output_dim=action_dim)
        actor_encoder = CompositionalEncoder(
            encoder_kwargs=encoder_kwargs,
            observation_shape=(state_dim,)
        ).to(DEVICE)
        
        class CompActor(nn.Module):
            def __init__(self, encoder, max_action):
                super().__init__()
                self.encoder = encoder
                self.max_action = max_action
                self.tanh = nn.Tanh()
            def forward(self, state):
                return self.max_action * self.tanh(self.encoder(state))
        
        actor = CompActor(actor_encoder, max_action).to(DEVICE)
        actor_target = CompActor(
            CompositionalEncoder(encoder_kwargs, (state_dim,)).to(DEVICE),
            max_action
        ).to(DEVICE)
        
        encoder_kwargs_q = create_cp_encoderfactory_params(with_action=True, output_dim=1)
        critic_1 = CompositionalEncoderWithAction(
            encoder_kwargs=encoder_kwargs_q,
            observation_shape=(state_dim,),
            action_size=action_dim
        ).to(DEVICE)
        critic_2 = CompositionalEncoderWithAction(
            encoder_kwargs=encoder_kwargs_q,
            observation_shape=(state_dim,),
            action_size=action_dim
        ).to(DEVICE)
        critic_1_target = CompositionalEncoderWithAction(
            encoder_kwargs=encoder_kwargs_q,
            observation_shape=(state_dim,),
            action_size=action_dim
        ).to(DEVICE)
        critic_2_target = CompositionalEncoderWithAction(
            encoder_kwargs=encoder_kwargs_q,
            observation_shape=(state_dim,),
            action_size=action_dim
        ).to(DEVICE)
        
        # Optimizers
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)
        
        # Simulate a few training steps
        for step in range(3):
            # Create dummy batch
            state = torch.zeros(batch_size, 93).to(DEVICE)
            action = torch.randn(batch_size, 8).to(DEVICE)
            reward = torch.randn(batch_size, 1).to(DEVICE)
            next_state = torch.zeros(batch_size, 93).to(DEVICE)
            done = torch.zeros(batch_size, 1).to(DEVICE)
            
            for i in range(batch_size):
                # Current state
                state[i, 45 + (i % 4)] = 1.0
                state[i, 49 + (i % 4)] = 1.0
                state[i, 53 + (i % 4)] = 1.0
                state[i, 57 + (i % 4)] = 1.0
                state[i, :14] = torch.randn(14).to(DEVICE)
                state[i, 14:28] = torch.randn(14).to(DEVICE)
                state[i, 28:45] = torch.randn(17).to(DEVICE)
                state[i, 61:] = torch.randn(32).to(DEVICE)
                
                # Next state
                next_state[i, 45 + (i % 4)] = 1.0
                next_state[i, 49 + (i % 4)] = 1.0
                next_state[i, 53 + (i % 4)] = 1.0
                next_state[i, 57 + (i % 4)] = 1.0
                next_state[i, :14] = torch.randn(14).to(DEVICE)
                next_state[i, 14:28] = torch.randn(14).to(DEVICE)
                next_state[i, 28:45] = torch.randn(17).to(DEVICE)
                next_state[i, 61:] = torch.randn(32).to(DEVICE)
            
            # Critic update
            with torch.no_grad():
                noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
                next_action = (actor_target(next_state) + noise).clamp(-max_action, max_action)
                
                target_q1 = critic_1_target(next_state, next_action)
                target_q2 = critic_2_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * 0.99 * target_q
            
            current_q1 = critic_1(state, action)
            current_q2 = critic_2(state, action)
            
            critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(current_q2, target_q)
            
            critic_1_optimizer.zero_grad()
            critic_2_optimizer.zero_grad()
            critic_loss.backward()
            critic_1_optimizer.step()
            critic_2_optimizer.step()
            
            # Actor update (every other step)
            if step % 2 == 0:
                pi = actor(state)
                q = critic_1(state, pi)
                lmbda = 2.5 / q.abs().mean().detach()
                
                actor_loss = -lmbda * q.mean() + nn.functional.mse_loss(pi, action)
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                print(f"  Step {step}: critic_loss={critic_loss.item():.4f}, actor_loss={actor_loss.item():.4f}")
            else:
                print(f"  Step {step}: critic_loss={critic_loss.item():.4f}")
        
        print(f"✓ TD3+BC training loop simulation successful")
        return True
        
    except Exception as e:
        print(f"✗ Training loop simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    print("\n" + "="*60)
    print("COMPOSITIONAL NETWORK COMPATIBILITY TEST SUITE")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    tests = [
        ("Basic Instantiation", test_basic_instantiation),
        ("Single Sample Forward", test_forward_pass_single),
        ("Batch Forward", test_forward_pass_batch),
        ("Q-Function Forward", test_q_function_forward),
        ("TD3+BC Integration", test_td3bc_integration),
        ("Gradient Flow", test_gradient_flow),
        ("Mixed Task Batch Stress Test", test_mixed_task_batch),
        ("TD3+BC Training Loop Simulation", test_training_loop_simulation),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! Networks are ready for TD3+BC training.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())


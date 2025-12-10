#!/usr/bin/env python3
"""
Transformer Neural Networks for Multitask RL.

This module implements transformer-based policy networks using semantic tokens
and Adaptive LayerNorm (AdaLN) conditioning, similar to DiT architecture.

Key idea: Represent state as 4 semantic tokens (robot, object, obstacle, goal),
with task ID as conditioning vector that modulates every Transformer block via AdaLN.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Sequence, Optional
import math

# Device configuration
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed(embed_dim, length):
    """Create 1D sinusoidal position embeddings."""
    pos = np.arange(length, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('L,d->Ld', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# ============================================================================
# TRANSFORMER BLOCKS
# ============================================================================

class ConditionEmbedder(nn.Module):
    """Embeds task ID conditioning vector into hidden representation."""
    def __init__(self, cond_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, cond):
        return self.mlp(cond)


class TransformerBlock(nn.Module):
    """
    Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Similar to DiTBlock but for policy networks.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x: (batch, seq_len, hidden_size)
        # c: (batch, hidden_size) - conditioning vector
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention with modulation
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP with modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Final output layer with AdaLN modulation."""
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x: (batch, seq_len, hidden_size) - we'll pool over sequence
        # c: (batch, hidden_size) - conditioning vector
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # Apply modulation to each token, then pool
        # Note: modulate expects (batch, hidden_size) for shift/scale and does unsqueeze internally
        x = modulate(self.norm_final(x), shift, scale)
        # Pool over sequence dimension (mean pooling)
        x = x.mean(dim=1)  # (batch, hidden_size)
        x = self.linear(x)  # (batch, out_dim)
        return x


# ============================================================================
# TRANSFORMER ENCODERS
# ============================================================================

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for state inputs (actor).
    
    Uses semantic tokens: [object, obstacle, goal, robot]
    Task ID is used as conditioning (not as token).
    """
    
    def __init__(
        self,
        observation_shape: Sequence[int],
        output_dim: int,
        hidden_size: int = 72,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 1.20,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self._observation_shape = observation_shape
        self._output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_patches = 4  # object, obstacle, goal, robot
        
        # Component dimensions from CompoSuite observation structure
        # State components (without task IDs):
        # object-state: 0-13 (14 dim)
        # obstacle-state: 14-27 (14 dim)
        # goal-state: 28-44 (17 dim)
        # robot-proprio: 61-92 (32 dim)
        # Task IDs: 45-60 (16 dim one-hot: object_id(4) + robot_id(4) + obstacle_id(4) + subtask_id(4))
        
        self.object_dim = 14
        self.obstacle_dim = 14
        self.goal_dim = 17
        self.robot_dim = 32
        self.cond_dim = 16  # task ID one-hot
        
        # Task-conditioned encoders (4 variants each for specialization)
        self.num_object_types = 4
        self.num_obstacle_types = 4
        self.num_goal_types = 4  # based on subtask
        self.num_robot_types = 4
        
        # Object encoders (task-conditioned)
        self.object_encoders = nn.ModuleList([
            nn.Linear(self.object_dim, hidden_size, bias=True) 
            for _ in range(self.num_object_types)
        ])
        
        # Obstacle encoders (task-conditioned)
        self.obstacle_encoders = nn.ModuleList([
            nn.Linear(self.obstacle_dim, hidden_size, bias=True) 
            for _ in range(self.num_obstacle_types)
        ])
        
        # Goal encoders (task-conditioned, based on subtask)
        self.goal_encoders = nn.ModuleList([
            nn.Linear(self.goal_dim, hidden_size, bias=True) 
            for _ in range(self.num_goal_types)
        ])
        
        # Robot encoders (task-conditioned)
        self.robot_encoders = nn.ModuleList([
            nn.Linear(self.robot_dim, hidden_size, bias=True) 
            for _ in range(self.num_robot_types)
        ])
        
        # Condition embedder (embeds task ID)
        self.cond_embedder = ConditionEmbedder(self.cond_dim, hidden_size)
        
        # Positional embedding (for 4 tokens)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), 
            requires_grad=False
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Final output layer
        self.final_layer = FinalLayer(hidden_size, output_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize network weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize task-conditioned encoders
        for encoder in self.object_encoders:
            nn.init.xavier_uniform_(encoder.weight)
            nn.init.constant_(encoder.bias, 0)
        for encoder in self.obstacle_encoders:
            nn.init.xavier_uniform_(encoder.weight)
            nn.init.constant_(encoder.bias, 0)
        for encoder in self.goal_encoders:
            nn.init.xavier_uniform_(encoder.weight)
            nn.init.constant_(encoder.bias, 0)
        for encoder in self.robot_encoders:
            nn.init.xavier_uniform_(encoder.weight)
            nn.init.constant_(encoder.bias, 0)
        
        # Initialize condition embedder
        nn.init.normal_(self.cond_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.cond_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def extract_components(self, state):
        """
        Extract semantic components and task ID from state.
        
        Args:
            state: (batch, 93) - full observation with task IDs
        
        Returns:
            components: dict with 'object', 'obstacle', 'goal', 'robot'
            cond: (batch, 16) - task ID one-hot
        """
        # Extract state components (without task IDs)
        object_state = state[:, 0:14]  # (batch, 14)
        obstacle_state = state[:, 14:28]  # (batch, 14)
        goal_state = state[:, 28:45]  # (batch, 17)
        robot_state = state[:, 61:93]  # (batch, 32)
        
        # Extract task ID one-hot (16-dim)
        cond = state[:, 45:61]  # (batch, 16)
        # Format: [object_id(4), robot_id(4), obstacle_id(4), subtask_id(4)]
        
        return {
            'object': object_state,
            'obstacle': obstacle_state,
            'goal': goal_state,
            'robot': robot_state
        }, cond
    
    def encode_components(self, components, cond):
        """
        Encode semantic components using task-conditioned encoders.
        
        Args:
            components: dict with 'object', 'obstacle', 'goal', 'robot'
            cond: (batch, 16) - task ID one-hot
        
        Returns:
            tokens: (batch, 4, hidden_size) - encoded tokens
        """
        batch_size = components['object'].shape[0]
        device = components['object'].device
        
        # Extract onehot indicators
        object_id_onehot = cond[:, :4]  # (batch, 4)
        robot_id_onehot = cond[:, 4:8]  # (batch, 4)
        obstacle_id_onehot = cond[:, 8:12]  # (batch, 4)
        subtask_id_onehot = cond[:, 12:16]  # (batch, 4)
        
        # Get encoder indices
        object_idx = torch.argmax(object_id_onehot, dim=1)  # (batch,)
        robot_idx = torch.argmax(robot_id_onehot, dim=1)
        obstacle_idx = torch.argmax(obstacle_id_onehot, dim=1)
        subtask_idx = torch.argmax(subtask_id_onehot, dim=1)
        
        # Efficient batched encoding for each component type
        encoded_object = torch.zeros(batch_size, self.hidden_size, device=device)
        for obj_type in range(self.num_object_types):
            mask = object_idx == obj_type
            if mask.any():
                encoder = self.object_encoders[obj_type]
                encoded_object[mask] = encoder(components['object'][mask])
        
        encoded_obstacle = torch.zeros(batch_size, self.hidden_size, device=device)
        for obst_type in range(self.num_obstacle_types):
            mask = obstacle_idx == obst_type
            if mask.any():
                encoder = self.obstacle_encoders[obst_type]
                encoded_obstacle[mask] = encoder(components['obstacle'][mask])
        
        encoded_goal = torch.zeros(batch_size, self.hidden_size, device=device)
        for goal_type in range(self.num_goal_types):
            mask = subtask_idx == goal_type
            if mask.any():
                encoder = self.goal_encoders[goal_type]
                encoded_goal[mask] = encoder(components['goal'][mask])
        
        encoded_robot = torch.zeros(batch_size, self.hidden_size, device=device)
        for robot_type in range(self.num_robot_types):
            mask = robot_idx == robot_type
            if mask.any():
                encoder = self.robot_encoders[robot_type]
                encoded_robot[mask] = encoder(components['robot'][mask])
        
        # Stack into tokens: [object, obstacle, goal, robot]
        tokens = torch.stack([
            encoded_object,
            encoded_obstacle,
            encoded_goal,
            encoded_robot
        ], dim=1)  # (batch, 4, hidden_size)
        
        return tokens
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (batch, 93) - full observation with task IDs
        
        Returns:
            output: (batch, output_dim) - action for actor
        """
        # Extract components and task ID
        components, cond = self.extract_components(state)
        
        # Encode components to tokens
        tokens = self.encode_components(components, cond)  # (batch, 4, hidden_size)
        
        # Add positional embedding
        x = tokens + self.pos_embed  # (batch, 4, hidden_size)
        
        # Embed conditioning
        c = self.cond_embedder(cond)  # (batch, hidden_size)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)  # (batch, 4, hidden_size)
        
        # Final output layer (pools over tokens and outputs)
        output = self.final_layer(x, c)  # (batch, output_dim)
        
        return output
    
    @property
    def feature_size(self):
        return self._output_dim


class TransformerEncoderWithAction(nn.Module):
    """
    Transformer encoder for state + action inputs (critic/Q-function).
    
    Uses semantic tokens: [object, obstacle, goal, robot, action]
    Task ID is used as conditioning (not as token).
    """
    
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        output_dim: int = 1,
        hidden_size: int = 72,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 1.20,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self._observation_shape = observation_shape
        self._action_size = action_size
        self._output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_patches = 5  # object, obstacle, goal, robot, action
        
        # Component dimensions (same as TransformerEncoder)
        self.object_dim = 14
        self.obstacle_dim = 14
        self.goal_dim = 17
        self.robot_dim = 32
        self.action_dim = action_size
        self.cond_dim = 16  # task ID one-hot
        
        # Task-conditioned encoders (same as TransformerEncoder)
        self.num_object_types = 4
        self.num_obstacle_types = 4
        self.num_goal_types = 4
        self.num_robot_types = 4
        
        self.object_encoders = nn.ModuleList([
            nn.Linear(self.object_dim, hidden_size, bias=True) 
            for _ in range(self.num_object_types)
        ])
        self.obstacle_encoders = nn.ModuleList([
            nn.Linear(self.obstacle_dim, hidden_size, bias=True) 
            for _ in range(self.num_obstacle_types)
        ])
        self.goal_encoders = nn.ModuleList([
            nn.Linear(self.goal_dim, hidden_size, bias=True) 
            for _ in range(self.num_goal_types)
        ])
        self.robot_encoders = nn.ModuleList([
            nn.Linear(self.robot_dim, hidden_size, bias=True) 
            for _ in range(self.num_robot_types)
        ])
        
        # Action encoder (not task-conditioned)
        self.action_encoder = nn.Linear(self.action_dim, hidden_size, bias=True)
        
        # Condition embedder
        self.cond_embedder = ConditionEmbedder(self.cond_dim, hidden_size)
        
        # Positional embedding (for 5 tokens)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), 
            requires_grad=False
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Final output layer
        self.final_layer = FinalLayer(hidden_size, output_dim)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize network weights."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize encoders
        for encoder in self.object_encoders:
            nn.init.xavier_uniform_(encoder.weight)
            nn.init.constant_(encoder.bias, 0)
        for encoder in self.obstacle_encoders:
            nn.init.xavier_uniform_(encoder.weight)
            nn.init.constant_(encoder.bias, 0)
        for encoder in self.goal_encoders:
            nn.init.xavier_uniform_(encoder.weight)
            nn.init.constant_(encoder.bias, 0)
        for encoder in self.robot_encoders:
            nn.init.xavier_uniform_(encoder.weight)
            nn.init.constant_(encoder.bias, 0)
        nn.init.xavier_uniform_(self.action_encoder.weight)
        nn.init.constant_(self.action_encoder.bias, 0)
        
        # Initialize condition embedder
        nn.init.normal_(self.cond_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.cond_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def extract_components(self, state):
        """Extract semantic components and task ID from state."""
        object_state = state[:, 0:14]
        obstacle_state = state[:, 14:28]
        goal_state = state[:, 28:45]
        robot_state = state[:, 61:93]
        cond = state[:, 45:61]  # task ID
        
        return {
            'object': object_state,
            'obstacle': obstacle_state,
            'goal': goal_state,
            'robot': robot_state
        }, cond
    
    def encode_components(self, components, cond, action):
        """
        Encode semantic components + action using task-conditioned encoders.
        
        Args:
            components: dict with 'object', 'obstacle', 'goal', 'robot'
            cond: (batch, 16) - task ID one-hot
            action: (batch, action_size)
        
        Returns:
            tokens: (batch, 5, hidden_size) - encoded tokens
        """
        batch_size = components['object'].shape[0]
        device = components['object'].device
        
        # Extract onehot indicators
        object_id_onehot = cond[:, :4]
        robot_id_onehot = cond[:, 4:8]
        obstacle_id_onehot = cond[:, 8:12]
        subtask_id_onehot = cond[:, 12:16]
        
        # Get encoder indices
        object_idx = torch.argmax(object_id_onehot, dim=1)
        robot_idx = torch.argmax(robot_id_onehot, dim=1)
        obstacle_idx = torch.argmax(obstacle_id_onehot, dim=1)
        subtask_idx = torch.argmax(subtask_id_onehot, dim=1)
        
        # Encode state components (same as TransformerEncoder)
        encoded_object = torch.zeros(batch_size, self.hidden_size, device=device)
        for obj_type in range(self.num_object_types):
            mask = object_idx == obj_type
            if mask.any():
                encoder = self.object_encoders[obj_type]
                encoded_object[mask] = encoder(components['object'][mask])
        
        encoded_obstacle = torch.zeros(batch_size, self.hidden_size, device=device)
        for obst_type in range(self.num_obstacle_types):
            mask = obstacle_idx == obst_type
            if mask.any():
                encoder = self.obstacle_encoders[obst_type]
                encoded_obstacle[mask] = encoder(components['obstacle'][mask])
        
        encoded_goal = torch.zeros(batch_size, self.hidden_size, device=device)
        for goal_type in range(self.num_goal_types):
            mask = subtask_idx == goal_type
            if mask.any():
                encoder = self.goal_encoders[goal_type]
                encoded_goal[mask] = encoder(components['goal'][mask])
        
        encoded_robot = torch.zeros(batch_size, self.hidden_size, device=device)
        for robot_type in range(self.num_robot_types):
            mask = robot_idx == robot_type
            if mask.any():
                encoder = self.robot_encoders[robot_type]
                encoded_robot[mask] = encoder(components['robot'][mask])
        
        # Encode action (not task-conditioned)
        encoded_action = self.action_encoder(action)  # (batch, hidden_size)
        
        # Stack into tokens: [object, obstacle, goal, robot, action]
        tokens = torch.stack([
            encoded_object,
            encoded_obstacle,
            encoded_goal,
            encoded_robot,
            encoded_action
        ], dim=1)  # (batch, 5, hidden_size)
        
        return tokens
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (batch, 93) - full observation with task IDs
            action: (batch, action_size)
        
        Returns:
            output: (batch, output_dim) - Q-value for critic
        """
        # Extract components and task ID
        components, cond = self.extract_components(state)
        
        # Encode components + action to tokens
        tokens = self.encode_components(components, cond, action)  # (batch, 5, hidden_size)
        
        # Add positional embedding
        x = tokens + self.pos_embed  # (batch, 5, hidden_size)
        
        # Embed conditioning
        c = self.cond_embedder(cond)  # (batch, hidden_size)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)  # (batch, 5, hidden_size)
        
        # Final output layer (pools over tokens and outputs)
        output = self.final_layer(x, c)  # (batch, output_dim)
        
        return output
    
    @property
    def feature_size(self):
        return self._output_dim


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_transformer_encoderfactory_params(
    with_action=False, 
    output_dim=None,
    hidden_size=72,
    depth=1,
    num_heads=4,
    mlp_ratio=1.20,
    dropout=0.0,
):
    """
    Create encoder factory parameters for transformer networks.
    
    Returns dict with architecture parameters (for compatibility with training script).
    """
    act_dim = 8
    
    return {
        "hidden_size": hidden_size,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "dropout": dropout,
        "output_dim": output_dim if output_dim is not None else act_dim,
        "with_action": with_action,
    }

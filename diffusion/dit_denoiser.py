"""
Diffusion Transformer (DiT) adapted for 1D sequential/tabular data.
Based on the DiT architecture from https://github.com/facebookresearch/DiT
Adapted for RL transition data instead of images.
"""

import math
from typing import Optional

import gin
import torch
import torch.nn as nn
import numpy as np


def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionEmbedder(nn.Module):
    """
    Embeds continuous condition vectors (like task indicators) into vector representations.
    """
    def __init__(self, cond_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, cond):
        return self.mlp(cond)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Adapted for 1D sequential data.
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
        # c: (batch, hidden_size)
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
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Create 1D sinusoidal position embeddings.
    
    Args:
        embed_dim: dimension of the embedding
        length: number of positions
    
    Returns:
        pos_embed: [length, embed_dim]
    """
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


@gin.configurable
class DiT1D(nn.Module):
    """
    Diffusion Transformer for 1D sequential/tabular data.
    
    Args:
        d_in: input dimension
        hidden_size: hidden dimension of transformer (dimension of each patch/token)
        depth: number of transformer blocks
        num_heads: number of attention heads (hidden_size must be divisible by num_heads)
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        patch_size: uniform patch size for STANDARD patching only (IGNORED for semantic patching)
        cond_dim: dimension of conditioning vector (e.g., task indicators)
        dropout: dropout rate
        learned_sinusoidal_cond: compatibility flag (always True for this architecture)
        random_fourier_features: compatibility flag (always True for this architecture)
        use_semantic_patching: if True, use component-aware patching (patch_size ignored)
                               if False, use uniform patching with patch_size
    
    Note:
        - Standard patching: splits d_in into uniform patches of size patch_size
        - Semantic patching: splits into 11 semantic components, each encoded to hidden_size
                            (patch_size parameter is not used)
    """
    def __init__(
        self,
        d_in: int,
        hidden_size: int = 416,  # Default: 26.09M params with semantic patching
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 4,  # Only used for standard patching
        cond_dim: Optional[int] = None,
        dropout: float = 0.0,
        learned_sinusoidal_cond: bool = True,
        random_fourier_features: bool = True,
        learned_sinusoidal_dim: int = 16,
        use_semantic_patching: bool = True,  # Default: use semantic component-aware patching
    ):
        super().__init__()
        
        # Compatibility attributes for ElucidatedDiffusion
        self.random_or_learned_sinusoidal_cond = True
        self.conditional = cond_dim is not None
        
        self.d_in = d_in
        self.hidden_size = hidden_size
        self.patch_size = patch_size  # Only used for standard patching
        self.num_heads = num_heads
        self.use_semantic_patching = use_semantic_patching
        
        if use_semantic_patching:
            # Semantic patching: patch_size parameter is IGNORED
            # Each component becomes a patch, encoded directly to hidden_size
            # Semantic component-aware patching for RL transitions
            # Input: [cs_obj(14), cs_obst(14), cs_goal(17), cs_prop(32), 
            #         action(8), reward(1), 
            #         ns_obj(14), ns_obst(14), ns_goal(17), ns_prop(32), terminal(1)]
            # Total: 164 dims → 11 patches at hidden_size
            # Only down-project at the very end (decoders)
            
            self.component_dims = [14, 14, 17, 32, 8, 1, 14, 14, 17, 32, 1]
            self.num_patches = len(self.component_dims)
            
            # Task-conditioned encoders for state components
            # 4 variants each for: object, obstacle, goal, robot
            self.num_object_types = 4
            self.num_obstacle_types = 4
            self.num_goal_types = 4  # based on subtask
            self.num_robot_types = 4
            
            self.object_encoders = nn.ModuleList([
                nn.Linear(14, hidden_size, bias=True) for _ in range(self.num_object_types)
            ])
            self.obstacle_encoders = nn.ModuleList([
                nn.Linear(14, hidden_size, bias=True) for _ in range(self.num_obstacle_types)
            ])
            self.goal_encoders = nn.ModuleList([
                nn.Linear(17, hidden_size, bias=True) for _ in range(self.num_goal_types)
            ])
            self.robot_encoders = nn.ModuleList([
                nn.Linear(32, hidden_size, bias=True) for _ in range(self.num_robot_types)
            ])
            
            # Simple encoders for non-state components
            self.action_encoder = nn.Linear(8, hidden_size, bias=True)
            self.reward_encoder = nn.Linear(1, hidden_size, bias=True)
            self.terminal_encoder = nn.Linear(1, hidden_size, bias=True)
            
            # Task-conditioned decoders for state components (matching encoder pattern)
            # 4 variants each for: object, obstacle, goal, robot
            self.object_decoders = nn.ModuleList([
                nn.Linear(hidden_size, 14, bias=True) for _ in range(self.num_object_types)
            ])
            self.obstacle_decoders = nn.ModuleList([
                nn.Linear(hidden_size, 14, bias=True) for _ in range(self.num_obstacle_types)
            ])
            self.goal_decoders = nn.ModuleList([
                nn.Linear(hidden_size, 17, bias=True) for _ in range(self.num_goal_types)
            ])
            self.robot_decoders = nn.ModuleList([
                nn.Linear(hidden_size, 32, bias=True) for _ in range(self.num_robot_types)
            ])
            
            # Simple decoders for non-state components
            self.action_decoder = nn.Linear(hidden_size, 8, bias=True)
            self.reward_decoder = nn.Linear(hidden_size, 1, bias=True)
            self.terminal_decoder = nn.Linear(hidden_size, 1, bias=True)
            
            # No need for x_embedder - we go directly to hidden_size
            self.x_embedder = None
            
        else:
            # Standard uniform patching
            # Handle non-divisible case by padding
            if d_in % patch_size != 0:
                self.padding_size = patch_size - (d_in % patch_size)
                self.padded_d_in = d_in + self.padding_size
                self.num_patches = self.padded_d_in // patch_size
                print(f"Warning: d_in ({d_in}) not divisible by patch_size ({patch_size}). "
                      f"Padding to {self.padded_d_in} (adding {self.padding_size} dimensions).")
            else:
                self.padding_size = 0
                self.padded_d_in = d_in
                self.num_patches = d_in // patch_size
            
            # Input projection: map each patch to hidden_size
            self.x_embedder = nn.Linear(patch_size, hidden_size, bias=True)
        
        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Condition embedder
        if self.conditional:
            self.cond_embedder = ConditionEmbedder(cond_dim, hidden_size)
        
        # Positional embedding (learnable or fixed)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout) 
            for _ in range(depth)
        ])
        
        # Final layer
        # For semantic patching, output stays at hidden_size (no down-projection)
        # For standard patching, output to patch_size
        final_out_dim = hidden_size if use_semantic_patching else patch_size
        self.final_layer = FinalLayer(hidden_size, final_out_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize positional embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch embedding (only for standard patching)
        if self.x_embedder is not None:
            nn.init.xavier_uniform_(self.x_embedder.weight)
            nn.init.constant_(self.x_embedder.bias, 0)
        
        if self.use_semantic_patching:
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
            
            # Initialize simple encoders
            nn.init.xavier_uniform_(self.action_encoder.weight)
            nn.init.constant_(self.action_encoder.bias, 0)
            nn.init.xavier_uniform_(self.reward_encoder.weight)
            nn.init.constant_(self.reward_encoder.bias, 0)
            nn.init.xavier_uniform_(self.terminal_encoder.weight)
            nn.init.constant_(self.terminal_encoder.bias, 0)
            
            # Initialize task-conditioned decoders
            for decoder in self.object_decoders:
                nn.init.xavier_uniform_(decoder.weight)
                nn.init.constant_(decoder.bias, 0)
            for decoder in self.obstacle_decoders:
                nn.init.xavier_uniform_(decoder.weight)
                nn.init.constant_(decoder.bias, 0)
            for decoder in self.goal_decoders:
                nn.init.xavier_uniform_(decoder.weight)
                nn.init.constant_(decoder.bias, 0)
            for decoder in self.robot_decoders:
                nn.init.xavier_uniform_(decoder.weight)
                nn.init.constant_(decoder.bias, 0)
            
            # Initialize simple decoders
            nn.init.xavier_uniform_(self.action_decoder.weight)
            nn.init.constant_(self.action_decoder.bias, 0)
            nn.init.xavier_uniform_(self.reward_decoder.weight)
            nn.init.constant_(self.reward_decoder.bias, 0)
            nn.init.xavier_uniform_(self.terminal_decoder.weight)
            nn.init.constant_(self.terminal_decoder.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize condition embedding if present
        if self.conditional:
            nn.init.normal_(self.cond_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.cond_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, cond=None):
        """
        Convert patches back to original sequence.
        
        For semantic patching:
            x: (batch, num_patches, hidden_size)
            cond: (batch, 16) - onehot indicators [object_id(4), robot_id(4), obstacle_id(4), subtask_id(4)]
            output: (batch, d_in)
            
        For standard patching:
            x: (batch, num_patches, patch_size)
            output: (batch, d_in)
        """
        if self.use_semantic_patching:
            # Task-conditioned decoding (matching encoder pattern)
            # Decode each patch (at hidden_size) back to its original component dimension
            # This is the ONLY down-projection in the network
            batch_size = x.shape[0]
            device = x.device
            
            # Extract patches: [cs_obj, cs_obst, cs_goal, cs_prop, action, reward, ns_obj, ns_obst, ns_goal, ns_prop, terminal]
            cs_obj_patch = x[:, 0, :]  # (batch, hidden_size)
            cs_obst_patch = x[:, 1, :]
            cs_goal_patch = x[:, 2, :]
            cs_prop_patch = x[:, 3, :]
            action_patch = x[:, 4, :]
            reward_patch = x[:, 5, :]
            ns_obj_patch = x[:, 6, :]
            ns_obst_patch = x[:, 7, :]
            ns_goal_patch = x[:, 8, :]
            ns_prop_patch = x[:, 9, :]
            terminal_patch = x[:, 10, :]
            
            # Task-conditioned decoding requires conditioning
            assert cond is not None, "Conditioning is required for semantic patching with task-conditioned decoders"
            
            # Extract onehot indicators
            object_id_onehot = cond[:, :4]
            robot_id_onehot = cond[:, 4:8]
            obstacle_id_onehot = cond[:, 8:12]
            subtask_id_onehot = cond[:, 12:16]
            
            # Get decoder indices
            object_idx = torch.argmax(object_id_onehot, dim=1)
            robot_idx = torch.argmax(robot_id_onehot, dim=1)
            obstacle_idx = torch.argmax(obstacle_id_onehot, dim=1)
            subtask_idx = torch.argmax(subtask_id_onehot, dim=1)
            
            # Efficient batched decoding for object components (cs_obj, ns_obj)
            decoded_cs_obj = torch.zeros(batch_size, 14, device=device)
            decoded_ns_obj = torch.zeros(batch_size, 14, device=device)
            for obj_type in range(self.num_object_types):
                mask = object_idx == obj_type
                if mask.any():
                    decoder = self.object_decoders[obj_type]
                    # Stack current and next state patches
                    stacked_obj_patches = torch.stack([cs_obj_patch[mask], ns_obj_patch[mask]], dim=1)  # [n, 2, hidden_size]
                    stacked_obj_patches_flat = stacked_obj_patches.view(-1, self.hidden_size)  # [n*2, hidden_size]
                    decoded_stacked_flat = decoder(stacked_obj_patches_flat)  # [n*2, 14]
                    decoded_stacked = decoded_stacked_flat.view(-1, 2, 14)  # [n, 2, 14]
                    decoded_cs_obj[mask] = decoded_stacked[:, 0]
                    decoded_ns_obj[mask] = decoded_stacked[:, 1]
            
            # Efficient batched decoding for obstacle components (cs_obst, ns_obst)
            decoded_cs_obst = torch.zeros(batch_size, 14, device=device)
            decoded_ns_obst = torch.zeros(batch_size, 14, device=device)
            for obst_type in range(self.num_obstacle_types):
                mask = obstacle_idx == obst_type
                if mask.any():
                    decoder = self.obstacle_decoders[obst_type]
                    stacked_obst_patches = torch.stack([cs_obst_patch[mask], ns_obst_patch[mask]], dim=1)
                    stacked_obst_patches_flat = stacked_obst_patches.view(-1, self.hidden_size)
                    decoded_stacked_flat = decoder(stacked_obst_patches_flat)
                    decoded_stacked = decoded_stacked_flat.view(-1, 2, 14)
                    decoded_cs_obst[mask] = decoded_stacked[:, 0]
                    decoded_ns_obst[mask] = decoded_stacked[:, 1]
            
            # Efficient batched decoding for goal components (cs_goal, ns_goal)
            decoded_cs_goal = torch.zeros(batch_size, 17, device=device)
            decoded_ns_goal = torch.zeros(batch_size, 17, device=device)
            for goal_type in range(self.num_goal_types):
                mask = subtask_idx == goal_type
                if mask.any():
                    decoder = self.goal_decoders[goal_type]
                    stacked_goal_patches = torch.stack([cs_goal_patch[mask], ns_goal_patch[mask]], dim=1)
                    stacked_goal_patches_flat = stacked_goal_patches.view(-1, self.hidden_size)
                    decoded_stacked_flat = decoder(stacked_goal_patches_flat)
                    decoded_stacked = decoded_stacked_flat.view(-1, 2, 17)
                    decoded_cs_goal[mask] = decoded_stacked[:, 0]
                    decoded_ns_goal[mask] = decoded_stacked[:, 1]
            
            # Efficient batched decoding for robot components (cs_prop, ns_prop)
            decoded_cs_prop = torch.zeros(batch_size, 32, device=device)
            decoded_ns_prop = torch.zeros(batch_size, 32, device=device)
            for robot_type in range(self.num_robot_types):
                mask = robot_idx == robot_type
                if mask.any():
                    decoder = self.robot_decoders[robot_type]
                    stacked_prop_patches = torch.stack([cs_prop_patch[mask], ns_prop_patch[mask]], dim=1)
                    stacked_prop_patches_flat = stacked_prop_patches.view(-1, self.hidden_size)
                    decoded_stacked_flat = decoder(stacked_prop_patches_flat)
                    decoded_stacked = decoded_stacked_flat.view(-1, 2, 32)
                    decoded_cs_prop[mask] = decoded_stacked[:, 0]
                    decoded_ns_prop[mask] = decoded_stacked[:, 1]
            
            # Decode simple components (action, reward, terminal)
            decoded_action = self.action_decoder(action_patch)  # (batch, 8)
            decoded_reward = self.reward_decoder(reward_patch)  # (batch, 1)
            decoded_terminal = self.terminal_decoder(terminal_patch)  # (batch, 1)
            
            # Concatenate all components in order: [cs_obj, cs_obst, cs_goal, cs_prop, action, reward, ns_obj, ns_obst, ns_goal, ns_prop, terminal]
            return torch.cat([
                decoded_cs_obj,      # (batch, 14)
                decoded_cs_obst,     # (batch, 14)
                decoded_cs_goal,     # (batch, 17)
                decoded_cs_prop,     # (batch, 32)
                decoded_action,      # (batch, 8)
                decoded_reward,      # (batch, 1)
                decoded_ns_obj,      # (batch, 14)
                decoded_ns_obst,     # (batch, 14)
                decoded_ns_goal,     # (batch, 17)
                decoded_ns_prop,     # (batch, 32)
                decoded_terminal,     # (batch, 1)
            ], dim=1)  # (batch, 164)
        else:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
            # Remove padding if necessary
            if self.padding_size > 0:
                x = x[:, :-self.padding_size]
            return x

    def patchify(self, x, cond=None):
        """
        Convert sequence to patches.
        
        For semantic patching:
            x: (batch, d_in)
            cond: (batch, 16) - onehot indicators [object_id(4), robot_id(4), obstacle_id(4), subtask_id(4)]
            output: (batch, num_patches, hidden_size)
            
        For standard patching:
            x: (batch, d_in)
            output: (batch, num_patches, patch_size)
        """
        if self.use_semantic_patching:
            # Split input into semantic components
            # [cs_obj(14), cs_obst(14), cs_goal(17), cs_prop(32), 
            #  action(8), reward(1), 
            #  ns_obj(14), ns_obst(14), ns_goal(17), ns_prop(32), terminal(1)]
            batch_size = x.shape[0]
            device = x.device
            
            cs_obj = x[:, 0:14]
            cs_obst = x[:, 14:28]
            cs_goal = x[:, 28:45]
            cs_prop = x[:, 45:77]
            action = x[:, 77:85]
            reward = x[:, 85:86]
            ns_obj = x[:, 86:100]
            ns_obst = x[:, 100:114]
            ns_goal = x[:, 114:131]
            ns_prop = x[:, 131:163]
            terminal = x[:, 163:164]
            
            # Initialize output patches
            patches = []
            
            # Task-conditioned encoders for state components
            assert cond is not None, "Conditioning is required for semantic patching with task-conditioned encoders"
            
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
            
            # Efficient batched encoding for object components (cs_obj, ns_obj)
            encoded_cs_obj = torch.zeros(batch_size, self.hidden_size, device=device)
            encoded_ns_obj = torch.zeros(batch_size, self.hidden_size, device=device)
            for obj_type in range(self.num_object_types):
                mask = object_idx == obj_type
                if mask.any():
                    encoder = self.object_encoders[obj_type]
                    # Stack current and next states
                    stacked_obj = torch.stack([cs_obj[mask], ns_obj[mask]], dim=1)  # [n, 2, 14]
                    stacked_obj_flat = stacked_obj.view(-1, 14)  # [n*2, 14]
                    encoded_stacked_flat = encoder(stacked_obj_flat)  # [n*2, hidden_size]
                    encoded_stacked = encoded_stacked_flat.view(-1, 2, self.hidden_size)  # [n, 2, hidden_size]
                    encoded_cs_obj[mask] = encoded_stacked[:, 0]
                    encoded_ns_obj[mask] = encoded_stacked[:, 1]
            
            # Efficient batched encoding for obstacle components (cs_obst, ns_obst)
            encoded_cs_obst = torch.zeros(batch_size, self.hidden_size, device=device)
            encoded_ns_obst = torch.zeros(batch_size, self.hidden_size, device=device)
            for obst_type in range(self.num_obstacle_types):
                mask = obstacle_idx == obst_type
                if mask.any():
                    encoder = self.obstacle_encoders[obst_type]
                    stacked_obst = torch.stack([cs_obst[mask], ns_obst[mask]], dim=1)
                    stacked_obst_flat = stacked_obst.view(-1, 14)
                    encoded_stacked_flat = encoder(stacked_obst_flat)
                    encoded_stacked = encoded_stacked_flat.view(-1, 2, self.hidden_size)
                    encoded_cs_obst[mask] = encoded_stacked[:, 0]
                    encoded_ns_obst[mask] = encoded_stacked[:, 1]
            
            # Efficient batched encoding for goal components (cs_goal, ns_goal)
            encoded_cs_goal = torch.zeros(batch_size, self.hidden_size, device=device)
            encoded_ns_goal = torch.zeros(batch_size, self.hidden_size, device=device)
            for goal_type in range(self.num_goal_types):
                mask = subtask_idx == goal_type
                if mask.any():
                    encoder = self.goal_encoders[goal_type]
                    stacked_goal = torch.stack([cs_goal[mask], ns_goal[mask]], dim=1)
                    stacked_goal_flat = stacked_goal.view(-1, 17)
                    encoded_stacked_flat = encoder(stacked_goal_flat)
                    encoded_stacked = encoded_stacked_flat.view(-1, 2, self.hidden_size)
                    encoded_cs_goal[mask] = encoded_stacked[:, 0]
                    encoded_ns_goal[mask] = encoded_stacked[:, 1]
            
            # Efficient batched encoding for robot components (cs_prop, ns_prop)
            encoded_cs_prop = torch.zeros(batch_size, self.hidden_size, device=device)
            encoded_ns_prop = torch.zeros(batch_size, self.hidden_size, device=device)
            for robot_type in range(self.num_robot_types):
                mask = robot_idx == robot_type
                if mask.any():
                    encoder = self.robot_encoders[robot_type]
                    stacked_prop = torch.stack([cs_prop[mask], ns_prop[mask]], dim=1)
                    stacked_prop_flat = stacked_prop.view(-1, 32)
                    encoded_stacked_flat = encoder(stacked_prop_flat)
                    encoded_stacked = encoded_stacked_flat.view(-1, 2, self.hidden_size)
                    encoded_cs_prop[mask] = encoded_stacked[:, 0]
                    encoded_ns_prop[mask] = encoded_stacked[:, 1]
            
            # Build patches list: [cs_obj, cs_obst, cs_goal, cs_prop, action, reward, ns_obj, ns_obst, ns_goal, ns_prop, terminal]
            patches = [
                encoded_cs_obj,
                encoded_cs_obst,
                encoded_cs_goal,
                encoded_cs_prop,
                self.action_encoder(action),
                self.reward_encoder(reward),
                encoded_ns_obj,
                encoded_ns_obst,
                encoded_ns_goal,
                encoded_ns_prop,
                self.terminal_encoder(terminal),
            ]
            
            # Stack into patches: (batch, num_patches, hidden_size)
            return torch.stack(patches, dim=1)
        else:
            batch_size = x.shape[0]
            # Pad if necessary
            if self.padding_size > 0:
                padding = torch.zeros(batch_size, self.padding_size, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
            return x.reshape(batch_size, self.num_patches, self.patch_size)

    def forward(self, x, timesteps, cond=None):
        """
        Forward pass of DiT1D.
        
        Args:
            x: (batch, d_in) tensor of input data
            timesteps: (batch,) tensor of diffusion timesteps
            cond: (batch, cond_dim) tensor of conditioning (optional)
        
        Returns:
            output: (batch, d_in) tensor of denoised output
        """
        # Patchify input
        # For semantic patching: (batch, num_patches, hidden_size)
        # For standard patching: (batch, num_patches, patch_size)
        x_patches = self.patchify(x, cond=cond)
        
        # Embed patches to hidden_size (only needed for standard patching)
        if self.use_semantic_patching:
            # Already at hidden_size, just add positional embedding
            x = x_patches + self.pos_embed  # (batch, num_patches, hidden_size)
        else:
            # Need to project patch_size to hidden_size
            x = self.x_embedder(x_patches) + self.pos_embed  # (batch, num_patches, hidden_size)
        
        # Embed timesteps
        t = self.t_embedder(timesteps)  # (batch, hidden_size)
        
        # Embed and combine conditioning
        if self.conditional:
            assert cond is not None, "Conditioning is required for conditional model"
            
            # Print raw conditioning input (first sample in batch)
            # print("\n" + "=" * 80)
            # print("CONDITIONAL VECTORS (Forward Pass)")
            # print("=" * 80)
            # print(f"Batch size: {cond.shape[0]}")
            # print(f"\n1. Raw conditioning input (cond) - shape: {cond.shape}")
            # print(f"   First sample: {cond[0]}")
            # print(f"   One-hot positions: {torch.where(cond[0] == 1.0)[0].tolist() if cond[0].max() <= 1.0 else 'Not one-hot'}")
            
            # Embed conditioning
            y = self.cond_embedder(cond)  # (batch, hidden_size)
            # print(f"\n2. Embedded conditioning (y) - shape: {y.shape}")
            # print(f"   First sample (first 10 dims): {y[0, :10]}")
            # print(f"   Mean: {y[0].mean().item():.4f}, Std: {y[0].std().item():.4f}")
            # print(f"   Min: {y[0].min().item():.4f}, Max: {y[0].max().item():.4f}")
            
            # Combine with timestep
            c = t + y  # (batch, hidden_size)
            # print(f"\n3. Combined (c = timestep + cond) - shape: {c.shape}")
            # print(f"   First sample (first 10 dims): {c[0, :10]}")
            # print(f"   Mean: {c[0].mean().item():.4f}, Std: {c[0].std().item():.4f}")
            # print("=" * 80 + "\n")
        else:
            c = t
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)  # (batch, num_patches, hidden_size)
        
        # Final layer
        # For semantic patching: outputs (batch, num_patches, hidden_size)
        # For standard patching: outputs (batch, num_patches, patch_size)
        x = self.final_layer(x, c)
        
        # Unpatchify
        # For semantic patching: decodes from hidden_size to original dims (ONLY down-projection)
        # For standard patching: reshapes patches back to sequence
        x = self.unpatchify(x, cond=cond)  # (batch, d_in)
        
        return x


# Predefined DiT configurations
def DiT1D_XL(**kwargs):
    """Extra Large DiT model"""
    return DiT1D(depth=24, hidden_size=1024, num_heads=16, **kwargs)

def DiT1D_L(**kwargs):
    """Large DiT model"""
    return DiT1D(depth=16, hidden_size=768, num_heads=12, **kwargs)

def DiT1D_B(**kwargs):
    """Base DiT model"""
    return DiT1D(depth=12, hidden_size=512, num_heads=8, **kwargs)

def DiT1D_S(**kwargs):
    """Small DiT model"""
    return DiT1D(depth=8, hidden_size=384, num_heads=6, **kwargs)

def DiT1D_XS(**kwargs):
    """Extra Small DiT model"""
    return DiT1D(depth=6, hidden_size=256, num_heads=4, **kwargs)


DiT1D_models = {
    'DiT1D-XL': DiT1D_XL,
    'DiT1D-L': DiT1D_L,
    'DiT1D-B': DiT1D_B,
    'DiT1D-S': DiT1D_S,
    'DiT1D-XS': DiT1D_XS,
}







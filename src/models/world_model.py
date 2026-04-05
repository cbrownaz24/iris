from dataclasses import dataclass
from typing import Any, Optional

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .tokenizer import Tokenizer
from .transformer import TransformerConfig, MiniTransformerEncoder
from .stu import STUBackbone
from utils import init_weights, LossWithIntermediateLosses


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


class STUCache:
    """
    Cache for incremental generation with the mini-transformer + STU architecture.

    Stores projected block vectors (stu_dim) so STU can be re-run over the full
    history each step.  Also holds the current observation encoding waiting to
    be paired with an action.
    """

    def __init__(self, n: int, max_blocks: int, stu_dim: int,
                 tokens_per_block: int, device: torch.device) -> None:
        self.n = n
        self.max_blocks = max_blocks
        self.tokens_per_block = tokens_per_block
        self.device = device

        # Projected block inputs for STU (filled as blocks complete)
        self.block_inputs = torch.zeros(n, max_blocks, stu_dim, device=device)
        self.num_blocks = 0

        # Obs encoding from the mini transformer, waiting for an action
        self.pending_obs_enc: Optional[torch.Tensor] = None  # (B, stu_obs_dim)

    @property
    def size(self) -> int:
        """Approximate token count consumed (for capacity checks)."""
        return self.num_blocks * self.tokens_per_block

    def reset(self) -> None:
        self.block_inputs.zero_()
        self.num_blocks = 0
        self.pending_obs_enc = None

    def prune(self, mask: np.ndarray) -> None:
        self.block_inputs = self.block_inputs[mask]
        self.n = self.block_inputs.shape[0]
        if self.pending_obs_enc is not None:
            self.pending_obs_enc = self.pending_obs_enc[mask]

    def add_block(self, block_vec: torch.Tensor) -> None:
        """Store a projected block vector (B, stu_dim)."""
        self.block_inputs[:, self.num_blocks] = block_vec
        self.num_blocks += 1


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config

        K = config.tokens_per_block - 1   # obs tokens per block (16)
        self.num_obs_tokens = K
        d_stu = config.stu_dim

        # ---- token embedding ------------------------------------------------
        self.obs_embed = nn.Embedding(obs_vocab_size, config.embed_dim)

        # ---- mini transformer encoder: (B, K, embed_dim) -> (B, stu_obs_dim) -
        self.mini_encoder = MiniTransformerEncoder(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            embed_dim=config.embed_dim,
            output_dim=config.stu_obs_dim,
            num_tokens=K,
            dropout=config.attn_pdrop,
        )

        # ---- input projection: (stu_obs_dim + num_actions) -> stu_dim -------
        self.input_proj = nn.Linear(config.stu_obs_dim + act_vocab_size, d_stu)

        # ---- block-level positional embedding -------------------------------
        self.block_pos_emb = nn.Embedding(config.max_blocks, d_stu)

        # ---- STU backbone ---------------------------------------------------
        self.stu = STUBackbone(
            d_model=d_stu,
            num_layers=config.num_stu_layers,
            num_filters=config.num_filters,
            max_seq_len=config.max_blocks,
            dropout=config.resid_pdrop,
        )

        # ---- output projection: stu_dim -> K * embed_dim --------------------
        obs_block_dim = K * config.embed_dim
        self.output_proj = nn.Sequential(
            nn.Linear(d_stu, d_stu),
            nn.GELU(),
            nn.Linear(d_stu, obs_block_dim),
        )

        # ---- per-token observation head (embed_dim -> obs_vocab_size) -------
        self.head_observations = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, obs_vocab_size),
        )

        # ---- reward / end heads (from stu_dim) ------------------------------
        self.head_rewards = nn.Sequential(
            nn.Linear(d_stu, d_stu),
            nn.ReLU(),
            nn.Linear(d_stu, 3),
        )
        self.head_ends = nn.Sequential(
            nn.Linear(d_stu, d_stu),
            nn.ReLU(),
            nn.Linear(d_stu, 2),
        )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_obs(self, obs_tokens: torch.LongTensor) -> torch.Tensor:
        """
        Encode observation tokens through the mini transformer.

        Args:
            obs_tokens: (B, K) observation token ids
        Returns:
            (B, stu_obs_dim) pooled observation vector
        """
        obs_emb = self.obs_embed(obs_tokens)        # (B, K, embed_dim)
        return self.mini_encoder(obs_emb)            # (B, stu_obs_dim)

    def _encode_obs_batched(self, obs_tokens: torch.LongTensor) -> torch.Tensor:
        """
        Encode a batch of observation sequences through the mini transformer.

        Args:
            obs_tokens: (B, L, K) observation token ids for L blocks
        Returns:
            (B, L, stu_obs_dim) pooled observation vectors
        """
        B, L, K = obs_tokens.shape
        obs_emb = self.obs_embed(obs_tokens)                      # (B, L, K, embed_dim)
        obs_emb_flat = obs_emb.reshape(B * L, K, -1)              # (B*L, K, embed_dim)
        obs_enc = self.mini_encoder(obs_emb_flat)                  # (B*L, stu_obs_dim)
        return obs_enc.view(B, L, -1)                              # (B, L, stu_obs_dim)

    def _build_block_inputs(self, obs_enc: torch.Tensor, act_tokens: torch.LongTensor) -> torch.Tensor:
        """
        Combine obs encodings with one-hot actions and project to stu_dim.

        Args:
            obs_enc:    (B, L, stu_obs_dim)
            act_tokens: (B, L) action token ids
        Returns:
            (B, L, stu_dim) projected block vectors
        """
        act_onehot = F.one_hot(act_tokens, self.act_vocab_size).float()   # (B, L, num_actions)
        block_input = torch.cat([obs_enc, act_onehot], dim=-1)            # (B, L, stu_obs_dim + num_actions)
        return self.input_proj(block_input)                                # (B, L, stu_dim)

    def _decode_obs(self, stu_out: torch.Tensor) -> torch.Tensor:
        """
        Project STU output to per-token observation logits.

        Args:
            stu_out: (B, L, stu_dim)
        Returns:
            (B, L, K, obs_vocab_size)
        """
        K = self.num_obs_tokens
        E = self.config.embed_dim
        B, L, _ = stu_out.shape
        obs_out = self.output_proj(stu_out).view(B, L, K, E)     # (B, L, K, embed_dim)
        return self.head_observations(obs_out)                     # (B, L, K, obs_vocab)

    def generate_empty_cache(self, n: int) -> STUCache:
        device = next(self.parameters()).device
        return STUCache(n, self.config.max_blocks, self.config.stu_dim,
                        self.config.tokens_per_block, device)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[STUCache] = None) -> WorldModelOutput:
        """Training forward pass. past_keys_values is ignored (use generation methods)."""
        return self._forward_training(tokens)

    def _forward_training(self, tokens: torch.LongTensor) -> WorldModelOutput:
        B, T = tokens.shape
        tpb = self.config.tokens_per_block
        L = T // tpb
        K = self.num_obs_tokens

        block_tokens = tokens.view(B, L, tpb)                             # (B, L, K+1)
        obs_tokens = block_tokens[:, :, :K]                                # (B, L, K)
        act_tokens = block_tokens[:, :, K]                                 # (B, L)

        # Mini transformer: encode each block's obs tokens
        obs_enc = self._encode_obs_batched(obs_tokens)                     # (B, L, stu_obs_dim)

        # Build block inputs and add positional embeddings
        block_vec = self._build_block_inputs(obs_enc, act_tokens)          # (B, L, stu_dim)
        block_vec = block_vec + self.block_pos_emb(
            torch.arange(L, device=tokens.device))

        # STU temporal backbone
        stu_out = self.stu(block_vec)                                      # (B, L, stu_dim)

        # Predictions
        logits_obs = self._decode_obs(stu_out)                             # (B, L, K, obs_vocab)
        logits_rew = self.head_rewards(stu_out)                            # (B, L, 3)
        logits_end = self.head_ends(stu_out)                               # (B, L, 2)

        return WorldModelOutput(stu_out, logits_obs, logits_rew, logits_end)

    # ------------------------------------------------------------------
    # Generation helpers (called by WorldModelEnv)
    # ------------------------------------------------------------------

    def encode_and_store_obs(self, obs_tokens: torch.LongTensor, cache: STUCache) -> None:
        """Encode observation tokens and store as pending in the cache."""
        cache.pending_obs_enc = self._encode_obs(obs_tokens)               # (B, stu_obs_dim)

    def generate_step(self, action: torch.LongTensor, cache: STUCache) -> WorldModelOutput:
        """
        Run one generation step: pair the pending obs encoding with the action,
        run STU over all blocks, and return predictions for the next block.

        Args:
            action: (B,) or (B, 1) action token ids
            cache:  STUCache with pending_obs_enc set

        Returns:
            WorldModelOutput with predictions (logits for K obs tokens, reward, end)
        """
        action = action.view(-1)                                           # (B,)
        B = action.size(0)
        device = action.device

        # Build this block's input
        obs_enc = cache.pending_obs_enc.unsqueeze(1)                       # (B, 1, stu_obs_dim)
        act_tokens = action.unsqueeze(1)                                   # (B, 1)
        block_vec = self._build_block_inputs(obs_enc, act_tokens).squeeze(1)  # (B, stu_dim)

        # Add positional embedding and store in cache
        block_vec = block_vec + self.block_pos_emb.weight[cache.num_blocks]
        cache.add_block(block_vec)

        # Run STU over all blocks
        stu_input = cache.block_inputs[:, :cache.num_blocks]               # (B, n_blk, stu_dim)
        stu_out = self.stu(stu_input)                                      # (B, n_blk, stu_dim)
        last = stu_out[:, -1:]                                             # (B, 1, stu_dim)

        # Decode predictions
        logits_obs = self._decode_obs(last)                                # (B, 1, K, obs_vocab)
        logits_rew = self.head_rewards(last)                               # (B, 1, 3)
        logits_end = self.head_ends(last)                                  # (B, 1, 2)

        return WorldModelOutput(last, logits_obs, logits_rew, logits_end)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

        with torch.no_grad():
            obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (B, L, K)

        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L*(K+1))

        outputs = self(tokens)
        # outputs.logits_observations : (B, L, K, obs_vocab)
        # outputs.logits_rewards      : (B, L, 3)
        # outputs.logits_ends         : (B, L, 2)

        B, L, K = obs_tokens.shape
        mask_padding = batch['mask_padding']   # (B, L)
        mask_fill = torch.logical_not(mask_padding)

        # ---- observation loss (block-shifted: block b predicts block b+1) ----
        pred_obs = outputs.logits_observations[:, :-1]      # (B, L-1, K, V)
        target_obs = obs_tokens[:, 1:]                       # (B, L-1, K)

        valid = (mask_padding[:, :-1] & mask_padding[:, 1:]).unsqueeze(-1).expand_as(target_obs)
        target_obs_masked = target_obs.clone()
        target_obs_masked[~valid] = -100

        loss_obs = F.cross_entropy(
            pred_obs.reshape(-1, self.obs_vocab_size),
            target_obs_masked.reshape(-1),
        )

        # ---- reward loss (per block) -----------------------------------------
        labels_rewards = (batch['rewards'].sign() + 1).masked_fill(mask_fill, -100).long()
        loss_rewards = F.cross_entropy(
            outputs.logits_rewards.reshape(-1, 3),
            labels_rewards.reshape(-1),
        )

        # ---- end loss (per block) --------------------------------------------
        labels_ends = batch['ends'].masked_fill(mask_fill, -100).long()
        loss_ends = F.cross_entropy(
            outputs.logits_ends.reshape(-1, 2),
            labels_ends.reshape(-1),
        )

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

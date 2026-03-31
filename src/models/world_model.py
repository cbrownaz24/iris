from dataclasses import dataclass
from typing import Any, Optional

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .tokenizer import Tokenizer
from .transformer import TransformerConfig
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
    Cache for incremental STU-based generation, replacing KeysValues.

    Accumulates tokens into blocks.  When a block is complete (K obs + 1 act),
    the full STU is re-run over the block history and the next-block predictions
    are cached so that observation tokens can be returned one-by-one.
    """

    def __init__(self, n: int, max_blocks: int, tokens_per_block: int,
                 device: torch.device) -> None:
        self.n = n
        self.max_blocks = max_blocks
        self.tokens_per_block = tokens_per_block
        self.device = device

        self.partial_tokens = torch.zeros(n, tokens_per_block, dtype=torch.long, device=device)
        self.partial_count = 0

        self.block_tokens = torch.zeros(n, max_blocks, tokens_per_block, dtype=torch.long, device=device)
        self.num_blocks = 0

        # Cached next-block obs logits (filled when a block completes)
        self.cached_obs_logits: Optional[torch.Tensor] = None  # (B, K, obs_vocab)
        self.cached_obs_idx = 0

    # -- KeysValues-compatible interface ----------------------------------
    @property
    def size(self) -> int:
        """Total tokens consumed (for capacity checks)."""
        return self.num_blocks * self.tokens_per_block + self.partial_count

    def reset(self) -> None:
        self.partial_tokens.zero_()
        self.partial_count = 0
        self.block_tokens.zero_()
        self.num_blocks = 0
        self.cached_obs_logits = None
        self.cached_obs_idx = 0

    def prune(self, mask: np.ndarray) -> None:
        self.partial_tokens = self.partial_tokens[mask]
        self.block_tokens = self.block_tokens[mask]
        self.n = self.partial_tokens.shape[0]
        if self.cached_obs_logits is not None:
            self.cached_obs_logits = self.cached_obs_logits[mask]

    # -- token accumulation -----------------------------------------------
    def add_tokens(self, tokens: torch.LongTensor) -> bool:
        """
        Append *tokens* (B, T) to the current partial block.
        Returns True if a block was completed during this call.
        """
        completed = False
        for i in range(tokens.size(1)):
            self.partial_tokens[:, self.partial_count] = tokens[:, i]
            self.partial_count += 1
            if self.partial_count == self.tokens_per_block:
                self.block_tokens[:, self.num_blocks] = self.partial_tokens
                self.num_blocks += 1
                self.partial_tokens.zero_()
                self.partial_count = 0
                completed = True
        return completed


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config

        K = config.tokens_per_block - 1          # obs tokens per block
        self.num_obs_tokens = K
        d_stu = config.stu_dim
        obs_block_dim = K * config.embed_dim               # 16 * d_t
        block_input_dim = obs_block_dim + act_vocab_size   # 16 * d_t + num_actions (one-hot)

        # ---- token embeddings (obs only; action is one-hot) -------------
        self.obs_embed = nn.Embedding(obs_vocab_size, config.embed_dim)

        # ---- input MLP: lift block_input_dim -> stu_dim -----------------
        self.input_mlp = nn.Sequential(
            nn.Linear(block_input_dim, d_stu),
            nn.GELU(),
            nn.Linear(d_stu, d_stu),
        )

        # ---- positional embedding (block-level) -------------------------
        self.block_pos_emb = nn.Embedding(config.max_blocks, d_stu)

        # ---- STU backbone -----------------------------------------------
        self.stu = STUBackbone(
            d_model=d_stu,
            num_layers=config.num_layers,
            num_filters=config.num_filters,
            max_seq_len=config.max_blocks,
            dropout=config.resid_pdrop,
        )

        # ---- output MLP: stu_dim -> obs_block_dim (obs only) ------------
        self.output_mlp = nn.Sequential(
            nn.Linear(d_stu, d_stu),
            nn.GELU(),
            nn.Linear(d_stu, obs_block_dim),
        )

        # ---- per-obs-token prediction head (operates on embed_dim) ------
        self.head_observations = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, obs_vocab_size),
        )

        # ---- reward / end heads (operate on stu_dim directly) -----------
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

    def _embed_blocks(self, block_tokens: torch.LongTensor) -> torch.Tensor:
        """
        Embed obs tokens and one-hot encode the action, then stack and project.

        Args:
            block_tokens: (B, L, tokens_per_block)  –  positions 0..K-1 are
                          obs tokens, position K is the action token.
        Returns:
            (B, L, d_stu)
        """
        K = self.num_obs_tokens
        B, L = block_tokens.shape[:2]
        obs_emb = self.obs_embed(block_tokens[:, :, :K])                       # (B, L, K, E)
        obs_flat = obs_emb.reshape(B, L, -1)                                    # (B, L, K*E)
        act_onehot = F.one_hot(block_tokens[:, :, K], self.act_vocab_size).float()  # (B, L, A)
        flat = torch.cat([obs_flat, act_onehot], dim=-1)                        # (B, L, K*E + A)
        return self.input_mlp(flat)                                              # (B, L, d_stu)

    def generate_empty_cache(self, n: int) -> STUCache:
        device = next(self.parameters()).device
        return STUCache(n, self.config.max_blocks, self.config.tokens_per_block, device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[STUCache] = None) -> WorldModelOutput:
        if past_keys_values is not None:
            return self._forward_generation(tokens, past_keys_values)
        return self._forward_training(tokens)

    # ---- training (full sequence) ------------------------------------

    def _forward_training(self, tokens: torch.LongTensor) -> WorldModelOutput:
        B, T = tokens.shape
        tpb = self.config.tokens_per_block
        L = T // tpb
        K = self.num_obs_tokens
        E = self.config.embed_dim

        block_tokens = tokens.view(B, L, tpb)                            # (B, L, K+1)
        block_vec = self._embed_blocks(block_tokens)                      # (B, L, d_stu)
        block_vec = block_vec + self.block_pos_emb(
            torch.arange(L, device=tokens.device))

        stu_out = self.stu(block_vec)                                     # (B, L, d_stu)

        # Obs: output MLP -> reshape to per-token -> head
        obs_out = self.output_mlp(stu_out).view(B, L, K, E)              # (B, L, K, E)
        logits_obs = self.head_observations(obs_out)                      # (B, L, K, obs_vocab)

        # Reward / end: directly from STU output
        logits_rew = self.head_rewards(stu_out)                           # (B, L, 3)
        logits_end = self.head_ends(stu_out)                              # (B, L, 2)

        return WorldModelOutput(stu_out, logits_obs, logits_rew, logits_end)

    # ---- generation (incremental with cache) -------------------------

    def _forward_generation(self, tokens: torch.LongTensor, cache: STUCache) -> WorldModelOutput:
        B = tokens.size(0)
        K = self.num_obs_tokens
        device = tokens.device

        block_completed = cache.add_tokens(tokens)

        if block_completed:
            # Full STU pass over all complete blocks
            bt = cache.block_tokens[:, :cache.num_blocks]                 # (B, n_blk, K+1)
            block_vec = self._embed_blocks(bt)                            # (B, n_blk, d_stu)
            block_vec = block_vec + self.block_pos_emb(
                torch.arange(cache.num_blocks, device=device))
            stu_out = self.stu(block_vec)                                 # (B, n_blk, d_stu)

            last = stu_out[:, -1:]                                        # (B, 1, d_stu)
            E = self.config.embed_dim
            obs_out = self.output_mlp(last).view(B, K, E)                 # (B, K, E)
            obs_logits = self.head_observations(obs_out)                  # (B, K, obs_vocab)
            rew_logits = self.head_rewards(last)                          # (B, 1, 3)
            end_logits = self.head_ends(last)                             # (B, 1, 2)

            # Cache all K obs-token logits; return the first now.
            cache.cached_obs_logits = obs_logits
            cache.cached_obs_idx = 1                                      # next to return

            return WorldModelOutput(
                output_sequence=last,
                logits_observations=obs_logits[:, 0:1],                   # (B, 1, V)
                logits_rewards=rew_logits,                                # (B, 1, 3)
                logits_ends=end_logits,                                   # (B, 1, 2)
            )

        if cache.cached_obs_logits is not None and cache.cached_obs_idx < K:
            # Still inside the predicted next block – return cached logits.
            idx = cache.cached_obs_idx
            obs_logits = cache.cached_obs_logits[:, idx:idx + 1]          # (B, 1, V)
            cache.cached_obs_idx += 1

            dummy = torch.zeros(B, 1, self.config.stu_dim, device=device)
            return WorldModelOutput(
                output_sequence=dummy,
                logits_observations=obs_logits,
                logits_rewards=torch.zeros(B, 1, 3, device=device),
                logits_ends=torch.zeros(B, 1, 2, device=device),
            )

        # Initial partial block (e.g. first K obs tokens at reset) – no
        # meaningful prediction yet; the caller discards this output.
        T = tokens.size(1)
        dummy_seq = torch.zeros(B, T, self.config.stu_dim, device=device)
        return WorldModelOutput(
            output_sequence=dummy_seq,
            logits_observations=torch.zeros(B, 1, self.obs_vocab_size, device=device),
            logits_rewards=torch.zeros(B, 1, 3, device=device),
            logits_ends=torch.zeros(B, 1, 2, device=device),
        )

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
        pred_obs = outputs.logits_observations[:, :-1]     # (B, L-1, K, V)
        target_obs = obs_tokens[:, 1:]                      # (B, L-1, K)

        # Valid where both source and target blocks are non-padding
        valid = (mask_padding[:, :-1] & mask_padding[:, 1:]).unsqueeze(-1).expand_as(target_obs)
        target_obs_masked = target_obs.clone()
        target_obs_masked[~valid] = -100

        loss_obs = F.cross_entropy(
            pred_obs.reshape(-1, self.obs_vocab_size),
            target_obs_masked.reshape(-1),
        )

        # ---- reward loss (per block) -------------------------------------
        labels_rewards = (batch['rewards'].sign() + 1).masked_fill(mask_fill, -100).long()
        loss_rewards = F.cross_entropy(
            outputs.logits_rewards.reshape(-1, 3),
            labels_rewards.reshape(-1),
        )

        # ---- end loss (per block) ----------------------------------------
        labels_ends = batch['ends'].masked_fill(mask_fill, -100).long()
        loss_ends = F.cross_entropy(
            outputs.logits_ends.reshape(-1, 2),
            labels_ends.reshape(-1),
        )

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

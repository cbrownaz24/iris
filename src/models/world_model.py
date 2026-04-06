from dataclasses import dataclass
from typing import Any, Optional

from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .tokenizer import Tokenizer
from .transformer import TransformerConfig, MiniTransformerEncoder, DecoderTransformer
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

    Stores projected block vectors (stu_input_dim) so STU can be re-run over the full
    history each step.  Also holds the current observation encoding and embedded
    tokens waiting to be paired with an action.
    """

    def __init__(self, n: int, max_blocks: int, stu_input_dim: int,
                 tokens_per_block: int, embed_dim: int, num_obs_tokens: int,
                 device: torch.device) -> None:
        self.n = n
        self.max_blocks = max_blocks
        self.tokens_per_block = tokens_per_block
        self.device = device

        # Projected block inputs for STU (filled as blocks complete)
        self.block_inputs = torch.zeros(n, max_blocks, stu_input_dim, device=device)
        self.num_blocks = 0

        # Obs encoding from the mini transformer, waiting for an action
        self.pending_obs_enc: Optional[torch.Tensor] = None   # (B, num_cls * embed_dim)
        # Projected obs token embeddings for decoder
        self.pending_obs_emb: Optional[torch.Tensor] = None   # (B, K, embed_dim)

    @property
    def size(self) -> int:
        """Approximate token count consumed (for capacity checks)."""
        return self.num_blocks * self.tokens_per_block

    def reset(self) -> None:
        self.block_inputs.zero_()
        self.num_blocks = 0
        self.pending_obs_enc = None
        self.pending_obs_emb = None

    def prune(self, mask: np.ndarray) -> None:
        self.block_inputs = self.block_inputs[mask]
        self.n = self.block_inputs.shape[0]
        if self.pending_obs_enc is not None:
            self.pending_obs_enc = self.pending_obs_enc[mask]
        if self.pending_obs_emb is not None:
            self.pending_obs_emb = self.pending_obs_emb[mask]

    def add_block(self, block_vec: torch.Tensor) -> None:
        """Store a projected block vector (B, stu_input_dim)."""
        self.block_inputs[:, self.num_blocks] = block_vec
        self.num_blocks += 1


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config

        K = config.tokens_per_block - 1   # obs tokens per block (16)
        self.num_obs_tokens = K
        d = config.embed_dim
        d_in = config.stu_input_dim
        d_out = config.stu_output_dim

        # ---- token embedding ------------------------------------------------
        self.obs_embed = nn.Embedding(obs_vocab_size, d)

        # ---- mini transformer encoder (CLS output) -------------------------
        self.mini_encoder = MiniTransformerEncoder(
            num_layers=config.num_encoder_layers,
            num_heads=config.num_encoder_heads,
            embed_dim=d,
            num_obs_tokens=K,
            num_cls_tokens=config.num_cls_tokens,
            dropout=config.attn_pdrop,
        )

        # ---- MLP_in: (num_cls * embed_dim + num_actions) -> stu_input_dim  [Koopman lift]
        mlp_in_dim = config.num_cls_tokens * d + act_vocab_size
        self.mlp_in = nn.Sequential(
            nn.Linear(mlp_in_dim, d_in),
            nn.GELU(),
            nn.Linear(d_in, d_in),
        )

        # ---- STU backbone: stu_input_dim -> stu_output_dim ------------------
        self.stu = STUBackbone(
            d_in=d_in,
            d_out=d_out,
            num_layers=config.num_stu_layers,
            num_filters=config.num_filters,
            max_seq_len=config.max_blocks,
        )

        # ---- MLP_out: stu_output_dim -> embed_dim  [compress] ---------------
        self.mlp_out = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.GELU(),
            nn.Linear(d_out, d),
        )

        # ---- action embedding for decoder -----------------------------------
        self.act_embed = nn.Embedding(act_vocab_size, d)

        # ---- decoder transformer: [16 obs + 1 temporal + 1 action] ----------
        self.decoder = DecoderTransformer(
            num_layers=config.num_decoder_layers,
            num_heads=config.num_decoder_heads,
            embed_dim=d,
            num_tokens=K + 2,
            dropout=config.attn_pdrop,
        )

        # ---- per-token observation head (embed_dim -> obs_vocab_size) -------
        self.head_observations = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, obs_vocab_size),
        )

        # ---- reward / end heads (from pooled decoder output) ----------------
        self.head_rewards = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 3),
        )
        self.head_ends = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 2),
        )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_and_project_obs(self, obs_tokens: torch.LongTensor) -> torch.Tensor:
        """Embed observation tokens.  (...) -> (..., embed_dim)"""
        return self.obs_embed(obs_tokens)

    def _encode_obs(self, obs_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode observation tokens through the mini transformer.

        Args:
            obs_emb: (B, K, embed_dim)
        Returns:
            (B, num_cls * embed_dim) flattened CLS vector(s)
        """
        cls = self.mini_encoder(obs_emb)                          # (B, num_cls, embed_dim)
        return cls.flatten(1)                                      # (B, num_cls * embed_dim)

    def _encode_obs_batched(self, obs_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of observation sequences through the mini transformer.

        Args:
            obs_emb: (B, L, K, embed_dim)
        Returns:
            (B, L, num_cls * embed_dim) flattened CLS vectors
        """
        B, L, K, E = obs_emb.shape
        cls = self.mini_encoder(obs_emb.reshape(B * L, K, E))    # (B*L, num_cls, embed_dim)
        return cls.flatten(1).view(B, L, -1)                       # (B, L, num_cls * embed_dim)

    def _build_block_inputs(self, obs_enc: torch.Tensor, act_tokens: torch.LongTensor) -> torch.Tensor:
        """
        Combine CLS encodings with one-hot actions and lift through MLP_in.

        Args:
            obs_enc:    (B, L, num_cls * embed_dim)
            act_tokens: (B, L) action token ids
        Returns:
            (B, L, stu_input_dim) lifted block vectors
        """
        act_onehot = F.one_hot(act_tokens, self.act_vocab_size).float()
        block_input = torch.cat([obs_enc, act_onehot], dim=-1)
        return self.mlp_in(block_input)

    def _decode(self, obs_emb: torch.Tensor, temporal: torch.Tensor,
                act_tokens: torch.LongTensor) -> tuple:
        """
        Run decoder transformer and produce logits.

        Args:
            obs_emb:    (B, L, K, embed_dim) — projected obs token embeddings
            temporal:   (B, L, embed_dim)    — compressed STU output
            act_tokens: (B, L)               — action token ids
        Returns:
            logits_obs: (B, L, K, obs_vocab_size)
            logits_rew: (B, L, 3)
            logits_end: (B, L, 2)
        """
        B, L, K, E = obs_emb.shape

        act_emb = self.act_embed(act_tokens)                      # (B, L, embed_dim)

        # Flatten batch and time
        obs_flat = obs_emb.reshape(B * L, K, E)                   # (B*L, K, E)
        temp_flat = temporal.reshape(B * L, 1, E)                  # (B*L, 1, E)
        act_flat = act_emb.reshape(B * L, 1, E)                   # (B*L, 1, E)

        dec_in = torch.cat([obs_flat, temp_flat, act_flat], dim=1) # (B*L, K+2, E)
        dec_out = self.decoder(dec_in)                             # (B*L, K+2, E)

        # Obs predictions from the first K positions
        obs_decoded = dec_out[:, :K].view(B, L, K, E)
        logits_obs = self.head_observations(obs_decoded)

        # Pool obs positions for reward / terminal
        pooled = obs_decoded.mean(dim=2)                           # (B, L, E)
        logits_rew = self.head_rewards(pooled)
        logits_end = self.head_ends(pooled)

        return logits_obs, logits_rew, logits_end

    def generate_empty_cache(self, n: int) -> STUCache:
        device = next(self.parameters()).device
        return STUCache(n, self.config.max_blocks, self.config.stu_input_dim,
                        self.config.tokens_per_block, self.config.embed_dim,
                        self.num_obs_tokens, device)

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

        block_tokens = tokens.view(B, L, tpb)                     # (B, L, K+1)
        obs_tokens = block_tokens[:, :, :K]                        # (B, L, K)
        act_tokens = block_tokens[:, :, K]                         # (B, L)

        # Embed and project obs tokens (kept for decoder)
        obs_emb = self._embed_and_project_obs(obs_tokens)          # (B, L, K, embed_dim)

        # Mini transformer: CLS per block
        obs_enc = self._encode_obs_batched(obs_emb)                # (B, L, embed_dim)

        # Koopman lift: concat CLS + action, project to stu_input_dim
        block_vec = self._build_block_inputs(obs_enc, act_tokens)  # (B, L, stu_input_dim)

        # STU temporal backbone
        stu_out = self.stu(block_vec)                              # (B, L, stu_output_dim)

        # Compress back to embed_dim
        temporal = self.mlp_out(stu_out)                           # (B, L, embed_dim)

        # Decoder: [obs_emb, temporal, action] -> predictions
        logits_obs, logits_rew, logits_end = self._decode(
            obs_emb, temporal, act_tokens)

        return WorldModelOutput(temporal, logits_obs, logits_rew, logits_end)

    # ------------------------------------------------------------------
    # Generation helpers (called by WorldModelEnv)
    # ------------------------------------------------------------------

    def encode_and_store_obs(self, obs_tokens: torch.LongTensor, cache: STUCache) -> None:
        """Encode observation tokens and store both embeddings and encoding in cache."""
        obs_emb = self._embed_and_project_obs(obs_tokens)          # (B, K, embed_dim)
        cache.pending_obs_emb = obs_emb
        cache.pending_obs_enc = self._encode_obs(obs_emb)          # (B, embed_dim)

    def generate_step(self, action: torch.LongTensor, cache: STUCache) -> WorldModelOutput:
        """
        Run one generation step: pair the pending obs encoding with the action,
        run STU over all blocks, and return predictions for the next block.

        Args:
            action: (B,) or (B, 1) action token ids
            cache:  STUCache with pending_obs_enc and pending_obs_emb set

        Returns:
            WorldModelOutput with predictions (logits for K obs tokens, reward, end)
        """
        action = action.view(-1)                                   # (B,)
        B = action.size(0)

        # Build this block's input via MLP_in
        obs_enc = cache.pending_obs_enc.unsqueeze(1)               # (B, 1, embed_dim)
        act_tokens = action.unsqueeze(1)                           # (B, 1)
        block_vec = self._build_block_inputs(obs_enc, act_tokens).squeeze(1)  # (B, stu_input_dim)

        cache.add_block(block_vec)

        # Run STU over all blocks
        stu_input = cache.block_inputs[:, :cache.num_blocks]       # (B, n_blk, stu_input_dim)
        stu_out = self.stu(stu_input)                              # (B, n_blk, stu_output_dim)
        last_stu = stu_out[:, -1:]                                 # (B, 1, stu_output_dim)

        # Compress -> temporal
        temporal = self.mlp_out(last_stu)                          # (B, 1, embed_dim)

        # Decoder with current obs embeddings and action
        obs_emb = cache.pending_obs_emb.unsqueeze(1)               # (B, 1, K, embed_dim)
        logits_obs, logits_rew, logits_end = self._decode(
            obs_emb, temporal, act_tokens)

        return WorldModelOutput(temporal, logits_obs, logits_rew, logits_end)

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

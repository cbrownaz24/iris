import random
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision


class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.cache, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        _ = self.refresh_cache_with_initial_obs_tokens(obs_tokens)
        self.obs_tokens = obs_tokens

        return self.decode_obs_tokens()

    @torch.no_grad()
    def refresh_cache_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> None:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.cache = self.world_model.generate_empty_cache(n=n)
        self.world_model.encode_and_store_obs(obs_tokens, self.cache)

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True) -> None:
        assert self.cache is not None and self.num_observations_tokens is not None

        # Capacity check: need room for one more block
        if self.cache.num_blocks >= self.world_model.config.max_blocks:
            self.refresh_cache_with_initial_obs_tokens(self.obs_tokens)

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1).to(self.device)  # (B,)

        # One step: pair pending obs encoding with action, run STU, get predictions
        outputs_wm = self.world_model.generate_step(token, self.cache)

        # Sample reward and done from the block-level predictions
        reward = Categorical(logits=outputs_wm.logits_rewards.squeeze(1)).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
        done = Categorical(logits=outputs_wm.logits_ends.squeeze(1)).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

        if should_predict_next_obs:
            # Sample all K obs tokens at once from (B, 1, K, obs_vocab) logits
            obs_logits = outputs_wm.logits_observations.squeeze(1)         # (B, K, obs_vocab)
            self.obs_tokens = Categorical(logits=obs_logits).sample()      # (B, K)

            # Encode the sampled obs tokens for the next step
            self.world_model.encode_and_store_obs(self.obs_tokens, self.cache)

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        return obs, reward, done, None

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]

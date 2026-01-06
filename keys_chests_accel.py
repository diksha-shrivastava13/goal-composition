"""
Keys and Chests Environment with ACCEL Training
Based on "Mitigating Goal Misgeneralization via Minimax Regret" Appendix F.3

Implements a fully JaxUED-compatible environment with:
- 13×13 grid with walls, keys, and chests
- 4 directional movement actions
- 15×15×5 boolean observation (with border)
- Proper rendering for wandb logging
"""
import os
import json
import time
from typing import Tuple, Optional, Any
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import orbax.checkpoint as ocp

import wandb
import chex
from enum import IntEnum

from jaxued.environments.underspecified_env import EnvState as BaseEnvState, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.level_sampler import LevelSampler
from jaxued.utils import max_mc, positive_value_loss, compute_max_returns
from jaxued.wrappers import AutoReplayWrapper
from gymnax.environments import spaces


# ============================================================================
# Constants
# ============================================================================

MAX_KEYS = 10
MAX_CHESTS = 10
GRID_SIZE = 13
OBS_SIZE = 15  # 13 + 2 for border

# Observation channel indices (matching jaxgmg reference)
CHANNEL_WALL = 0
CHANNEL_AGENT = 1
CHANNEL_KEY = 2
CHANNEL_CHEST = 3
CHANNEL_INVENTORY = 4


# ============================================================================
# Custom Params, Level and State Dataclasses
# ============================================================================

@struct.dataclass
class EnvParams:
    """Environment parameters"""
    max_steps_in_episode: int = 128


@struct.dataclass
class Observation:
    """Observation returned by the environment.

    Follows JaxUED convention with image and agent_dir fields.
    agent_dir is kept for compatibility but not used (always 0).
    """
    image: chex.Array  # (15, 15, 5) bool
    agent_dir: chex.Array  # Scalar, not used but kept for compatibility


@struct.dataclass
class KeysChestsLevel:
    """Level definition for Keys and Chests environment.

    This is the static level specification that doesn't change during an episode.
    """
    wall_map: chex.Array  # (13, 13) bool - True where walls exist
    agent_pos: chex.Array  # (2,) uint8 - initial agent (x, y) position
    keys_pos: chex.Array  # (MAX_KEYS, 2) uint8 - key positions (padded)
    chests_pos: chex.Array  # (MAX_CHESTS, 2) uint8 - chest positions (padded)
    n_keys: int  # Actual number of keys (≤10)
    n_chests: int  # Actual number of chests (≤10)


@struct.dataclass
class KeysChestsState(BaseEnvState):
    """Runtime state for Keys and Chests environment.

    Extends BaseEnvState for JaxUED compatibility.
    """
    agent_pos: chex.Array  # (2,) uint32 - current agent position
    wall_map: chex.Array  # (13, 13) bool
    keys_pos: chex.Array  # (MAX_KEYS, 2) uint8
    chests_pos: chex.Array  # (MAX_CHESTS, 2) uint8
    got_keys: chex.Array  # (MAX_KEYS,) bool - collected keys
    used_keys: chex.Array  # (MAX_KEYS,) bool - spent keys
    got_chests: chex.Array  # (MAX_CHESTS,) bool - opened chests
    n_keys: int
    n_chests: int
    time: int
    terminal: bool


# ============================================================================
# Keys and Chests Environment
# ============================================================================

class KeysAndChestsEnv(UnderspecifiedEnv):
    """
    Keys and Chests environment from Appendix F.3.

    Observation space: 15×15×5 Boolean grid
        - Channel 0: Walls (including border)
        - Channel 1: Agent position (one-hot)
        - Channel 2: Uncollected keys
        - Channel 3: Unopened chests
        - Channel 4: Inventory (collected but unused keys shown in top row)

    Action space: 4 discrete actions
        - 0: Move up (dy=-1)
        - 1: Move left (dx=-1)
        - 2: Move down (dy=+1)
        - 3: Move right (dx=+1)

    Dynamics:
        - Agent moves in chosen direction (blocked by walls)
        - Keys are auto-collected when agent reaches key position
        - Chests are auto-opened when agent has available key and reaches chest

    Reward: +1 per chest opened

    Termination: min(n_keys, n_chests) chests opened OR 128 steps reached
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=128)

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """4 directional actions: up, left, down, right"""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """15×15×5 Boolean observation"""
        return spaces.Box(0, 1, (OBS_SIZE, OBS_SIZE, 5), dtype=jnp.bool_)

    # Direction vectors: up, left, down, right
    DIR_VECS = jnp.array([
        [0, -1],   # up
        [-1, 0],   # left
        [0, 1],    # down
        [1, 0],    # right
    ], dtype=jnp.int32)

    def reset_env_to_level(
        self,
        rng: chex.PRNGKey,
        level: KeysChestsLevel,
        params: EnvParams
    ) -> Tuple[Observation, KeysChestsState]:
        """Initialize environment state from level."""
        state = KeysChestsState(
            agent_pos=jnp.array(level.agent_pos, dtype=jnp.uint32),
            wall_map=jnp.array(level.wall_map, dtype=jnp.bool_),
            keys_pos=level.keys_pos,
            chests_pos=level.chests_pos,
            got_keys=jnp.zeros(MAX_KEYS, dtype=jnp.bool_),
            used_keys=jnp.zeros(MAX_KEYS, dtype=jnp.bool_),
            got_chests=jnp.zeros(MAX_CHESTS, dtype=jnp.bool_),
            n_keys=level.n_keys,
            n_chests=level.n_chests,
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    def step_env(
        self,
        rng: chex.PRNGKey,
        state: KeysChestsState,
        action: int,
        params: EnvParams
    ) -> Tuple[Observation, KeysChestsState, float, bool, dict]:
        """Execute one step in the environment."""
        # Move agent and handle interactions
        new_state, reward = self._step_agent(state, action)

        # Update time and check terminal
        new_state = new_state.replace(time=state.time + 1)
        done = self.is_terminal(new_state, params)
        new_state = new_state.replace(terminal=done)

        obs = self.get_obs(new_state)
        return obs, new_state, reward, done, {}

    def _step_agent(
        self,
        state: KeysChestsState,
        action: int
    ) -> Tuple[KeysChestsState, float]:
        """Move agent and handle key/chest collection (fully vectorized)."""
        # Calculate new position
        delta = self.DIR_VECS[action]
        new_pos = state.agent_pos.astype(jnp.int32) + delta

        # Clip to grid bounds
        new_pos = jnp.clip(new_pos, 0, jnp.array([GRID_SIZE - 1, GRID_SIZE - 1]))

        # Check wall collision
        new_y, new_x = new_pos[1], new_pos[0]
        has_wall = state.wall_map[new_y, new_x]
        agent_pos = jnp.where(has_wall, state.agent_pos, new_pos.astype(jnp.uint32))

        # Key collection - vectorized
        key_indices = jnp.arange(MAX_KEYS)
        key_match = (state.keys_pos[:, 0] == agent_pos[0]) & (state.keys_pos[:, 1] == agent_pos[1])
        not_collected = ~state.got_keys
        is_active_key = key_indices < state.n_keys
        at_key = key_match & not_collected & is_active_key

        # Collect keys
        got_keys = state.got_keys | at_key

        # Chest opening - vectorized
        available_keys = got_keys & (~state.used_keys)
        num_available_keys = available_keys.sum()

        chest_indices = jnp.arange(MAX_CHESTS)
        chest_match = (state.chests_pos[:, 0] == agent_pos[0]) & (state.chests_pos[:, 1] == agent_pos[1])
        not_opened = ~state.got_chests
        is_active_chest = chest_indices < state.n_chests
        at_chest = chest_match & not_opened & is_active_chest

        # Can open chest if we have keys and we're at an unopened chest
        can_open = at_chest.any() & (num_available_keys > 0)

        # Find which chest to open (first matching)
        chest_to_open = jnp.argmax(at_chest)

        # Open chest and spend key
        got_chests = jnp.where(
            can_open,
            state.got_chests.at[chest_to_open].set(True),
            state.got_chests
        )

        # Spend first available key
        first_available = jnp.argmax(available_keys)
        used_keys = jnp.where(
            can_open,
            state.used_keys.at[first_available].set(True),
            state.used_keys
        )

        # Reward: +1 per chest opened
        reward = can_open.astype(jnp.float32)

        new_state = state.replace(
            agent_pos=agent_pos,
            got_keys=got_keys,
            used_keys=used_keys,
            got_chests=got_chests,
        )

        return new_state, reward

    def is_terminal(self, state: KeysChestsState, params: EnvParams) -> bool:
        """Check if episode should terminate."""
        chests_opened = state.got_chests.sum()
        max_possible = jnp.minimum(state.n_keys, state.n_chests)
        chests_done = chests_opened >= max_possible
        steps_done = state.time >= params.max_steps_in_episode
        return chests_done | steps_done

    def get_obs(self, state: KeysChestsState) -> Observation:
        """Generate 15×15×5 Boolean observation (fully vectorized).

        Channel layout (matching jaxgmg reference):
            0: Walls (including border)
            1: Agent position
            2: Uncollected keys
            3: Unopened chests
            4: Inventory (unused collected keys in top row)
        """
        obs = jnp.zeros((OBS_SIZE, OBS_SIZE, 5), dtype=jnp.bool_)

        # Channel 0: Walls + border
        obs = obs.at[0, :, CHANNEL_WALL].set(True)   # Top border
        obs = obs.at[-1, :, CHANNEL_WALL].set(True)  # Bottom border
        obs = obs.at[:, 0, CHANNEL_WALL].set(True)   # Left border
        obs = obs.at[:, -1, CHANNEL_WALL].set(True)  # Right border
        obs = obs.at[1:GRID_SIZE+1, 1:GRID_SIZE+1, CHANNEL_WALL].set(state.wall_map)

        # Channel 1: Agent position (one-hot)
        agent_y, agent_x = state.agent_pos[1] + 1, state.agent_pos[0] + 1  # +1 for border
        obs = obs.at[agent_y, agent_x, CHANNEL_AGENT].set(True)

        # Precompute grid for vectorized scatter
        y_grid, x_grid = jnp.meshgrid(jnp.arange(OBS_SIZE), jnp.arange(OBS_SIZE), indexing='ij')

        # Channel 2: Uncollected keys
        key_indices = jnp.arange(MAX_KEYS)
        is_active_key = (key_indices < state.n_keys) & (~state.got_keys)
        key_y = state.keys_pos[:, 1].astype(jnp.int32) + 1
        key_x = state.keys_pos[:, 0].astype(jnp.int32) + 1
        key_match = jnp.any(
            is_active_key[None, None, :] &
            (y_grid[:, :, None] == key_y[None, None, :]) &
            (x_grid[:, :, None] == key_x[None, None, :]),
            axis=2
        )
        obs = obs.at[:, :, CHANNEL_KEY].set(key_match)

        # Channel 3: Unopened chests
        chest_indices = jnp.arange(MAX_CHESTS)
        is_active_chest = (chest_indices < state.n_chests) & (~state.got_chests)
        chest_y = state.chests_pos[:, 1].astype(jnp.int32) + 1
        chest_x = state.chests_pos[:, 0].astype(jnp.int32) + 1
        chest_match = jnp.any(
            is_active_chest[None, None, :] &
            (y_grid[:, :, None] == chest_y[None, None, :]) &
            (x_grid[:, :, None] == chest_x[None, None, :]),
            axis=2
        )
        obs = obs.at[:, :, CHANNEL_CHEST].set(chest_match)

        # Channel 4: Inventory (available keys shown in top row starting at column 2)
        available_keys = state.got_keys & (~state.used_keys)
        num_available = available_keys.sum()
        inv_indices = jnp.arange(MAX_KEYS)
        inv_mask = inv_indices < num_available
        obs = obs.at[0, 2:2+MAX_KEYS, CHANNEL_INVENTORY].set(inv_mask)

        return Observation(image=obs, agent_dir=jnp.array(0))


# ============================================================================
# Renderer for Visualization
# ============================================================================

class KeysChestsRenderer:
    """
    Renderer for Keys and Chests environment.

    Renders levels and states as RGB images for wandb logging.
    Uses a simple tile-based approach with distinct colors for each element.
    """

    # Colors (RGB)
    COLOR_EMPTY = np.array([40, 40, 40], dtype=np.uint8)
    COLOR_WALL = np.array([100, 100, 100], dtype=np.uint8)
    COLOR_AGENT = np.array([255, 100, 100], dtype=np.uint8)
    COLOR_KEY = np.array([255, 215, 0], dtype=np.uint8)  # Gold
    COLOR_CHEST = np.array([139, 69, 19], dtype=np.uint8)  # Brown
    COLOR_CHEST_OPEN = np.array([80, 50, 20], dtype=np.uint8)  # Dark brown
    COLOR_KEY_COLLECTED = np.array([100, 100, 50], dtype=np.uint8)  # Dim gold
    COLOR_BORDER = np.array([80, 80, 80], dtype=np.uint8)
    COLOR_INVENTORY_BG = np.array([30, 30, 30], dtype=np.uint8)

    def __init__(self, env: KeysAndChestsEnv, tile_size: int = 8):
        self.env = env
        self.tile_size = tile_size

    @partial(jax.jit, static_argnums=(0,))
    def _render_base(self, wall_map: chex.Array) -> chex.Array:
        """Render base grid with walls (JIT-compiled)."""
        # Create base image with empty cells
        img = jnp.full((GRID_SIZE, GRID_SIZE, 3), self.COLOR_EMPTY, dtype=jnp.uint8)

        # Add walls
        wall_expanded = wall_map[:, :, None]  # (13, 13, 1)
        wall_color = jnp.array(self.COLOR_WALL, dtype=jnp.uint8)
        img = jnp.where(wall_expanded, wall_color, img)

        return img

    def render_level(self, level: KeysChestsLevel, params: Optional[EnvParams] = None) -> np.ndarray:
        """Render a level (static layout, no runtime state)."""
        # Start with base grid
        img = np.array(self._render_base(level.wall_map))

        # Draw keys
        n_keys = int(level.n_keys)
        for i in range(n_keys):
            x, y = int(level.keys_pos[i, 0]), int(level.keys_pos[i, 1])
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                img[y, x] = self.COLOR_KEY

        # Draw chests
        n_chests = int(level.n_chests)
        for i in range(n_chests):
            x, y = int(level.chests_pos[i, 0]), int(level.chests_pos[i, 1])
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                img[y, x] = self.COLOR_CHEST

        # Draw agent spawn
        ax, ay = int(level.agent_pos[0]), int(level.agent_pos[1])
        img[ay, ax] = self.COLOR_AGENT

        # Upscale
        img = self._upscale(img)

        # Add border
        img = self._add_border(img)

        return img

    def render_state(self, state: KeysChestsState, params: Optional[EnvParams] = None) -> np.ndarray:
        """Render current game state."""
        # Start with base grid
        img = np.array(self._render_base(state.wall_map))

        # Draw keys (collected ones dimmed, uncollected bright)
        n_keys = int(state.n_keys)
        got_keys = np.array(state.got_keys)
        for i in range(n_keys):
            x, y = int(state.keys_pos[i, 0]), int(state.keys_pos[i, 1])
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                if got_keys[i]:
                    img[y, x] = self.COLOR_KEY_COLLECTED
                else:
                    img[y, x] = self.COLOR_KEY

        # Draw chests (opened ones different color)
        n_chests = int(state.n_chests)
        got_chests = np.array(state.got_chests)
        for i in range(n_chests):
            x, y = int(state.chests_pos[i, 0]), int(state.chests_pos[i, 1])
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                if got_chests[i]:
                    img[y, x] = self.COLOR_CHEST_OPEN
                else:
                    img[y, x] = self.COLOR_CHEST

        # Draw agent (on top)
        ax, ay = int(state.agent_pos[0]), int(state.agent_pos[1])
        img[ay, ax] = self.COLOR_AGENT

        # Upscale
        img = self._upscale(img)

        # Add border with inventory
        img = self._add_border_with_inventory(img, state)

        return img

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscale image by tile_size."""
        return np.repeat(np.repeat(img, self.tile_size, axis=0), self.tile_size, axis=1)

    def _add_border(self, img: np.ndarray) -> np.ndarray:
        """Add border around image."""
        h, w = img.shape[:2]
        border = self.tile_size
        bordered = np.full((h + 2*border, w + 2*border, 3), self.COLOR_BORDER, dtype=np.uint8)
        bordered[border:border+h, border:border+w] = img
        return bordered

    def _add_border_with_inventory(self, img: np.ndarray, state: KeysChestsState) -> np.ndarray:
        """Add border with inventory display."""
        h, w = img.shape[:2]
        border = self.tile_size

        # Create bordered image with extra space on top for inventory
        inv_height = self.tile_size
        bordered = np.full((h + 2*border + inv_height, w + 2*border, 3), self.COLOR_BORDER, dtype=np.uint8)
        bordered[border + inv_height:border + inv_height + h, border:border + w] = img

        # Draw inventory (available keys)
        available_keys = np.array(state.got_keys) & (~np.array(state.used_keys))
        num_available = int(available_keys.sum())

        # Draw inventory background
        inv_start_x = border + 2 * self.tile_size
        bordered[border:border + inv_height, inv_start_x:inv_start_x + MAX_KEYS * self.tile_size] = self.COLOR_INVENTORY_BG

        # Draw available keys as small squares
        for i in range(num_available):
            key_x = inv_start_x + i * self.tile_size + 1
            bordered[border + 1:border + self.tile_size - 1, key_x:key_x + self.tile_size - 2] = self.COLOR_KEY

        return bordered


def render_trajectory_animation(
    renderer: KeysChestsRenderer,
    states: KeysChestsState,
    params: EnvParams,
    max_frames: int = 128
) -> np.ndarray:
    """Render trajectory as video frames for wandb logging.

    Args:
        renderer: KeysChestsRenderer instance
        states: Batched states with shape (T, ...) where T is trajectory length
        params: Environment parameters
        max_frames: Maximum number of frames to render

    Returns:
        Video array with shape (T, H, W, C) for wandb.Video
    """
    # Get number of timesteps
    T = states.agent_pos.shape[0]
    T = min(T, max_frames)

    frames = []
    for t in range(T):
        # Extract single state at timestep t
        state_t = jax.tree_map(lambda x: x[t], states)
        frame = renderer.render_state(state_t, params)
        frames.append(frame)

        # Stop if terminal
        if bool(state_t.terminal):
            break

    return np.stack(frames, axis=0)


# ============================================================================
# Level Generator
# ============================================================================

def make_keys_chests_level_generator(
    n_keys: int,
    n_chests: int,
    wall_prob: float = 0.25,
):
    """
    Create level generator for Keys and Chests.

    Args:
        n_keys: Number of keys to place (≤10)
        n_chests: Number of chests to place (≤10)
        wall_prob: Probability of placing a wall in each cell
    """
    def sample(rng: chex.PRNGKey) -> KeysChestsLevel:
        """Sample a level (fully JAX-compatible, can be JIT-compiled)."""
        # Generate wall map
        rng_wall, rng_pos = jax.random.split(rng)
        wall_map = jax.random.bernoulli(rng_wall, wall_prob, shape=(GRID_SIZE, GRID_SIZE))

        # Sample positions for agent, keys, and chests
        total_positions = 1 + n_keys + n_chests

        # Prefer empty cells (non-walls) but allow walls with small probability
        probs = (~wall_map).ravel().astype(jnp.float32)
        probs = probs + 1e-6  # Small epsilon for walls
        probs = probs / probs.sum()

        selected = jax.random.choice(
            rng_pos,
            a=GRID_SIZE * GRID_SIZE,
            shape=(total_positions,),
            replace=False,
            p=probs
        )

        # Convert indices to (x, y) positions
        def idx_to_pos(idx):
            y = idx // GRID_SIZE
            x = idx % GRID_SIZE
            return jnp.array([x, y], dtype=jnp.uint8)

        # Agent position
        agent_pos = idx_to_pos(selected[0])

        # Key positions (pad to MAX_KEYS)
        keys_pos_list = jax.vmap(idx_to_pos)(selected[1:1 + n_keys])
        keys_pos = jnp.zeros((MAX_KEYS, 2), dtype=jnp.uint8)
        keys_pos = keys_pos.at[:n_keys].set(keys_pos_list)

        # Chest positions (pad to MAX_CHESTS)
        chests_pos_list = jax.vmap(idx_to_pos)(selected[1 + n_keys:1 + n_keys + n_chests])
        chests_pos = jnp.zeros((MAX_CHESTS, 2), dtype=jnp.uint8)
        chests_pos = chests_pos.at[:n_chests].set(chests_pos_list)

        # Clear walls at all selected positions
        all_selected_y = selected // GRID_SIZE
        all_selected_x = selected % GRID_SIZE
        y_grid, x_grid = jnp.meshgrid(jnp.arange(GRID_SIZE), jnp.arange(GRID_SIZE), indexing='ij')
        positions_to_clear = jnp.any(
            (y_grid[:, :, None] == all_selected_y[None, None, :]) &
            (x_grid[:, :, None] == all_selected_x[None, None, :]),
            axis=2
        )
        wall_map = wall_map & (~positions_to_clear)

        return KeysChestsLevel(
            wall_map=wall_map,
            agent_pos=agent_pos,
            keys_pos=keys_pos,
            chests_pos=chests_pos,
            n_keys=n_keys,
            n_chests=n_chests,
        )

    return sample


# Generator factories for the two distributions
def make_non_distinguishing_generator():
    """Non-distinguishing: 3 keys, 10 chests (proxy-goal distribution)"""
    return make_keys_chests_level_generator(n_keys=3, n_chests=10)

def make_distinguishing_generator():
    """Distinguishing: 10 keys, 3 chests (true-goal distribution)"""
    return make_keys_chests_level_generator(n_keys=10, n_chests=3)


# ============================================================================
# Level Mutator (for ACCEL)
# ============================================================================

def make_keys_chests_mutator(num_edits: int = 12, alpha: float = 0.5, variant: str = "identity"):
    """
    Create level mutator with classification-preserving and transforming edits.

    Args:
        num_edits: Number of elementary edits per mutation
        alpha: Probability of switching to (10,3) vs (3,10) for transforming edits
        variant: ACCEL variant to use:
            - "identity": all edits are classification-preserving
            - "constant": n-1 preserving, then 1 biased transforming
            - "binomial": each edit is transforming w.p. 1/n
            - "unrestricted": n-1 preserving, then 1 unrestricted transforming
    """
    # Convert variant string to integer code at function definition time
    variant_codes = {"identity": 0, "constant": 1, "binomial": 2, "unrestricted": 3}
    _variant_code = variant_codes.get(variant)
    if _variant_code is None:
        raise ValueError(f"Unknown variant: {variant}")

    def mutate_single(rng_and_idx: Tuple[chex.PRNGKey, int], level: KeysChestsLevel) -> KeysChestsLevel:
        """Apply one elementary edit."""
        rng, edit_idx = rng_and_idx
        rng_type, rng_edit = jax.random.split(rng)

        # Determine if this edit should be preserving based on variant
        is_preserving = jax.lax.switch(
            _variant_code,
            [
                lambda: jnp.bool_(True),  # identity
                lambda: edit_idx < (num_edits - 1),  # constant
                lambda: jax.random.uniform(rng_type) >= (1.0 / num_edits),  # binomial
                lambda: edit_idx < (num_edits - 1),  # unrestricted
            ]
        )

        def preserving_edit(rng, lvl):
            """Classification-preserving: move wall/agent/key/chest"""
            rng_choice, rng_op = jax.random.split(rng)
            edit_type = jax.random.randint(rng_choice, (), 0, 4)
            return jax.lax.switch(
                edit_type,
                [
                    lambda r, l: _edit_wall(r, l),
                    lambda r, l: _edit_agent(r, l),
                    lambda r, l: _edit_key(r, l),
                    lambda r, l: _edit_chest(r, l),
                ],
                rng_op,
                lvl
            )

        def transforming_edit(rng, lvl):
            """Classification-transforming: change (k,c) counts and regenerate positions"""
            rng_choice, rng_sample = jax.random.split(rng)
            use_distinguishing = jax.random.uniform(rng_choice) < alpha
            new_n_keys = jnp.where(use_distinguishing, 10, 3)
            new_n_chests = jnp.where(use_distinguishing, 3, 10)

            # Get empty cells for resampling
            agent_x, agent_y = lvl.agent_pos[0], lvl.agent_pos[1]
            is_empty = ~lvl.wall_map
            is_not_agent = jnp.ones((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_).at[agent_y, agent_x].set(False)
            available = is_empty & is_not_agent

            rng_choice, rng_sample = jax.random.split(rng_sample)
            probs = available.ravel().astype(jnp.float32)
            probs = probs + 1e-6
            probs = probs / probs.sum()

            selected_13 = jax.random.choice(
                rng_choice, a=GRID_SIZE*GRID_SIZE, shape=(13,), replace=False, p=probs
            )

            def idx_to_pos(idx):
                y = idx // GRID_SIZE
                x = idx % GRID_SIZE
                return jnp.array([x, y], dtype=jnp.uint8)

            all_positions = jax.vmap(idx_to_pos)(selected_13)

            def set_distinguishing_positions():
                """10 keys, 3 chests"""
                keys = all_positions[:10]
                chests_padded = jnp.zeros((MAX_CHESTS, 2), dtype=jnp.uint8)
                chests_padded = chests_padded.at[:3].set(all_positions[10:13])
                return keys, chests_padded

            def set_non_distinguishing_positions():
                """3 keys, 10 chests"""
                keys_padded = jnp.zeros((MAX_KEYS, 2), dtype=jnp.uint8)
                keys_padded = keys_padded.at[:3].set(all_positions[:3])
                keys_padded = keys_padded.at[3:10].set(all_positions[3:10])
                chests = all_positions[3:13]
                return keys_padded, chests

            new_keys_pos, new_chests_pos = jax.lax.cond(
                use_distinguishing,
                set_distinguishing_positions,
                set_non_distinguishing_positions
            )

            return lvl.replace(
                n_keys=new_n_keys,
                n_chests=new_n_chests,
                keys_pos=new_keys_pos,
                chests_pos=new_chests_pos
            )

        return jax.lax.cond(is_preserving, preserving_edit, transforming_edit, rng_edit, level)

    def _edit_wall(rng, level):
        """Add or remove a wall (but never on occupied cells)."""
        rng_pos, rng_add = jax.random.split(rng)
        rng_y, rng_x = jax.random.split(rng_pos)
        y = jax.random.randint(rng_y, (), 0, GRID_SIZE)
        x = jax.random.randint(rng_x, (), 0, GRID_SIZE)

        at_agent = (level.agent_pos[0] == x) & (level.agent_pos[1] == y)

        key_indices = jnp.arange(MAX_KEYS)
        at_key = jnp.any((key_indices < level.n_keys) &
                        (level.keys_pos[:, 0] == x) & (level.keys_pos[:, 1] == y))

        chest_indices = jnp.arange(MAX_CHESTS)
        at_chest = jnp.any((chest_indices < level.n_chests) &
                          (level.chests_pos[:, 0] == x) & (level.chests_pos[:, 1] == y))

        is_occupied = at_agent | at_key | at_chest
        add_wall = jax.random.bernoulli(rng_add)
        new_wall_value = jnp.where(is_occupied, level.wall_map[y, x], add_wall)
        new_wall_map = level.wall_map.at[y, x].set(new_wall_value)
        return level.replace(wall_map=new_wall_map)

    def _edit_agent(rng, level):
        """Move agent to valid position."""
        rng_y, rng_x = jax.random.split(rng)
        y = jax.random.randint(rng_y, (), 0, GRID_SIZE)
        x = jax.random.randint(rng_x, (), 0, GRID_SIZE)

        is_not_wall = ~level.wall_map[y, x]
        key_indices = jnp.arange(MAX_KEYS)
        at_key = jnp.any((key_indices < level.n_keys) &
                        (level.keys_pos[:, 0] == x) & (level.keys_pos[:, 1] == y))
        chest_indices = jnp.arange(MAX_CHESTS)
        at_chest = jnp.any((chest_indices < level.n_chests) &
                          (level.chests_pos[:, 0] == x) & (level.chests_pos[:, 1] == y))

        is_valid = is_not_wall & (~at_key) & (~at_chest)
        new_pos = jnp.where(is_valid, jnp.array([x, y], dtype=jnp.uint8), level.agent_pos)
        return level.replace(agent_pos=new_pos)

    def _edit_key(rng, level):
        """Move a random key to valid position."""
        rng_idx, rng_pos = jax.random.split(rng)
        key_idx = jax.random.randint(rng_idx, (), 0, jnp.maximum(level.n_keys, 1))
        rng_y, rng_x = jax.random.split(rng_pos)
        y = jax.random.randint(rng_y, (), 0, GRID_SIZE)
        x = jax.random.randint(rng_x, (), 0, GRID_SIZE)

        is_not_wall = ~level.wall_map[y, x]
        at_agent = (level.agent_pos[0] == x) & (level.agent_pos[1] == y)

        key_indices = jnp.arange(MAX_KEYS)
        at_other_key = jnp.any(((key_indices < level.n_keys) & (key_indices != key_idx)) &
                               (level.keys_pos[:, 0] == x) & (level.keys_pos[:, 1] == y))
        chest_indices = jnp.arange(MAX_CHESTS)
        at_chest = jnp.any((chest_indices < level.n_chests) &
                          (level.chests_pos[:, 0] == x) & (level.chests_pos[:, 1] == y))

        is_valid = is_not_wall & (~at_agent) & (~at_other_key) & (~at_chest)
        new_pos = jnp.where(is_valid, jnp.array([x, y], dtype=jnp.uint8), level.keys_pos[key_idx])
        new_keys_pos = level.keys_pos.at[key_idx].set(new_pos)
        return level.replace(keys_pos=new_keys_pos)

    def _edit_chest(rng, level):
        """Move a random chest to valid position."""
        rng_idx, rng_pos = jax.random.split(rng)
        chest_idx = jax.random.randint(rng_idx, (), 0, jnp.maximum(level.n_chests, 1))
        rng_y, rng_x = jax.random.split(rng_pos)
        y = jax.random.randint(rng_y, (), 0, GRID_SIZE)
        x = jax.random.randint(rng_x, (), 0, GRID_SIZE)

        is_not_wall = ~level.wall_map[y, x]
        at_agent = (level.agent_pos[0] == x) & (level.agent_pos[1] == y)

        key_indices = jnp.arange(MAX_KEYS)
        at_key = jnp.any((key_indices < level.n_keys) &
                        (level.keys_pos[:, 0] == x) & (level.keys_pos[:, 1] == y))
        chest_indices = jnp.arange(MAX_CHESTS)
        at_other_chest = jnp.any(((chest_indices < level.n_chests) & (chest_indices != chest_idx)) &
                                 (level.chests_pos[:, 0] == x) & (level.chests_pos[:, 1] == y))

        is_valid = is_not_wall & (~at_agent) & (~at_key) & (~at_other_chest)
        new_pos = jnp.where(is_valid, jnp.array([x, y], dtype=jnp.uint8), level.chests_pos[chest_idx])
        new_chests_pos = level.chests_pos.at[chest_idx].set(new_pos)
        return level.replace(chests_pos=new_chests_pos)

    def mutate(rng: chex.PRNGKey, level: KeysChestsLevel, n_edits: int) -> KeysChestsLevel:
        """Apply multiple edits."""
        rngs = jax.random.split(rng, n_edits)
        edit_indices = jnp.arange(n_edits)

        def apply_edit(lvl, rng_and_idx):
            return mutate_single(rng_and_idx, lvl), None

        final_level, _ = jax.lax.scan(apply_edit, level, (rngs, edit_indices))
        return final_level

    return mutate


# ============================================================================
# Training State and Update Logic
# ============================================================================

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1


class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)


def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """GAE computation."""
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: KeysChestsState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple, Tuple]:
    """Sample trajectories using RNN policy."""
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (rng, train_state, init_hstate, init_obs, init_env_state, jnp.zeros(num_envs, dtype=bool)),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)
    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj


def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: KeysChestsState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Evaluate RNN policy."""
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)

        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)

    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_levels, dtype=bool),
         jnp.ones(num_levels, dtype=bool), jnp.zeros(num_levels, dtype=jnp.int32)),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths


def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool = True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """PPO update."""
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages

    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch

            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy
                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0).reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1).reshape(x.shape[0], n_minibatch, -1, *x.shape[2:]).swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)


# ============================================================================
# Actor-Critic Network
# ============================================================================

class ActorCritic(nn.Module):
    """Actor-Critic with LSTM for Keys and Chests (15×15×5 observation)."""
    action_dim: int

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        # Image embedding: 15×15×5 → features
        img_embed = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2), padding="VALID")(
            obs.image.astype(jnp.float32)
        )
        img_embed = nn.relu(img_embed)
        img_embed = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2), padding="VALID")(img_embed)
        img_embed = nn.relu(img_embed)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)

        # LSTM
        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))(
            (img_embed, dones), initial_carry=hidden
        )

        # Actor head
        actor_mean = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(
            jax.random.PRNGKey(0), (*batch_dims, 256)
        )


# ============================================================================
# Utilities
# ============================================================================

def setup_checkpointing(config: dict, train_state: TrainState, env, env_params) -> ocp.CheckpointManager:
    """Setup checkpointing."""
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)

    with open(os.path.join(overall_save_dir, "config.json"), "w+") as f:
        f.write(json.dumps(config, indent=True))

    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, "models"),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config["checkpoint_save_interval"],
            max_to_keep=config["max_number_of_checkpoints"],
        )
    )
    return checkpoint_manager


def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
    """Extract logging info from train state."""
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    s = jnp.maximum(idx.sum(), 1)

    return {
        "log": {
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": sampler["scores"].max(),
            "level_sampler/weighted_score": (sampler["scores"] * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": (sampler["scores"] * idx).sum() / s,
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        }
    }


def compute_score(config, dones, values, max_returns, advantages, levels=None):
    """Compute level scores for prioritization."""
    if config["score_function"] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config["score_function"] == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")


# ============================================================================
# Main Training Function
# ============================================================================

def main(config=None, project="JAXUED_KeysChests"):
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")

    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config

    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    # Setup environment
    env = KeysAndChestsEnv()
    env_params = env.default_params
    sample_random_level = make_non_distinguishing_generator()
    env_renderer = KeysChestsRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    mutate_level = make_keys_chests_mutator(
        config["num_edits"],
        alpha=0.5,
        variant=config["accel_variant"]
    )

    # Level sampler
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config["top_k"]},
        duplicate_check=config["buffer_duplicate_check"],
    )

    def log_eval(stats, train_state_info, highest_scoring_level=None, eval_states=None):
        """Log evaluation metrics and visualizations to wandb."""
        print(f"Logging update: {stats['update_count']}")

        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats["time_delta"],
        }

        # Evaluation performance
        returns = stats["eval_returns"]
        log_dict.update({f"return/env_{i}": float(ret) for i, ret in enumerate(returns)})
        log_dict["return/mean"] = float(returns.mean())
        log_dict["return/non_distinguishing_mean"] = float(returns[:4].mean())
        log_dict["return/distinguishing_mean"] = float(returns[4:].mean())
        log_dict["eval_ep_lengths/mean"] = float(stats["eval_ep_lengths"].mean())

        # Level sampler metrics
        log_dict.update(train_state_info["log"])

        # Visualizations
        if highest_scoring_level is not None:
            level_img = env_renderer.render_level(highest_scoring_level, env_params)
            log_dict["images/highest_scoring_level"] = wandb.Image(level_img)

        # Animation of evaluation trajectory (first env)
        if eval_states is not None and config.get("log_animations", False):
            try:
                # Extract first trajectory
                first_traj_states = jax.tree_map(lambda x: x[:, 0], eval_states)
                frames = render_trajectory_animation(env_renderer, first_traj_states, env_params, max_frames=64)
                # Transpose to (T, C, H, W) for wandb.Video
                frames = np.transpose(frames, (0, 3, 1, 2))
                log_dict["animations/eval_trajectory"] = wandb.Video(frames, fps=8)
            except Exception as e:
                print(f"Failed to render animation: {e}")

        wandb.log(log_dict)

    @jax.jit
    def create_train_state(rng) -> TrainState:
        def linear_schedule(count):
            frac = 1.0 - (count // (config["num_minibatches"] * config["epoch_ppo"])) / config["num_updates"]
            return config["lr"] * frac

        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )

        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
        pholder_level_batch = jax.tree_util.tree_map(
            lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level
        )

        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """Main training loop step."""
        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            sampler = train_state.sampler
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params
            )

            ((rng, train_state, hstate, last_obs, last_env_state, last_value),
             (obs, actions, rewards, dones, log_probs, values, info)) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs, init_env_state, config["num_train_envs"], config["num_steps"],
            )

            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages, levels=new_levels)
            sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores, {"max_return": max_returns})

            (rng, train_state), losses = update_actor_critic_rnn(
                rng, train_state, ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"], config["num_steps"], config["num_minibatches"],
                config["epoch_ppo"], config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )

            train_state = train_state.replace(
                sampler=sampler, update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1, dr_last_level_batch=new_levels,
            )
            return (rng, train_state), {"losses": jax.tree_util.tree_map(lambda x: x.mean(), losses)}

        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            sampler = train_state.sampler
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(
                sampler, rng_levels, config["num_train_envs"]
            )
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params
            )

            ((rng, train_state, hstate, last_obs, last_env_state, last_value),
             (obs, actions, rewards, dones, log_probs, values, info)) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs, init_env_state, config["num_train_envs"], config["num_steps"],
            )

            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = jnp.maximum(
                level_sampler.get_levels_extra(sampler, level_inds)["max_return"],
                compute_max_returns(dones, rewards)
            )
            scores = compute_score(config, dones, values, max_returns, advantages, levels=levels)
            sampler = level_sampler.update_batch(sampler, level_inds, scores, {"max_return": max_returns})

            (rng, train_state), losses = update_actor_critic_rnn(
                rng, train_state, ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"], config["num_steps"], config["num_minibatches"],
                config["epoch_ppo"], config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
                update_grad=True,
            )

            train_state = train_state.replace(
                sampler=sampler, update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1, replay_last_level_batch=levels,
            )
            return (rng, train_state), {"losses": jax.tree_util.tree_map(lambda x: x.mean(), losses)}

        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            sampler = train_state.sampler
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)

            parent_levels = train_state.replay_last_level_batch
            child_levels = jax.vmap(mutate_level, (0, 0, None))(
                jax.random.split(rng_mutate, config["num_train_envs"]), parent_levels, config["num_edits"]
            )
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
                jax.random.split(rng_reset, config["num_train_envs"]), child_levels, env_params
            )

            ((rng, train_state, hstate, last_obs, last_env_state, last_value),
             (obs, actions, rewards, dones, log_probs, values, info)) = sample_trajectories_rnn(
                rng, env, env_params, train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs, init_env_state, config["num_train_envs"], config["num_steps"],
            )

            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            max_returns = compute_max_returns(dones, rewards)
            scores = compute_score(config, dones, values, max_returns, advantages, levels=child_levels)
            sampler, _ = level_sampler.insert_batch(sampler, child_levels, scores, {"max_return": max_returns})

            (rng, train_state), losses = update_actor_critic_rnn(
                rng, train_state, ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"], config["num_steps"], config["num_minibatches"],
                config["epoch_ppo"], config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )

            train_state = train_state.replace(
                sampler=sampler, update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1, mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), {"losses": jax.tree_util.tree_map(lambda x: x.mean(), losses)}

        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)

        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)

        return jax.lax.switch(branch, [on_new_levels, on_replay_levels, on_mutate_levels], rng, train_state)

    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """Evaluate on both non-distinguishing and distinguishing levels."""
        rng_non_dist, rng_dist = jax.random.split(rng)

        non_dist_gen = make_non_distinguishing_generator()
        dist_gen = make_distinguishing_generator()

        non_dist_levels = jax.vmap(non_dist_gen)(jax.random.split(rng_non_dist, 4))
        dist_levels = jax.vmap(dist_gen)(jax.random.split(rng_dist, 4))

        all_levels = jax.tree_map(lambda x, y: jnp.concatenate([x, y]), non_dist_levels, dist_levels)

        rng, rng_reset = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, 8), all_levels, env_params
        )

        states, rewards, episode_lengths = evaluate_rnn(
            rng, env, env_params, train_state,
            ActorCritic.initialize_carry((8,)), init_obs, init_env_state,
            env_params.max_steps_in_episode,
        )

        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths

    @jax.jit
    def train_and_eval_step(runner_state, _):
        """Train for eval_freq steps, then evaluate."""
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])

        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(
            jax.random.split(rng_eval, config["eval_num_attempts"]), train_state
        )

        eval_returns = cum_rewards.mean(axis=0)

        metrics["update_count"] = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        metrics["eval_returns"] = eval_returns
        metrics["eval_ep_lengths"] = episode_lengths.mean(axis=0)
        # Store first attempt's states for animation (shape: T, num_levels, ...)
        metrics["eval_states"] = jax.tree_map(lambda x: x[0], states)

        return (rng, train_state), metrics

    # Initialize
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)

    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)

    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)

    # Training loop
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics["time_delta"] = curr_time - start_time

        # Get highest scoring level for visualization
        train_state = runner_state[1]
        highest_idx = jnp.argmax(train_state.sampler["scores"])
        highest_level = jax.tree_map(lambda x: x[highest_idx], level_sampler.get_levels(train_state.sampler))

        log_eval(metrics, train_state_to_log_dict(train_state, level_sampler),
                 highest_scoring_level=highest_level, eval_states=metrics.get("eval_states"))

        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()

    return runner_state[1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JAXUED_KeysChests")
    parser.add_argument("--run_name", type=str, default="keys_chests_accel")
    parser.add_argument("--seed", type=int, default=0)

    # Checkpointing
    parser.add_argument("--checkpoint_save_interval", type=int, default=0)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)

    # Eval
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--log_animations", action="store_true", default=False)

    group = parser.add_argument_group("Training params")

    # PPO (Table G.1 hyperparameters)
    group.add_argument("--lr", type=float, default=5e-5)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    group.add_argument("--num_updates", type=int, default=30000)
    group.add_argument("--num_steps", type=int, default=128)
    group.add_argument("--num_train_envs", type=int, default=256)
    group.add_argument("--num_minibatches", type=int, default=4)
    group.add_argument("--gamma", type=float, default=0.999)
    group.add_argument("--epoch_ppo", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.1)
    group.add_argument("--gae_lambda", type=float, default=0.95)
    group.add_argument("--entropy_coeff", type=float, default=1e-2)
    group.add_argument("--critic_coeff", type=float, default=0.5)

    # PLR/ACCEL
    group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl"])
    group.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--level_buffer_capacity", type=int, default=4096)
    group.add_argument("--replay_prob", type=float, default=0.5)
    group.add_argument("--staleness_coeff", type=float, default=0.1)
    group.add_argument("--temperature", type=float, default=0.1)
    group.add_argument("--top_k", type=int, default=4)
    group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    group.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    group.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)

    # ACCEL
    group.add_argument("--use_accel", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--num_edits", type=int, default=12)
    group.add_argument("--accel_variant", type=str, default="identity",
                       choices=["identity", "constant", "binomial", "unrestricted"])

    config = vars(parser.parse_args())

    wandb.login()
    main(config, project=config["project"])
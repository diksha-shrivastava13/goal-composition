from __future__ import annotations
from typing import Any, Tuple
from chex import PRNGKey
import chex
from jaxued.environments.underspecified_env import EnvParams, EnvState, Level, Observation, UnderspecifiedEnv
from flax import struct

@struct.dataclass
class WrappedEnvState(EnvState):
    _env_state: EnvState
    def __getattr__(self, name):
        """
        Returns an attribute with ``name``, unless ``name`` starts with an underscore.
        :param name: attribute name
        :return: attribute
        """
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self._env_state, name)


@struct.dataclass
class WrappedLevel(Level):
    _level: Level
    def __getattr__(self, name):
        """
        Returns an attribute with ``name``, unless ``name`` starts with an underscore.
        :param name: attribute name
        :return: attribute
        """
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self._level, name)


class UnderspecifiedEnvWrapper(UnderspecifiedEnv):
    def __init__(self, env: UnderspecifiedEnv) -> None:
        super().__init__()
        self._env = env

    def __getattr__(self, name):
        """
        Returns an attribute with ``name``, unless ``name`` starts with an underscore.
        :param name: attribute name
        :return: attribute
        """
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self._env, name)

    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params

    def step_env(self, rng: PRNGKey, state: WrappedEnvState, action: int | float, params: EnvParams) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        obs, new_state, reward, done, info = self._env.step_env(rng, state, action, params)
        return obs, new_state, reward, done, info

    def reset_env_to_level(self, rng: PRNGKey, level: Level, params: EnvParams) -> Tuple[Observation, EnvState]:
        obs, state = self._env.reset_env_to_level(rng, level, params)
        return obs, WrappedEnvState(state)

    def action_space(self, params: EnvParams) -> Any:
        return self._env.action_space(params)

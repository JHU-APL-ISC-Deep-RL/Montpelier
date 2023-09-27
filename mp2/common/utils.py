import gym
import torch
import numpy as np
from copy import deepcopy
from scipy.signal import lfilter
from mp2.common.networks import AtariNetwork, MLP, CategoricalMLP, GaussianMLP, BetaMLP
from typing import Dict


def get_env_object(config: Dict):
    """
    Helper function that returns environment object.  Can include more environments as they become available.  While
    the following does not explicitly require that the environment inherit from gym.Env, any environment that does
    follow the OpenAI gym format should be compatible.
    """
    env_config = deepcopy(config)
    env_config['clip_rewards'] = 0
    env_name = env_config.pop('name', None)
    if config['type'].lower() == 'gym':  # Load OpenAI gym environment
        if 'scale' in config:
            return ScaledGym(gym.make(config['name']), config['scale'])
        else:
            return gym.make(config['name'])
    elif config['type'] == 'meta-arcade':
        from mp2.common.atari_wrappers import make_atari, wrap_deepmind
        env = gym.make("MetaArcade-v0", config=config['config_path'], headless=True)
        return PyTorchAtari(wrap_deepmind(env, **env_config), (0, 3, 1, 2))
    elif config['type'].lower() == 'metaworld':
        config.setdefault('task_name', None)
        config.setdefault('stop_on_success', False)
        config.setdefault('zero_on_success', False)
        assert not config['stop_on_success'] or not config['zero_on_success'], 'Just use one environment option.'
        env_collection = load_metaworld(config['benchmark'], config['task_name'])
        single_task = config.get('single', False)
        if single_task:
            train = config.get('train', True)
            super_index = config['task_ind'][0]
            overall_index = super_index*50 + config['task_ind'][1]
            if train:
                env = list(env_collection.train_classes.values())[super_index]()
                env.set_task(env_collection.train_tasks[overall_index])
            else:
                env = list(env_collection.test_classes.values())[super_index]()
                env.set_task(env_collection.test_tasks[overall_index])
            env = ReducedLogging(env)
            return env
        else:
            return env_collection
    else:
        from mp2.common.atari_wrappers import make_atari, wrap_deepmind
        return PyTorchAtari(wrap_deepmind(make_atari(env_name), **env_config), (0, 3, 1, 2))


def get_meta_arcade_env_by_name(name):
    # until we have a config with many environments, using this...
    from mp2.common.atari_wrappers import wrap_deepmind
    env = gym.make("MetaArcade-v0", config=name, headless=True)
    return PyTorchAtari(wrap_deepmind(env, episode_life=False), (0, 3, 1, 2))


def load_metaworld(benchmark, task_name=None, r_seed=1337):
    import metaworld
    if benchmark[-1] == '1':
        assert task_name is not None, "Need a task name for one-task benchmark."
    if benchmark.lower() == 'ml1':
        environment_object = metaworld.ML1(task_name, seed=r_seed)
    elif benchmark.lower() == 'ml10':
        environment_object = metaworld.ML10(seed=r_seed)
    elif benchmark.lower() == 'ml45':
        environment_object = metaworld.ML45(seed=r_seed)
    elif benchmark.lower() == 'mt1':
        environment_object = metaworld.MT1(task_name, seed=r_seed)
    elif benchmark.lower() == 'mt10':
        environment_object = metaworld.MT10(seed=r_seed)
    elif benchmark.lower() == 'mt50':
        environment_object = metaworld.MT50(seed=r_seed)
    else:
        raise NotImplementedError('MetaWorld benchmark not recognized.')
    return environment_object


def get_network_object(config: Dict) -> torch.nn.Module:
    """
    Helper function that returns network object.  Can include more networks as they become available.
    The network is assumed to satisfy the following requirements:
    - It should have a config attribute that is a dict and has the following keys: clip_range, value_coeff,
      entropy_coeff, max_grad_norm (optional).  This dict is provided to the class by the PPO class, and default
      values for these quantities are given.  However, the best values for them will vary by problem and should
      be passed into PPO via its configuration dict.
    - It must have the following attributes: policies, values. That is, it must be configured for
      actor-critic learning.  If recurrent, the network should additionally have batch_size
      and trace_length attributes.
    """
    config.setdefault('humans_use_net', False)
    if 'network_type' not in config:
        raise ValueError('network_type missing from config')
    if config['network_type'].lower() == 'atari':
        return AtariNetwork(config)
    if config['network_type'].lower() == 'mlp':
        return MLP(config)
    elif config['network_type'].lower() == 'mlp_categorical':
        return CategoricalMLP(config)
    elif config['network_type'].lower() == 'mlp_gaussian':
        return GaussianMLP(config)
    elif config['network_type'].lower() == 'mlp_beta':
        return BetaMLP(config)
    else:
        raise ValueError('network_type not recognized.')


def get_sampler(config: Dict, deterministic=False):
    if config['sampler'] == 'categorical':
        from .samplers import CategoricalSampler
        return CategoricalSampler(config, deterministic)
    elif config['sampler'] == 'gaussian':
        from .samplers import GaussianSampler
        return GaussianSampler(config, deterministic)
    elif config['sampler'] == 'beta':
        from .samplers import BetaSampler
        return BetaSampler(config, deterministic)


def bind(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    Function from Alex Martelli, Nick T on stack overflow.
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


def discount(x, gamma):
    """  Computes discounted quantity (used for discounted future reward). """
    return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class PyTorchAtari(gym.Wrapper):
    def __init__(self, env, dim_order: tuple):
        """  Wrapper to appropriately re-shape arrays for PyTorch processing  """
        gym.Wrapper.__init__(self, env)
        self.dim_order = dim_order

        # construct transposed obs space
        shape = env.observation_space.shape
        low = env.observation_space.low
        high = env.observation_space.high
        if np.isscalar(low):
            # use shape as the basis for the observation space
            new_shape = []
            for idx in dim_order:
                new_shape.append(shape[idx])
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=tuple(new_shape))
        else:
            # use low and high to infer the shape
            new_low = np.transpose(low, (0, 3, 1, 2))
            new_high = np.transpose(high, (0, 3, 1, 2))
            self.observation_space = gym.spaces.Box(low=new_low, high=new_high)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return np.transpose(obs, (0, 3, 1, 2))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.transpose(obs, (0, 3, 1, 2)), reward, done, info


class ScaledGym(gym.Wrapper):
    def __init__(self, env, reward_scale):
        """
        Scales returned rewards from gym environment
        :param env: (Gym Environment); the environment to wrap
        :param reward_scale: (float); multiplier for reward
        """
        gym.Wrapper.__init__(self, env)
        self.reward_scale = reward_scale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward*self.reward_scale, done, info


class ReducedLogging(gym.Wrapper):
    def __init__(self, base_env):
        """  Reduces info to be logged (for Meta-World)  """
        gym.Wrapper.__init__(self, base_env)
        self.success_ever = 0

    def reset(self):
        self.success_ever = 0
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.success_ever += info['success']
        info = {'success': [np.sign(self.success_ever)]}  # just log this for now
        return obs, reward, done, info


class StopOnSuccess(gym.Wrapper):
    def __init__(self, base_env):
        """  Takes a MetaWorld environment and ends its episodes upon success  """
        gym.Wrapper.__init__(self, base_env)
        self.success_ever = 0

    def reset(self):
        self.success_ever = 0
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.success_ever += info['success']
        if info['success'] > 0:
            done = True
        info = {'success': [np.sign(self.success_ever)]}  # just log this for now
        return obs, reward, done, info


class ZeroOnSuccess(gym.Wrapper):
    def __init__(self, base_env):
        """  Takes a MetaWorld environment and ensures all rewards are 0 after success occurs """
        gym.Wrapper.__init__(self, base_env)
        self.success_ever = 0

    def reset(self):
        self.success_ever = 0
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.success_ever > 0:  # keep first terminal reward
            reward = 0.
        self.success_ever += info['success']
        info = {'success': [np.sign(self.success_ever)]}  # just log this for now
        return obs, reward, done, info


class NormalizedReward(gym.Wrapper):

    def __init__(self, base_env, reward_norm):
        """  Takes an environment and applies a scaling to the rewards it returns  """
        gym.Wrapper.__init__(self, base_env)
        self.reward_norm = reward_norm

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward * self.reward_norm, done, info


class OneHotAppended(gym.Wrapper):

    def __init__(self, base_env, task_index, total_tasks):
        """  Takes an environment and appends a one-hot task encoding to observations  """
        gym.Wrapper.__init__(self, base_env)
        self.one_hot = np.zeros(total_tasks)
        self.one_hot[task_index] = 1.

    def reset(self, **kwargs):
        obs = self.env.reset()
        return np.concatenate((obs, self.one_hot))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.concatenate((obs, self.one_hot)), reward, done, info

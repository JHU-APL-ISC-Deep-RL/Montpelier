import os
import json
import argparse
import pickle
import torch
import numpy as np
from mpi4py import MPI
from mp2.common.utils import get_env_object, ReducedLogging, StopOnSuccess, ZeroOnSuccess, \
    NormalizedReward, OneHotAppended
from mp2.common.samplers import compute_entropy
from mp2.common.mpi_data_utils import print_now
from mp2.base.rl.policy_optimizer import PolicyOptimizer


# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class MTPolicyOptimizer(PolicyOptimizer):

    def __init__(self, config):
        """  Construct MultiTask PolicyOptimizer agent  """
        super().__init__(config)
        self.env_collection = None
        self.task_indices = np.array([])
        self.num_tasks = 0
        self.task_scales = []
        self.current_task = -1
        self.load_environments()

    def process_config(self):
        super().process_config()
        self.config.setdefault('task_norm_file', None)
        self.config.setdefault('use_test_set', False)

    def load_environments(self):
        """  Initialize environment object at the beginning of training/testing  """
        if self.config['environment']['type'].lower() == 'metaworld':
            if self.config['task_norm_file'] is not None:
                task_scales = self.load_task_normalizations()
                for scale in task_scales:
                    self.task_scales += [scale] * 50
            self.env_collection = get_env_object(self.config['environment'])
            if self.config['use_test_set']:
                print_now(MPI.COMM_WORLD, 'Using test tasks...')
                self.task_indices = np.arange(len(self.env_collection.test_tasks))
            else:
                print_now(MPI.COMM_WORLD, 'Using training tasks...')
                self.task_indices = np.arange(len(self.env_collection.train_tasks))
            self.num_tasks = int(len(self.task_indices) / 50)
        else:
            raise ValueError('Currently only tested for Meta-World.')
            # for i in range(len(self.config['environment'])):
            #    self.envs.append(get_env_object(self.config['environment'][i]))
            #    self.obs[i] = self.envs[i].reset()

    def load_task_normalizations(self):
        """  Loads task normalizations from file  """
        with open(self.config['task_norm_file'], 'rb') as task_norm_file:
            data = pickle.load(task_norm_file)
        return data['task_norms']

    def initialize_env(self):
        """  Initialize environment object  """
        self.current_task = np.random.choice(self.task_indices)
        if self.config['environment']['type'].lower() == 'metaworld':
            class_index = self.current_task // 50
            if self.config['use_test_set']:
                env = list(self.env_collection.test_classes.values())[class_index]()
                env.set_task(self.env_collection.test_tasks[self.current_task])
            else:
                env = list(self.env_collection.train_classes.values())[class_index]()
                env.set_task(self.env_collection.train_tasks[self.current_task])
            if self.config['environment']['stop_on_success']:
                env = StopOnSuccess(env)
            elif self.config['environment']['zero_on_success']:
                env = ZeroOnSuccess(env)
            else:
                env = ReducedLogging(env)
            if self.config['environment']['normalize_reward'] > 0:
                env = NormalizedReward(env, self.config['environment']['normalize_reward'])
            if self.config['environment']['one_hot']:
                env = OneHotAppended(env, class_index, self.num_tasks)
            self.env = env
        else:
            raise NotImplementedError('Environment type not yet implemented.')

    def reset_env(self, random_seed):
        """  Resets for next environment instance  """
        self.initialize_env()
        super().reset_env(random_seed)

    def process_trajectory(self, trajectory_buffer, trajectory_info):
        """  Processes a completed trajectory, storing required data in buffer and return ep  """
        if self.mode == 'train':
            max_entropy_term = self.compute_max_entropy_term(trajectory_buffer)
            q_values = self.compute_target_values(trajectory_buffer[:, 2] + max_entropy_term)
            task_ids = (self.current_task // 50) * np.ones(q_values.shape)  # using Meta-World structure here
            self.buffer.update(trajectory_buffer, q_values, max_ent=max_entropy_term, task_ids=task_ids)
        if self.buffer.dones[-1]:
            entropies = [compute_entropy(self.buffer.current_episode[i, 3])
                         for i in range(self.buffer.current_episode.shape[0])]
            return {'episode_reward': np.sum(self.buffer.current_episode[:, 2]),
                    'episode_length': self.buffer.current_episode.shape[0],
                    'episode_mean_value': np.mean(self.buffer.current_episode[:, 5]),
                    'episode_entropy': np.mean(entropies),
                    **{k: self.flatten_list(v)[-1] for k, v in trajectory_info.items()}}
        else:
            return {}


if __name__ == '__main__':
    """  Runs MTPolicyOptimizer training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    parser.add_argument('--trpo', help='whether to force trpo update', required=False, type=int, default=0)
    parser.add_argument('--use_prior', help='use prior training', required=False, type=int, default=0)
    in_args = parser.parse_args()
    with open(os.path.join(os.getcwd(), in_args.config), 'r') as f1:
        config1 = json.load(f1)
    config1['seed'] = in_args.seed
    config1['use_prior_nets'] = bool(in_args.use_prior)
    if 'trpo' not in config1:
        config1['trpo'] = bool(in_args.trpo)
    if in_args.mode.lower() == 'train':
        mtpo_object = MTPolicyOptimizer(config1)
        mtpo_object.train()
    else:
        config1['use_prior_nets'] = True
        mtpo_object = MTPolicyOptimizer(config1)
        mtpo_object.test()

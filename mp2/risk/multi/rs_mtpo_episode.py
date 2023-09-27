import os
import json
import argparse
import sys

import torch
import numpy as np
from mpi4py import MPI
from mp2.base.multi.mtpo_episode import MTPolicyOptimizer
from mp2.common.mpi_data_utils import mpi_gather_objects
from mp2.risk.dist import load_weight_coefficient_function

# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class RiskSensitiveMTPolicyOptimizer(MTPolicyOptimizer):

    def __init__(self, config, weight_function=None):
        super().__init__(config)  # todo: possibly add support for utility functions
        self.weight_coefficient_function = load_weight_coefficient_function(self.config['weight'], weight_function)

    def process_config(self):
        super().process_config()
        self.config['weight'].setdefault('outcomes', 'episodes')

    def compute_weights(self):
        """  Updates task weightings based on current batch of data  """
        all_rewards = mpi_gather_objects(MPI.COMM_WORLD, self.buffer.rewards)
        all_dones = mpi_gather_objects(MPI.COMM_WORLD, self.buffer.dones)
        all_tasks = mpi_gather_objects(MPI.COMM_WORLD, self.buffer.task_ids)

        if self.id == 0:  # only need to compute weights on one worker
            # Sum episode rewards:
            episode_rewards = []
            task_rewards = [[] for _ in range(self.num_tasks)]
            all_indices = []
            current_ind = 0
            for i in range(len(all_rewards)):
                terminal_ind = np.where(all_dones[i])[0]
                episode_ind = np.ones(all_rewards[i].shape)
                for j in range(terminal_ind.shape[0]):
                    start_ind = 0 if j == 0 else terminal_ind[j - 1] + 1
                    stop_ind = terminal_ind[j] + 1
                    episode_rewards.append(np.sum(all_rewards[i][start_ind:stop_ind]))
                    task_rewards[all_tasks[i][start_ind]].append(np.sum(all_rewards[i][start_ind:stop_ind]))
                    episode_ind[start_ind:stop_ind] *= current_ind
                    current_ind += 1
                all_indices.append(episode_ind.astype(int))
            # Compute weights:
            if self.config['weight']['outcomes'] == 'episodes':  # consider over episode outcome distribution
                episode_weights = self.weight_coefficient_function(episode_rewards)
                all_weights = [episode_weights[all_indices[i]] for i in range(len(all_indices))]
            else:  # consider over task outcome distribution
                mean_task_rewards = [np.mean(item) for item in task_rewards if len(item) > 0]
                raw_task_weights = self.weight_coefficient_function(mean_task_rewards)
                task_weights = np.zeros(self.num_tasks)
                weight_tracker = 0
                for i in range(len(task_rewards)):
                    if len(task_rewards[i]) > 0:
                        task_weights[i] = raw_task_weights[weight_tracker]
                        weight_tracker += 1
                all_weights = [task_weights[all_tasks[i]] for i in range(len(all_tasks))]
        else:
            episode_rewards, all_weights = [], []
        # Distribute back to workers:
        all_weights = MPI.COMM_WORLD.bcast(all_weights, root=0)
        self.buffer.weights = all_weights[self.id]

    def update_network(self, indices, starts, stops):
        """  Updates the networks based on processing from all workers  """
        self.compute_weights()
        return super().update_network(indices, starts, stops)

    def compute_policy_loss(self, observations, actions, advantages, old_log_probs, old_policies, mb=None):
        """  Compute policy loss, entropy, kld  """
        pi_loss, entropy, kld = super().compute_policy_loss(observations, actions, advantages, old_log_probs,
                                                            old_policies, mb)
        weights = torch.from_numpy(self.buffer.weights.astype(float)).float()
        pi_loss = pi_loss * weights
        return pi_loss, entropy, kld


if __name__ == '__main__':
    """  Runs RiskSensitiveMTPolicyOptimizer training or testing for a given input configuration file  """
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
        rsmtpo_object = RiskSensitiveMTPolicyOptimizer(config1)
        rsmtpo_object.train()
    else:
        config1['use_prior_nets'] = True
        rsmtpo_object = RiskSensitiveMTPolicyOptimizer(config1)
        rsmtpo_object.test()

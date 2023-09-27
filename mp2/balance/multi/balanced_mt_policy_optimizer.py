import os
import json
import argparse
import torch
import numpy as np
from mpi4py import MPI
from mp2.base.multi.mt_policy_optimizer import MTPolicyOptimizer
from mp2.common.mpi_data_utils import mpi_gather_objects, sync_weights

# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class BalancedMTPolicyOptimizer(MTPolicyOptimizer):

    def __init__(self, config):
        """  Construct Balanced MultiTask PolicyOptimizer agent  """
        super().__init__(config)
        self.tw_param = None
        self.tw_optimizer = None
        self.task_weights = None
        self.batch_groups = []
        self.batch_start = 0

    def process_config(self):
        """  Processes configuration, filling in missing values as appropriate  """
        super().process_config()
        self.config.setdefault('tw_lr', .001)  # learning rate for task weights
        self.config.setdefault('max_tw', None)  # maximum task weight (to be added to 1 in policy optimization)
        if self.config['max_tw'] is not None and self.config['max_tw'] < 0:
            self.config['max_tw'] = None

    def initialize_environments(self):
        """  Initialize environment object at the beginning of training / testing  """
        super().initialize_environments()
        if self.config['environment']['type'].lower() == 'metaworld':
            self.config['task_groups'] = int(self.num_tasks / 50)
            self.batch_groups = list(range(self.config['task_groups'])) * self.num_workers
            self.batch_start = 0
        else:
            raise ValueError('Currently only tested for Meta-World.')

    def sample_tasks(self):
        """  Sample tasks evenly among task groups  """
        task_groups = self.batch_groups[self.batch_start: self.batch_start + self.config['tasks_per_batch']]
        if self.batch_start + self.config['tasks_per_batch'] >= len(self.batch_groups):
            task_groups += self.batch_groups[:self.batch_start+self.config['tasks_per_batch']-len(self.batch_groups)]
        task_list = 0
        if self.id == 0:
            task_indices = np.array(task_groups)*50 + np.random.randint(0, 50, self.config['tasks_per_batch'])
            task_list = task_indices.tolist()
        task_list = MPI.COMM_WORLD.bcast(task_list, root=0)
        self.batch_start += self.config['tasks_per_batch']
        return task_list

    def initialize_networks(self):
        """  Initialize network objects  """
        last_checkpoint = super().initialize_networks()
        self.tw_param = torch.autograd.Variable(torch.zeros(self.config['task_groups']), requires_grad=True)
        return last_checkpoint

    def initialize_optimizers(self):
        super().initialize_optimizers()
        if self.id == 0:
            self.tw_optimizer = torch.optim.Adam(params=[self.tw_param], lr=self.config['tw_lr'])

    def update_task_weights(self):
        """  Updates task weightings based on current batch of data  """
        all_buffers = self.flatten_list(mpi_gather_objects(MPI.COMM_WORLD, self.buffers))
        if self.id == 0:
            group_rewards = [[] for _ in range(self.config['task_groups'])]
            for buffer in all_buffers:
                terminal_ind = np.where(buffer.dones)[0]
                for j in range(terminal_ind.shape[0]):
                    start_ind = 0 if j == 0 else terminal_ind[j - 1] + 1
                    stop_ind = terminal_ind[j] + 1
                    group_rewards[buffer.task_groups[0]].append(sum(buffer.rewards[start_ind:stop_ind]))
            mean_group_rewards = [np.mean(group) if len(group) > 0 else None for group in group_rewards]
            nonempty_mean = np.mean(list(filter(lambda x: x, mean_group_rewards)))
            tw_loss_scale = np.array([item - nonempty_mean if item is not None else 0. for item in mean_group_rewards])
            tw_loss_scale /= np.std(tw_loss_scale)
            tw_loss_scale = torch.from_numpy(tw_loss_scale.astype(float)).float()
            self.tw_optimizer.zero_grad()
            tw_loss = torch.sum(tw_loss_scale * self.tw_param)
            tw_loss.backward()
            self.tw_optimizer.step()
        sync_weights(MPI.COMM_WORLD, self.tw_param)
        self.task_weights = np.clip(1 + self.tw_param.detach().numpy(), 0, self.config['max_tw'])

    def update_network(self, indices, starts, stops):
        """  Updates the networks based on processing from all workers  """
        self.update_task_weights()
        output1 = super().update_network(indices, starts, stops)
        for i in range(self.task_weights.shape[0]):
            output1['task ' + str(i) + ' weight'] = [self.task_weights[i]]  # for logging purposes
        return output1

    def compute_pi_gradients(self):
        """  Computes gradients of policy loss for each task  """
        pi_grads, entropies, klds = [], [], []
        for buffer in self.buffers:
            self.pi_optimizer.zero_grad()
            observations = torch.from_numpy(np.vstack(buffer.observations)).float()
            advantages = torch.from_numpy(buffer.advantages.astype(float)).float()
            actions = buffer.actions
            log_probs = torch.from_numpy(buffer.log_probs.astype(float)).float()
            policies = buffer.policies
            task_weight = self.task_weights[buffer.task_groups[0]]
            pi_loss, entropy, kld = self.compute_policy_loss(observations, actions, advantages, log_probs, policies)
            pi_loss = torch.mul(torch.mean(pi_loss, dim=0), task_weight)
            pi_loss.backward()
            entropies.append(entropy.item())
            klds.append(kld.item())
            pi_grads.append(self.store_gradients(self.pi_network))
        return pi_grads, entropies, klds


if __name__ == '__main__':
    """  Runs BalancedMTPolicyOptimizer training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    in_args = parser.parse_args()
    with open(os.path.join(os.getcwd(), in_args.config), 'r') as f1:
        config1 = json.load(f1)
    config1['seed'] = in_args.seed
    if in_args.mode.lower() == 'train':
        bmtpo_object = BalancedMTPolicyOptimizer(config1)
        bmtpo_object.train()
    else:
        config1['use_prior_nets'] = True
        bmtpo_object = BalancedMTPolicyOptimizer(config1)
        bmtpo_object.test()

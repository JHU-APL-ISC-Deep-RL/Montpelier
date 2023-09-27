import argparse
import os
import json
import torch
from mpi4py import MPI
from mp2.common.utils import get_sampler
from mp2.common.garage_utils import zero_optim_grads, update_module_params
from mp2.base.meta.maml import MAML
# import sys  # todo rm


# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class ProMP(MAML):
    """
    Object for MPI-parallelized Proximal Meta-Policy Search (ProMP)  Note that, as written, this code requires
    at least as many workers as tasks per meta-update.
    """
    def __init__(self, config):
        """  Constructs ProMP object  """
        super().__init__(config)
        assert self.num_workers >= self.config['tasks_per_update'], 'Not enough workers'

    def process_config(self):
        """  Processes input configuration  """
        if 'inner' not in self.config:
            self.config['inner'] = {'surrogate': True, 'clip': -1, 'kld_grad': False}  # ppo without clip
        if 'outer' not in self.config:
            self.config['outer'] = {'surrogate': True, 'clip': 0.2, 'kld_grad': True,
                                    'train_pi_iter': 5, 'eta': 0.01}  # promp
        self.config.setdefault('output_path', '../../../output/promp')
        self.config.setdefault('log_path', '../../../logs/promp')
        super().process_config()

    def train(self):
        """  Train meta-network  """
        self.mode = 'train'
        num_tasks = self.initialize_environments()
        initial_steps, last_checkpoint, theta_pi = self.initialize_network()
        self.sampler = get_sampler(self.config['pi_network'])
        self.initialize_policy_optimizers()
        current_steps = 0
        while current_steps < self.config['training_frames']:
            task_list = self.sample_tasks(num_tasks)
            task = task_list[self.id]
            self.group_comm = MPI.COMM_WORLD.Split(color=task, key=self.id)
            self.group_id = self.group_comm.Get_rank()
            self.configure_environment_for_task(task)
            # Inner loop:
            pre_summary = {}
            for i in range(self.config['adaptation_steps']):
                summary = self.collect_data()
                if i == 0:
                    pre_summary = summary
                self.update_value_function()
                self.update_experiences()
                self.inner_step(set_grad=i < self.config['adaptation_steps'] - 1)  # theta_pi does not change
                self.update_group()
            # Outer loop:
            post_summary = self.collect_data()
            self.update_value_function()
            self.update_experiences()
            self.update_group()
            # Perform meta-update:
            entropies = self.meta_update(theta_pi)
            # Reset for next task, store results and network:
            self.buffers = []
            prev_steps = current_steps + initial_steps
            new_steps = self.update_logging(pre_summary, post_summary, entropies, prev_steps)
            self.save_networks(prev_steps + new_steps, last_checkpoint)
            current_steps += new_steps

    def meta_update(self, current_params):
        """  Performs meta-update, across all workers and tasks  """
        entropies, pi_gradients = [], []
        update_module_params(self.pi_network, current_params)
        # if self.group_id == 0:
        #     self.update_meta_buffer()
        for i in range(self.config['outer']['train_pi_iter']):
            if self.group_id == 0:
                pi_loss, entropy, kld = self.compute_meta_loss()
                pi_loss = pi_loss + self.config['outer']['eta'] * kld
                zero_optim_grads(self.pi_optimizer)
                pi_loss.backward()
                pi_gradients = [self.store_gradients()]
                entropies.append(entropy)
            mean_policy_grad = self.collect_gradients(pi_gradients)
            self.outer_step(mean_policy_grad)  # current params update with self.pi_network
        return entropies


if __name__ == '__main__':
    """  Runs ProMP training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    parser.add_argument('--use_prior', help='use prior training', required=False, type=int, default=0)
    in_args = parser.parse_args()
    with open(os.path.join(os.getcwd(), in_args.config), 'r') as f1:
        config1 = json.load(f1)
    config1['seed'] = in_args.seed
    config1['use_prior_nets'] = bool(in_args.use_prior)
    if in_args.mode.lower() == 'train':
        config1['use_test_set'] = False
        promp_object = ProMP(config1)
        promp_object.train()
    else:
        raise NotImplementedError('Need to implement testing')

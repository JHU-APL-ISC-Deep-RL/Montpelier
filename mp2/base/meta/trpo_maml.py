import argparse
import os
import json
import torch
import numpy as np
from mpi4py import MPI
from mp2.common.utils import get_sampler
from mp2.common.garage_utils import update_module_params
from mp2.common.mpi_data_utils import mpi_avg, sync_weights, print_now
from mp2.base.meta.maml import MAML
import sys  # todo rm


# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class TRPOMAML(MAML):
    """
    Object for MPI-parallelized MAML, with TRPO as the outer learning algorithm.  Note that, as written,
    this code requires at least as many workers as tasks per meta-update.
    """
    def __init__(self, config):
        """  Constructs TRPO-MAML object.  """
        super().__init__(config)
        assert self.num_workers >= self.config['tasks_per_update'], 'Not enough workers'

    def process_config(self):
        """  Processes input configuration  """
        # TRPO-specific configuration:
        if 'inner' not in self.config:
            self.config['inner'] = {'surrogate': False, 'clip': -1, 'kld_grad': False}  # vpg
        if 'outer' not in self.config:
            self.config['outer'] = {'surrogate': False, 'clip': -1, 'kld_grad': True}  # no surrogate (as in garage)
        self.config.setdefault('max_kl', 0.01)
        self.config.setdefault('cg_iter', 10)  # Number of iterations in conjugate gradient method
        self.config.setdefault('cg_delta', 0)  # Early stopping in conjugate gradient
        self.config.setdefault('damping_coeff', 0.1)  # Improves numerical stability of hessian vector product
        self.config.setdefault('backtrack_iter', 10)  # Maximum number of backtracks allowed per line search
        self.config.setdefault('backtrack_coeff', 0.8)  # How far back to step during backtracking line search
        self.config['backtrack_ratios'] = self.config['backtrack_coeff'] ** np.arange(self.config['backtrack_iter'])
        self.config.setdefault('output_path', '../../../output/trpo_maml')
        self.config.setdefault('log_path', '../../../logs/trpo_maml')
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

    '''
    def compute_kld(self, set_grad=True):
        """  Uses all buffers for a given task to compute meta-loss  """
        theta_pi = dict(self.pi_network.named_parameters())
        update_module_params(self.pi_network, theta_pi)
        for j in range(self.config['adaptation_steps']):
            self.buffer = self.buffers[j]
            require_grad = j < self.config['adaptation_steps'] - 1 or set_grad
            self.inner_step(set_grad=require_grad)  # theta_pi doesn't change
        self.buffer = self.buffers[-1]
        # parameter: grad - None; grad_fn - filled; therefore no longer leaf nodes!
        with torch.set_grad_enabled(set_grad):
            _, _, kld = self.compute_policy_loss(self.config['outer'])
        update_module_params(self.pi_network, theta_pi)  # puts grad_fn to none, restoring leaf node
        return kld
    '''

    def meta_update(self, current_params):
        """  Performs meta-update, across all workers and tasks  """
        # todo: maybe I need to compute kld separately, make a fresh graph for it?

        entropy = None
        update_module_params(self.pi_network, current_params)
        update_comm = MPI.COMM_WORLD.Split(color=self.group_id, key=self.id)
        if self.group_id == 0:
            pi_loss, entropy, kld = self.compute_meta_loss()
            self.inner_optimizer.set_grads_none()  # grad - None; grad_fn - None; gets all parameters
            loss_current = mpi_avg(update_comm, pi_loss.item())
            pi_parameters = list(self.pi_network.parameters())
            loss_grad = self.flat_grad(pi_loss, pi_parameters, retain_graph=True)  # graph?
            g = torch.from_numpy(mpi_avg(update_comm, loss_grad.data.numpy()))
            g_kl = self.flat_grad(kld, pi_parameters, create_graph=True)

            def hessian_vector_product(v):
                hvp = self.flat_grad(g_kl @ v, pi_parameters, retain_graph=True)
                hvp += self.config['damping_coeff'] * v
                return torch.from_numpy(mpi_avg(update_comm, hvp.data.numpy()))

            search_dir = self.conjugate_gradient(hessian_vector_product, g)
            max_length = torch.sqrt(2*self.config['max_kl'] / (search_dir @ hessian_vector_product(search_dir) + 1.e-8))
            max_step = max_length * search_dir

            self.backtracking_line_search(pi_parameters, max_step, loss_current, update_comm)

        # print(str(self.id) + ' before ' + str(self.pi_network.state_dict()['mlp.2.weight'][0, 0]))
        sync_weights(self.group_comm, self.pi_network.parameters())  # this should be over everything, might not matter
        #print(str(self.id) + ' after ' + str(self.pi_network.state_dict()['mlp.2.weight'][0, :3]))
        #sys.stdout.flush()
        return [entropy]

    def backtracking_line_search(self, pi_parameters, max_step, loss_current, comm):

        def apply_update(grad_flattened):
            n = 0
            for p in pi_parameters:
                numel = p.numel()
                gf = grad_flattened[n:n + numel].view(p.shape)
                p.data -= gf
                n += numel

        loss_improvement, kld_new = 0, 0
        for r in self.config['backtrack_ratios']:
            step = r * max_step
            apply_update(step)
            loss_new, _, kld_new = self.compute_meta_loss(set_grad=False)
            loss_new = mpi_avg(comm, loss_new.mean().item())
            kld_new = mpi_avg(comm, kld_new.item())
            loss_improvement = loss_current - loss_new
            if loss_improvement > 0 and kld_new <= self.config['max_kl']:
                break
            apply_update(-step)
        if loss_improvement <= 0 or kld_new > self.config['max_kl']:
            if loss_improvement <= 0:
                print_now(comm, 'step rejected; loss does not improve')
            if kld_new > self.config['max_kl']:
                print_now(comm, 'step rejected; max kld exceeded')

    def conjugate_gradient(self, Ax, b):
        x = torch.zeros_like(b)
        r = b.clone()  # residual
        p = b.clone()  # basis vector
        epsilon = 1.e-8*torch.ones((r@r).size())
        for _ in range(self.config['cg_iter']):
            z = Ax(p)
            r_dot_old = r @ r
            alpha = r_dot_old / ((p @ z) + epsilon)
            x_new = x + alpha * p
            if (x - x_new).norm() <= self.config['cg_delta']:
                return x_new
            r = r - alpha * z
            beta = (r @ r) / r_dot_old
            p = r + beta * p
            x = x_new
        return x

    @staticmethod
    def flat_grad(y, x, retain_graph=False, create_graph=False):
        """  Compute a flat version of gradient of y wrt x  """
        if create_graph:
            retain_graph = True
        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g


if __name__ == '__main__':
    """  Runs TRPO-MAML training or testing for a given input configuration file  """
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
        trpo_maml_object = TRPOMAML(config1)
        trpo_maml_object.train()
    else:
        raise NotImplementedError('Need to implement testing')

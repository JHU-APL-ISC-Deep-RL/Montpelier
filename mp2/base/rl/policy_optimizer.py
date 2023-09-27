import sys
import os
import json
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from shutil import rmtree
from mpi4py import MPI
from datetime import datetime
from warnings import warn
from scipy.signal import lfilter
from mp2.common.mpi_logger import LoggerMPI
from mp2.common.experience_buffer import ExperienceBuffer
from mp2.common.utils import get_env_object, get_network_object, get_sampler, NormalizedReward
from mp2.common.samplers import compute_entropy
from mp2.common.mpi_data_utils import mpi_sum, average_grads, sync_weights, mpi_statistics_scalar, \
    mpi_avg, collect_dict_of_lists, mpi_gather_objects, print_now, zero_print


# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class PolicyOptimizer(object):

    def __init__(self, config):
        """
        Policy Optimization Agent that either uses policy gradient or surrogate objective in conjunction
        with configurable variance reduction measures.
        """
        self.id = MPI.COMM_WORLD.Get_rank()
        self.num_workers = MPI.COMM_WORLD.Get_size()
        self.config = config
        self.process_config()
        self.mode = ''
        self.env = None
        self.obs = None
        self.pi_network = None
        self.v_network = None
        self.pi_optimizer = None
        self.v_optimizer = None
        self.sampler = None
        self.buffer = None
        self.logger = None
        self.epsilon = 1.e-6
        t1 = int(10000 * (datetime.now().timestamp() - int(datetime.now().timestamp())))
        torch.manual_seed((self.id + 1 + self.config['seed'] * self.num_workers) * 2000 + t1)
        t2 = int(10000 * (datetime.now().timestamp() - int(datetime.now().timestamp())))
        np.random.seed((self.id + 1 + self.config['seed'] * self.num_workers) * 5000 + t2)

    def process_config(self):
        """  Processes configuration, filling in missing values as appropriate  """
        # General training configuration:
        self.config.setdefault('seed', 0)                     # Random seed parameter
        self.config.setdefault('training_frames', int(5e7))   # Number of frames to use for training run
        self.config.setdefault('batch_size', 30000)           # Average number of experiences to base an update on
        self.config.setdefault('minibatch_size', -1)          # Minibatch size; default is to not minibatch
        self.config.setdefault('max_ep_length', -1)           # Maximum episode length (< 0 means no max)
        self.config['environment'].setdefault('normalize_reward', -1)
        self.config['steps_per_worker'] = -(-self.config['batch_size'] // self.num_workers)
        self.config.setdefault('pi_lr', .0003)                # Policy optimizer learning rate
        self.config.setdefault('v_lr', .001)                  # Value optimizer learning rate
        self.config.setdefault('opt_epochs', 1)               # Passes through V, pi optimization
        self.config.setdefault('train_v_iter', 10)            # Value updates per epoch
        self.config.setdefault('train_pi_iter', 10)           # Policy updates per epoch
        self.config.setdefault('gamma', 0.99)                 # Discount factor gamma
        self.config.setdefault('lambda', 0.95)                # GAE factor lambda
        self.config.setdefault('clip', 0.2)                   # Clip factor for policy (< 0 means none)
        self.config.setdefault('max_kl', -1)                  # KL criteria for early stopping (< 0 means ignore)
        self.config.setdefault('surrogate', False)            # Whether to use surrogate objective
        self.config.setdefault('e_coeff', 0)                  # Coefficient for entropy term in policy loss
        self.config.setdefault('max_entropy', 0)              # Coefficient for policy entropy reward contribution
        self.config.setdefault('bound_corr', True)            # Whether to use bound correction in policy logprob
        self.config['pi_network']['bound_corr'] = self.config['bound_corr']
        assert self.config['e_coeff'] == 0 or self.config['max_entropy'] == 0, "Only use one entropy regularizer"
        # TRPO-specific configuration:
        self.config.setdefault('trpo', False)
        if self.config['trpo']:
            self.config['opt_epochs'] = 1
            self.config['minibatch_size'] = -1
            self.config['train_pi_iter'] = 1                  # Policy updates per epoch
            self.config['clip'] = -1                          # Clip factor for policy (< 0 means none)
            self.config['surrogate'] = True                   # Whether to use surrogate objective
            self.config.setdefault('cg_iter', 10)             # Number of iterations in conjugate gradient method
            self.config.setdefault('cg_delta', 0)             # Early stopping in conjugate gradient solver
            self.config.setdefault('damping_coeff', 0.1)      # Improves numerical stability of hessian vector product
            self.config.setdefault('backtrack_iter', 10)      # Maximum number of backtracks allowed per line search
            self.config.setdefault('backtrack_coeff', 0.8)    # How far back to step during backtracking line search
            self.config['backtrack_ratios'] = self.config['backtrack_coeff'] ** np.arange(self.config['backtrack_iter'])
        # Testing configuration:
        self.config.setdefault('test_episodes', 1000)
        self.config.setdefault('test_random_base', 100000)
        self.config['test_random_base'] = self.config['test_random_base']*(self.config['seed'] + 1)
        self.config.setdefault('render', False)
        # Logging and storage configurations:
        self.config.setdefault('checkpoint_every', int(5e7))
        self.config.setdefault('use_prior_nets', False)       # Whether to pick up where previous training left off
        self.config.setdefault('model_folder', '../../../output/rl_training')
        self.config.setdefault('log_folder', '../../../logs/rl_training')
        self.config['model_folder'] = os.path.join(os.getcwd(), self.config['model_folder'])
        self.config['log_folder'] = os.path.join(os.getcwd(), self.config['log_folder'])
        self.config['model_folder'] = self.config['model_folder'] + '_' + str(self.config['seed'])
        self.config['log_folder'] = self.config['log_folder'] + '_' + str(self.config['seed'])
        if sys.platform[:3] == 'win':
            self.config['model_folder'] = self.config['model_folder'].replace('/', '\\')
            self.config['log_folder'] = self.config['log_folder'].replace('/', '\\')
        if self.id == 0:
            if not self.config['use_prior_nets']:  # start a fresh training run
                if os.path.isdir(self.config['log_folder']):
                    rmtree(self.config['log_folder'], ignore_errors=True)
                if os.path.isdir(self.config['model_folder']):
                    rmtree(self.config['model_folder'], ignore_errors=True)
            Path(self.config['model_folder']).mkdir(parents=True, exist_ok=True)
            Path(self.config['log_folder']).mkdir(parents=True, exist_ok=True)

    def train(self):
        """  Train neural network  """
        # Initialize relevant objects:
        self.mode = 'train'
        self.initialize_env()
        initial_steps, last_checkpoint = self.initialize_networks()
        self.sampler = get_sampler(self.config['pi_network'])
        self.initialize_optimizers()
        self.obs = self.env.reset()
        self.initialize_logging()
        indices, starts, stops = self.allocate_minibatches()
        # Run training:
        current_steps = 0
        while current_steps < self.config['training_frames']:
            # Collect data:
            self.buffer = ExperienceBuffer()  # reset experience buffer
            steps_current, all_episode_summaries = 0, {}
            while self.buffer.steps < self.config['steps_per_worker']:
                episode_summary = self.run_trajectory()
                steps_current = mpi_sum(MPI.COMM_WORLD, self.buffer.steps)
                if steps_current == 0:  # first iteration
                    all_episode_summaries = {k: [v] for k, v in episode_summary.items()}
                else:
                    all_episode_summaries = self.concatenate_dict_of_lists(all_episode_summaries, episode_summary)
            # Update network(s), as prescribed by config:
            losses = self.update_network(indices, starts, stops)
            # Update logging, save the model:
            prev_steps = initial_steps + current_steps
            new_steps = self.update_logging(all_episode_summaries, losses, prev_steps)
            last_checkpoint = self.save_networks(prev_steps + new_steps, last_checkpoint)
            current_steps += new_steps

    def initialize_env(self):
        """  Initialize environment object.  Reset not included here to support CrowdNav.  """
        self.env = get_env_object(self.config['environment'])
        if self.config['environment']['normalize_reward'] > 0:
            self.env = NormalizedReward(self.env, self.config['environment']['normalize_reward'])
        if self.mode == 'test':
            self.env.seed(self.id*self.config['test_episodes'] + self.config['test_random_base'])

    def initialize_networks(self):
        """  Initialize network objects  """
        total_steps, last_checkpoint = 0, -1
        self.pi_network = get_network_object(self.config['pi_network'])
        self.v_network = get_network_object(self.config['v_network'])
        if self.config['use_prior_nets']:
            checkpoint_pi = torch.load(os.path.join(self.config['model_folder'], 'model-latest.pt'))
            self.pi_network.load_state_dict(checkpoint_pi['model'])
            total_steps = checkpoint_pi['steps']
            last_checkpoint = total_steps // self.config['checkpoint_every']
            checkpoint_v = torch.load(os.path.join(self.config['model_folder'], 'value-latest.pt'))
            self.v_network.load_state_dict(checkpoint_v['model'])
        sync_weights(MPI.COMM_WORLD, self.pi_network.parameters())
        sync_weights(MPI.COMM_WORLD, self.v_network.parameters())
        return total_steps, last_checkpoint

    def initialize_optimizers(self):
        """  Initializes Adam optimizer for training network.  Only one worker actually updates parameters.  """
        self.pi_optimizer = torch.optim.Adam(params=self.pi_network.parameters(), lr=self.config['pi_lr'])
        self.v_optimizer = torch.optim.Adam(params=self.v_network.parameters(), lr=self.config['v_lr'])
        if self.config['use_prior_nets']:
            checkpoint_pi = torch.load(os.path.join(self.config['model_folder'], 'model-latest.pt'))
            self.pi_optimizer.load_state_dict(checkpoint_pi['optimizer'])
            checkpoint_v = torch.load(os.path.join(self.config['model_folder'], 'value-latest.pt'))
            self.v_optimizer.load_state_dict(checkpoint_v['optimizer'])

    def initialize_logging(self):
        """  Initialize logger and store config (only on one process)  """
        with open(os.path.join(self.config['model_folder'], 'config.pkl'), 'wb') as config_file:
            pickle.dump(self.config, config_file)  # store configuration
        self.logger = LoggerMPI(self.config['log_folder'])
        self.logger.log_graph(self.obs, self.pi_network)

    def run_trajectory(self, random_seed=None):
        """  Run trajectories based on current network(s)  """
        trajectory_buffer, trajectory_info = np.array([]).reshape(0, 7), {}
        steps_left = self.config['steps_per_worker'] - self.buffer.steps
        num_frames = 0
        while True:
            policy, value = self.forward_pass()
            action, log_prob = self.sampler.get_action_and_log_prob(policy)
            output_obs, reward, done, info = self.env.step(action)
            self.concatenate_dict_of_lists(trajectory_info, info)
            if self.config['render'] and self.mode == 'test':
                self.env.render()
            num_frames += 1
            if num_frames == self.config['max_ep_length']:
                done = True
            trajectory_buffer = self.update_trajectory_buffer(trajectory_buffer, action, reward, policy,
                                                              log_prob, value, done)
            if done:
                self.buffer.bootstrap = np.zeros(1)
                episode_summary = self.process_trajectory(trajectory_buffer, trajectory_info)
                self.reset_env(random_seed)
                break
            elif num_frames == steps_left:
                _, self.buffer.bootstrap = self.forward_pass()
                episode_summary = self.process_trajectory(trajectory_buffer, trajectory_info)
                self.obs = output_obs
                break
            else:
                self.obs = output_obs
        return episode_summary

    def reset_env(self, random_seed):
        """  Resets for next environment instance  """
        if random_seed is not None:
            self.env.seed(random_seed)  # for testing
        self.obs = self.env.reset()

    def forward_pass(self):
        """  Runs forward pass of network(s).  For continuous action spaces, policy will be a tuple of mean, std. """
        with torch.no_grad():
            policy = self.pi_network.forward_with_processing(self.obs)
            value = self.v_network.forward_with_processing(self.obs)
            value = value.numpy()[0]
        return policy, value

    def update_trajectory_buffer(self, trajectory_buffer, action, reward, policy, log_prob, value, done):
        """  Updates episode buffer for current step  """
        if self.pi_network.config['discrete']:
            policy_to_store = np.squeeze(policy.detach().numpy())
        else:
            policy_to_store = np.concatenate((policy[0].detach().numpy(), policy[1].detach().numpy()))
        raw_action = self.sampler.get_raw_action(action)
        experience = np.reshape(np.array([self.obs, raw_action, reward, policy_to_store, log_prob, value, done],
                                         dtype=object), (1, 7))
        return np.concatenate((trajectory_buffer, experience))

    def process_trajectory(self, trajectory_buffer, trajectory_info):
        """  Processes a completed trajectory, storing required data in buffer and returning episode summary  """
        if self.mode == 'train':
            max_entropy_term = self.compute_max_entropy_term(trajectory_buffer)
            q_values = self.compute_target_values(trajectory_buffer[:, 2] + max_entropy_term)
            self.buffer.update(trajectory_buffer, q_values, max_ent=max_entropy_term)
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

    def compute_max_entropy_term(self, trajectory_buffer):
        """  Computes max entropy contribution for current trajectory buffer  """
        if self.config['max_entropy'] > 0:
            max_entropy_term = self.sampler.np_entropy(np.vstack(trajectory_buffer[:, 3])) * self.config['max_entropy']
        else:
            max_entropy_term = np.zeros(trajectory_buffer[:, 2].shape)
        return max_entropy_term

    def compute_target_values(self, rewards):
        """  Computes value function targets (without bootstrapping)  """
        rewards_plus = np.concatenate((rewards, self.buffer.bootstrap))
        return self.discount(rewards_plus, self.config['gamma'])[:-1]

    def update_network(self, indices, starts, stops):
        """  Updates the networks based on processing from all workers  """
        # Update value network:
        observations = torch.from_numpy(np.vstack(self.buffer.observations)).float()
        pi_losses, v_losses, entropies = [], [], []
        for epoch in range(self.config['opt_epochs']):
            np.random.shuffle(indices)
            for start, stop in zip(starts, stops):
                mb = indices[start:stop]
                if 'v_network' in self.config:
                    for i in range(self.config['train_v_iter']):  # update value function
                        self.v_optimizer.zero_grad()
                        target_values = torch.from_numpy(self.buffer.q_values.astype(float)).float()
                        v_loss = self.compute_value_loss(observations, target_values)
                        v_losses.append(v_loss.item())
                        v_loss.backward()
                        average_grads(MPI.COMM_WORLD, self.v_network.parameters())
                        if self.id == 0:
                            self.v_optimizer.step()
                        sync_weights(MPI.COMM_WORLD, self.v_network.parameters())
                    self.update_values(observations)
                # Update advantage estimates, standardizing across workers:
                self.estimate_advantages()
                advantages = torch.from_numpy(self.buffer.advantages.astype(float)).float()
                # Update policy network:
                actions = self.buffer.actions
                log_probs = torch.from_numpy(self.buffer.log_probs.astype(float)).float()
                if not self.config['trpo']:
                    for i in range(self.config['train_pi_iter']):
                        self.pi_optimizer.zero_grad()
                        pi_loss, entropy, kld = self.compute_policy_loss(observations, actions, advantages,
                                                                         log_probs, self.buffer.policies, mb)
                        pi_loss = torch.mean(pi_loss, dim=0)  # assumes equal experiences per worker
                        pi_losses.append(pi_loss.item())
                        entropies.append(entropy.item())
                        mean_kld = mpi_avg(MPI.COMM_WORLD, kld.item())
                        if mean_kld > self.config['max_kl'] > 0:
                            if self.id == 0:
                                print_now(MPI.COMM_WORLD,
                                          'Policy KL divergence exceeds limit; stopping update at step %d.' % i)
                            break
                        pi_loss.backward()
                        average_grads(MPI.COMM_WORLD, self.pi_network.parameters())
                        if self.id == 0:
                            self.pi_optimizer.step()
                        sync_weights(MPI.COMM_WORLD, self.pi_network.parameters())
                    return {'pi_losses': pi_losses, 'v_losses': v_losses, 'entropies': entropies}
                else:  # trpo update
                    pi_loss, entropy, kld = self.compute_policy_loss(observations, actions, advantages, 
                                                                     log_probs, self.buffer.policies, mb)
                    pi_loss = torch.mean(pi_loss, dim=0)
                    loss_current = mpi_avg(MPI.COMM_WORLD, pi_loss.item())
                    pi_parameters = list(self.pi_network.parameters())
                    loss_grad = self.flat_grad(pi_loss, pi_parameters, retain_graph=True)
                    g = torch.from_numpy(mpi_avg(MPI.COMM_WORLD, loss_grad.data.numpy()))
                    g_kl = self.flat_grad(kld, pi_parameters, create_graph=True)

                    def hessian_vector_product(v):
                        hvp = self.flat_grad(g_kl @ v, pi_parameters, retain_graph=True)
                        hvp += self.config['damping_coeff'] * v
                        return torch.from_numpy(mpi_avg(MPI.COMM_WORLD, hvp.data.numpy()))

                    search_dir = self.conjugate_gradient(hessian_vector_product, g)
                    max_length = torch.sqrt(2*self.config['max_kl'] /
                                            (search_dir@hessian_vector_product(search_dir) + 1.e-8))
                    max_step = max_length * search_dir

                    def backtracking_line_search():

                        def apply_update(grad_flattened):
                            n = 0
                            for p in pi_parameters:
                                numel = p.numel()
                                gf = grad_flattened[n:n + numel].view(p.shape)
                                p.data -= gf
                                n += numel

                        loss_improvement = 0
                        for r in self.config['backtrack_ratios']:
                            step = r * max_step
                            apply_update(step)
                            with torch.no_grad():
                                loss_new, _, kld_new = self.compute_policy_loss(observations, actions, advantages,
                                                                                log_probs, self.buffer.policies, mb)
                                loss_new = mpi_avg(MPI.COMM_WORLD, loss_new.mean().item())
                                kld_new = mpi_avg(MPI.COMM_WORLD, kld_new.item())
                            loss_improvement = loss_current - loss_new
                            if loss_improvement > 0 and kld_new <= self.config['max_kl']:
                                break
                            apply_update(-step)
                        if loss_improvement <= 0 or kld_new > self.config['max_kl']:
                            if loss_improvement <= 0:
                                zero_print(MPI.COMM_WORLD, 'step rejected; loss does not improve')
                            if kld_new > self.config['max_kl']:
                                zero_print(MPI.COMM_WORLD, 'step rejected; max kld exceeded')
                            return loss_current
                        else:
                            return loss_new

                    final_loss = backtracking_line_search()
                    return {'pi_losses': [final_loss], 'v_losses': v_losses, 'entropies': [entropy.item()]}

    def allocate_minibatches(self):
        """  Return minibatch indices  """
        indices = list(range(self.config['steps_per_worker']))
        if self.config['minibatch_size'] > 0:
            starts = list(range(0, len(indices), self.config['minibatch_size']))
            stops = [item + self.config['minibatch_size'] for item in starts]
            if stops[-1] != len(indices):
                warn('Trajectory length is not a multiple of minibatch size; wasting data')
            stops[-1] = len(indices)
        else:
            starts, stops = [0], [len(indices)]
        return indices, starts, stops

    def compute_value_loss(self, observations, target_values, mb=None):
        """  Compute value function loss  """
        if mb is None:
            new_values = self.v_network(observations).view(-1)
            return torch.mean(torch.pow(new_values - target_values, 2), dim=0)
        else:
            new_values = self.v_network(observations[mb]).view(-1)
            return torch.mean(torch.pow(new_values - target_values[mb], 2), dim=0)

    def update_values(self, observations):
        """  Estimate values with updated value network, store in buffer  """
        with torch.no_grad():
            self.buffer.values = self.v_network(observations).view(-1).numpy()

    def estimate_advantages(self):
        """  Estimate advantages for a sequence of observations and rewards  """
        rewards, values, dones = self.buffer.rewards + self.buffer.max_ent, self.buffer.values, self.buffer.dones
        self.buffer.advantages = self.estimate_generalized_advantage(rewards, values, dones)
        mean_adv, std_adv = mpi_statistics_scalar(MPI.COMM_WORLD, self.buffer.advantages)
        self.buffer.advantages = (self.buffer.advantages - mean_adv) / std_adv
        return mean_adv, std_adv

    def estimate_generalized_advantage(self, rewards, values, dones):
        """  Generalized advantage estimation, given rewards and value estimates  """
        terminals = np.nonzero(dones.astype(int))[0]
        terminals = list(np.concatenate((np.array([-1]), terminals)))
        if not dones[-1]:
            terminals += [rewards.shape[0] - 1]
        gae = np.zeros(rewards.shape)
        for i in range(len(terminals[:-1])):
            episode_rewards = rewards[terminals[i] + 1:terminals[i + 1] + 1]
            episode_values = values[terminals[i] + 1:terminals[i + 1] + 1]
            if i < len(terminals[:-1]) - 1:
                episode_next_values = np.concatenate((episode_values[1:], np.array([0.])))  # end-of-episode
            else:
                episode_next_values = np.concatenate((episode_values[1:], self.buffer.bootstrap))
            episode_deltas = episode_rewards + self.config['gamma'] * episode_next_values - episode_values
            for start in range(len(episode_values)):
                indices = np.arange(start, len(episode_values))
                discounts = np.power(self.config['gamma'] * self.config['lambda'], indices - start)
                discounted_future_deltas = episode_deltas[start:] * discounts
                gae[start + terminals[i] + 1] = np.sum(discounted_future_deltas)
        return gae

    def compute_policy_loss(self, observations, actions, advantages, old_log_probs, old_policies, mb=None):
        """  Compute policy loss, entropy, kld  """
        if mb is None:
            mb = list(range(observations.shape[0]))
        new_policies = self.pi_network(observations[mb])
        if self.pi_network.config['discrete']:
            actions_one_hot = torch.from_numpy(
                np.eye(self.pi_network.config['action_dim'])[np.squeeze(actions[mb])]).float()
            new_policies = torch.masked_fill(new_policies, new_policies < self.epsilon, self.epsilon)
            new_log_probs = torch.sum(torch.log(new_policies) * actions_one_hot, dim=1)
        else:
            new_dist = self.sampler.get_distribution(new_policies)
            actions_torch = torch.from_numpy(np.vstack(actions[mb])).float()
            if self.config['bound_corr']:
                below = new_dist.cdf(actions_torch.clamp(-1.0, 1.0))
                below = below.clamp(self.epsilon, 1.0).log() * actions_torch.le(-1).float()
                above = torch.ones(actions_torch.size()) - new_dist.cdf(actions_torch.clamp(-1.0, 1.0))
                above = above.clamp(self.epsilon, 1.0).log() * actions_torch.ge(1).float()
                inner = new_dist.log_prob(actions_torch) * actions_torch.gt(-1).float() * actions_torch.lt(1).float()
                new_log_probs = torch.sum(inner + below + above, dim=-1)
            else:
                new_log_probs = torch.sum(new_dist.log_prob(actions_torch), dim=-1)
        entropy = self.sampler.compute_entropy(new_policies)
        if self.config['surrogate']:
            ratio = torch.exp(new_log_probs - old_log_probs[mb])
            if self.config['clip'] > 0:
                pi_losses_1 = -advantages[mb] * ratio
                pi_losses_2 = -advantages[mb] * torch.clamp(ratio, 1.0 - self.config['clip'], 1.0 + self.config['clip'])
                pi_loss = torch.max(pi_losses_1, pi_losses_2) - self.config['e_coeff'] * entropy
            else:
                pi_loss = -advantages[mb] * ratio - self.config['e_coeff'] * entropy
        else:
            if self.config['clip'] > 0:
                log_diff = torch.clamp(new_log_probs - old_log_probs[mb],
                                       np.log(1.0 - self.config['clip']), np.log(1.0 + self.config['clip']))
                pi_losses_1 = -advantages[mb] * new_log_probs
                pi_losses_2 = -advantages[mb] * (old_log_probs[mb] + log_diff)
                pi_loss = torch.max(pi_losses_1, pi_losses_2) - self.config['e_coeff'] * entropy
            else:
                pi_loss = -advantages[mb] * new_log_probs - self.config['e_coeff'] * entropy
        kld = self.compute_kld(new_policies, old_policies[mb])
        return pi_loss, entropy, kld

    def compute_kld(self, policy_predictions, old_policies):
        if self.config['trpo']:
            new_dist = self.sampler.get_distribution(policy_predictions)
        else:
            with torch.no_grad():
                new_dist = self.sampler.get_distribution(policy_predictions)
        old_dist = self.sampler.restore_distribution(old_policies)
        return torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

    def update_logging(self, episode_summaries, losses, previous_steps):
        """  Updates TensorBoard logging based on most recent update  """
        steps = self.buffer.steps
        local_keys = list(episode_summaries.keys())
        all_keys = mpi_gather_objects(MPI.COMM_WORLD, local_keys)
        keys_in_each = self.find_common(all_keys)
        for k in keys_in_each:
            self.logger.log_mean_value('Performance/' + k, episode_summaries[k], steps, previous_steps)
        for k, v in losses.items():
            self.logger.log_mean_value('Losses/' + k, v, steps, previous_steps)
        self.logger.flush()
        steps_across_processes = mpi_sum(MPI.COMM_WORLD, steps)
        return steps_across_processes

    def save_networks(self, total_steps, last_checkpoint):
        """  Save networks, as required.  Update last_checkpoint.  """
        if self.id == 0:
            torch.save({'model': self.pi_network.state_dict(),
                        'optimizer': self.pi_optimizer.state_dict(),
                        'steps': total_steps},
                       os.path.join(self.config['model_folder'], 'model-latest.pt'))
            torch.save({'model': self.v_network.state_dict(),
                        'optimizer': self.v_optimizer.state_dict()},
                       os.path.join(self.config['model_folder'], 'value-latest.pt'))
            if total_steps // self.config['checkpoint_every'] > last_checkpoint:  # periodically keep checkpoint
                last_checkpoint += 1
                suffix = str(int(last_checkpoint * self.config['checkpoint_every']))
                torch.save({'model': self.pi_network.state_dict(),
                            'optimizer': self.pi_optimizer.state_dict(),
                            'steps': total_steps},
                           os.path.join(self.config['model_folder'], 'model-' + suffix + '.pt'))
                torch.save({'model': self.v_network.state_dict(),
                            'optimizer': self.v_optimizer.state_dict()},
                           os.path.join(self.config['model_folder'], 'value-' + suffix + '.pt'))
        return last_checkpoint

    def test(self):
        """  Run testing episodes with fixed random seed, collect and save data  """
        # Run testing episodes:
        self.mode = 'test'
        self.initialize_env()
        self.initialize_networks()
        self.sampler = get_sampler(self.config['pi_network'], True)
        self.obs = self.env.reset()
        local_episodes, total_episodes, local_episode_summaries = 0, 0, {}
        self.pi_network.eval()
        if self.v_network is not None:
            self.v_network.eval()
        while total_episodes < self.config['test_episodes']:
            self.buffer = ExperienceBuffer()  # reset experience buffer
            random_seed = self.id*self.config['test_episodes'] + self.config['test_random_base'] + local_episodes + 1
            episode_summary = self.run_trajectory(int(random_seed))
            if len(local_episode_summaries.keys()) == 0:  # first iteration
                local_episode_summaries = {k: [v] for k, v in episode_summary.items()}
            else:
                local_episode_summaries = self.concatenate_dict_of_lists(local_episode_summaries, episode_summary)
            local_episodes += 1
            total_episodes = int(mpi_sum(MPI.COMM_WORLD, local_episodes))
            if self.id == 0:
                print_now(MPI.COMM_WORLD, str(total_episodes) + ' episodes complete.')
        # Collect, process, save data:
        test_output = collect_dict_of_lists(MPI.COMM_WORLD, local_episode_summaries)
        self.store_test_results(test_output)
        return test_output

    def store_test_results(self, test_output):
        """  Save a pickle with test results  """
        if self.id == 0:
            test_file = os.path.join(self.config['model_folder'], 'test_results.pkl')
            with open(test_file, 'wb') as opened_test:
                pickle.dump(test_output, opened_test)

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

    @staticmethod
    def discount(x, gamma):
        """  Computes discounted quantity (used for discounted future reward). """
        return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    @staticmethod
    def compute_metrics(episode_data):
        """  Computes metrics to be evaluated as learning progresses  """
        mean_reward = sum(episode_data['episode_reward']) / len(episode_data)
        return {'mean': mean_reward}

    @staticmethod
    def concatenate_dict_of_arrays(base_dict, new_dict):
        """  Collect a dictionary of numpy arrays  """
        for k in new_dict:
            base_dict[k] = np.concatenate((base_dict[k], new_dict[k]))
        return base_dict

    @staticmethod
    def concatenate_dict_of_lists(base_dict, new_dict):
        """  Collect a dictionary of lists  """
        for k in new_dict:
            if k in base_dict:
                base_dict[k].append(new_dict[k])
            else:
                base_dict[k] = [new_dict[k]]
        return base_dict

    @staticmethod
    def find_common(list_of_lists):
        """  Returns members common to each list in a list of lists  """
        common = set(list_of_lists[0])
        for item in list_of_lists[1:]:
            common = common.intersection(set(item))
        return sorted(list(common))

    @staticmethod
    def flatten_list(input_list):
        """  Flattens a list of lists  """
        return [item for sublist in input_list for item in sublist]


if __name__ == '__main__':
    """  Runs PolicyOptimizer training or testing for a given input configuration file  """
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
        po_object = PolicyOptimizer(config1)
        po_object.train()
    else:
        config1['use_prior_nets'] = True
        po_object = PolicyOptimizer(config1)
        po_object.test()

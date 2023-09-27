import argparse
import sys
import os
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from shutil import rmtree
from mpi4py import MPI
from datetime import datetime
from copy import deepcopy
from scipy.signal import lfilter
from mp2.common.experience_buffer import ExperienceBuffer
from mp2.common.samplers import compute_entropy
from mp2.common.utils import get_env_object, get_network_object, get_sampler
from mp2.common.baselines import LinearFeatureBaseline
from mp2.common.utils import ReducedLogging, StopOnSuccess, ZeroOnSuccess, NormalizedReward
from mp2.common.garage_utils import DifferentiableSGD, update_module_params, zero_optim_grads
from mp2.common.mpi_data_utils import sync_weights, mpi_gather_objects, mpi_sum, \
    mpi_statistics_scalar, print_now
from mp2.common.mpi_logger import LoggerMPI

# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)


class MAML(object):
    """  Object for MPI-parallelized MAML  """
    def __init__(self, config):
        """  Constructs MAML object  """
        self.config = config                   # configuration dictionary
        self.mode = ''                         # train or test
        self.env_collection = []               # collection of environment objects
        self.env = None                        # current environment under consideration
        self.current_task = -1                 # current task index
        self.batch_tasks = []                  # list of tasks in this batch
        self.obs = None                        # current observation
        self.pi_network = None                 # policy meta-network
        self.v_function = None                 # value function
        self.sampler = None                    # for choosing actions
        self.buffer = None                     # for storing current experiences
        self.buffers = []                      # for storing experiences across all steps on current task
        self.pi_optimizer = None               # policy meta-optimizer
        self.inner_optimizer = None            # inner loop policy optimizer
        self.task_scales = []                  # scale coefficients for task rewards (not necessary for base MAML)
        self.logger = None                     # object for TensorBoard logging
        self.epsilon = 1.e-6
        self.id = MPI.COMM_WORLD.Get_rank()
        self.num_workers = MPI.COMM_WORLD.Get_size()
        self.group_comm = None
        self.group_id = None
        self.process_config()
        self.task_group = -(-self.num_workers // self.config['tasks_per_update'])  # process in groups if enough cores
        self.trajectories_this_task = -1
        self.experiences_this_task = -1
        t1 = int(10000 * (datetime.now().timestamp() - int(datetime.now().timestamp())))
        torch.manual_seed((self.id + 1 + self.config['seed'] * self.num_workers) * 2000 + t1)
        t2 = int(10000 * (datetime.now().timestamp() - int(datetime.now().timestamp())))
        np.random.seed((self.id + 1 + self.config['seed'] * self.num_workers) * 5000 + t2)

    def process_config(self):
        """  Processes input configuration  """
        self.config.setdefault('training_frames', 5e7)
        self.config.setdefault('seed', 0)
        self.config.setdefault('tasks_per_update', 20)
        if self.config['tasks_per_update'] > self.num_workers:
            assert self.config['tasks_per_update'] % self.num_workers == 0, 'Need # workers to evenly divide # tasks'
        self.config.setdefault('trajectories_per_task', 10)
        self.config.setdefault('experiences_per_task', -1)     # allows one to base updates on # experiences
        self.config.setdefault('adaptation_steps', 1)
        self.config.setdefault('max_ep_length', -1)            # -1: only stop episode on done
        self.config.setdefault('inner', 'vpg')
        self.config.setdefault('outer', 'vpg')
        alg_defaults = {'vpg': {'surrogate': False, 'clip': -1, 'kld_grad': False},
                        'ppo': {'surrogate': True, 'clip': 0.5, 'kld_grad': False}}
        if type(self.config['inner']) != dict:
            self.config['inner'] = alg_defaults[self.config['inner']]
        if type(self.config['outer']) != dict:
            self.config['outer'] = alg_defaults[self.config['outer']]
        self.config.setdefault('e_coeff', 0)
        self.config.setdefault('inner_lr', 0.01)
        self.config.setdefault('outer_lr', 0.001)
        self.config['environment'].setdefault('task_norm_file', None)
        self.config['environment'].setdefault('normalize_reward', False)
        self.config.setdefault('bound_corr', True)
        self.config['pi_network']['bound_corr'] = self.config['bound_corr']
        self.config.setdefault('max_entropy', -1)
        self.config.setdefault('checkpoint_every', 20000000)
        self.config.setdefault('use_prior_nets', False)
        self.config.setdefault('model_folder', '../../../output/maml')
        self.config.setdefault('log_folder', '../../../logs/maml')
        self.config['model_folder'] = os.path.join(os.getcwd(), self.config['model_folder'])
        self.config['log_folder'] = os.path.join(os.getcwd(), self.config['log_folder'])
        self.config['model_folder'] = self.config['model_folder'] + '_' + str(self.config['seed'])
        self.config['log_folder'] = self.config['log_folder'] + '_' + str(self.config['seed'])
        if sys.platform[:3] == 'win':
            self.config['model_folder'] = self.config['model_folder'].replace('/', '\\')
            self.config['log_folder'] = self.config['log_folder'].replace('/', '\\')
        if self.id == 0:  # start a fresh training run with worker 0, unless configured not to
            if not self.config['use_prior_nets']:
                if os.path.isdir(self.config['log_folder']):
                    rmtree(self.config['log_folder'], ignore_errors=True)
                if os.path.isdir(self.config['model_folder']):
                    rmtree(self.config['model_folder'], ignore_errors=True)
                Path(self.config['model_folder']).mkdir(parents=True, exist_ok=True)
                Path(self.config['log_folder']).mkdir(parents=True, exist_ok=True)

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
            worker_tasks = task_list[self.id:len(task_list):self.num_workers]  # tasks to be trained by this worker
            # print(str(self.id) + ': ' + str(worker_tasks))
            self.group_comm = MPI.COMM_WORLD.Split(color=worker_tasks[0], key=self.id)  # used when # workers > # tasks
            self.group_id = self.group_comm.Get_rank()
            policy_grads, entropies = [], []
            all_pre_summaries, all_post_summaries = {}, {}
            for task in worker_tasks:
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
                # Compute and tabulate meta-gradient:
                policy_grad, entropy = self.compute_meta_gradient(theta_pi)
                if policy_grad is not None:
                    policy_grads.append(policy_grad)
                    entropies.append(entropy)
                # Store values for logging:
                all_pre_summaries = self.concatenate_dict_of_lists([pre_summary, all_pre_summaries])
                all_post_summaries = self.concatenate_dict_of_lists([post_summary, all_post_summaries])
                # Reset for next task:
                update_module_params(self.pi_network, theta_pi)
                self.buffers = []  # maybe just use
            # Meta-update:
            mean_policy_grad = self.collect_gradients(policy_grads)
            self.outer_step(mean_policy_grad)  # theta_pi changes; update done in place w backward
            # Store results and network
            prev_steps = initial_steps + current_steps
            new_steps = self.update_logging(all_pre_summaries, all_post_summaries, entropies, prev_steps)
            last_checkpoint = self.save_networks(prev_steps + new_steps, last_checkpoint)
            current_steps += new_steps

    def sample_tasks(self, num_tasks):
        """  Sample tasks for a given iteration from list of environments  """
        task_list = []
        if self.id == 0:
            task_indices = np.arange(num_tasks)
            if self.config['tasks_per_update'] <= num_tasks:
                tasks_to_use = np.random.choice(task_indices, self.config['tasks_per_update'], replace=False)
            else:
                even_distribution = np.tile(task_indices, self.config['tasks_per_update'] // num_tasks)
                remainder = np.random.choice(task_indices, self.config['tasks_per_update'] % num_tasks, replace=False)
                tasks_to_use = np.concatenate((even_distribution, remainder))
            task_list = list(tasks_to_use)
            task_list *= self.task_group
            task_list = task_list[:max(self.config['tasks_per_update'], self.num_workers)]
            task_list.sort()
        task_list = MPI.COMM_WORLD.bcast(task_list, root=0)
        return task_list

    def configure_environment_for_task(self, task_index):
        """  Load training environment for a given task  """
        if task_index == self.current_task:
            return
        if self.config['environment']['type'].lower() == 'metaworld':
            class_index = task_index // 50
            if self.config['use_test_set']:
                env = list(self.env_collection.test_classes.values())[class_index]()
                env.set_task(self.env_collection.test_tasks[task_index])
            else:
                env = list(self.env_collection.train_classes.values())[class_index]()
                env.set_task(self.env_collection.train_tasks[task_index])
            if self.config['environment']['stop_on_success']:
                env = StopOnSuccess(env)
            elif self.config['environment']['zero_on_success']:
                env = ZeroOnSuccess(env)
            else:
                env = ReducedLogging(env)
            if self.config['environment']['task_norm_file'] is not None:
                env = NormalizedReward(env, self.task_scales[task_index])
            else:
                if self.config['environment']['normalize_reward'] > 0:
                    env = NormalizedReward(env, self.config['environment']['normalize_reward'])
            self.env = env
            self.current_task = task_index
            self.obs = self.env.reset()
        else:
            raise NotImplementedError('Environment type not yet implemented.')

    def collect_data(self):
        self.buffer = ExperienceBuffer()  # reset experience buffer
        workers_this_task = self.group_comm.Get_size()
        summaries = []
        if self.config['experiences_per_task'] <= 0:
            self.trajectories_this_task = -(-self.config['trajectories_per_task'] // workers_this_task)
            for _ in range(self.trajectories_this_task):
                trajectory_summary = self.run_trajectory()
                summaries.append(trajectory_summary)
        else:
            self.experiences_this_task = -(-self.config['experiences_per_task'] // workers_this_task)
            while True:
                trajectory_summary = self.run_trajectory()
                summaries.append(trajectory_summary)
                if self.buffer.observations.shape[0] >= self.experiences_this_task:
                    break
        summaries = self.mean_of_dict_of_lists(self.concatenate_dict_of_lists(summaries))
        return summaries

    def run_trajectory(self, random_seed=None):
        """  Run trajectories based on current network(s)  """
        trajectory_buffer, trajectory_info = np.array([]).reshape(0, 7), {}
        steps_left = self.experiences_this_task - self.buffer.steps
        num_frames = 0
        while True:
            policy = self.forward_pass()
            action, log_prob = self.sampler.get_action_and_log_prob(policy)
            output_obs, reward, done, info = self.env.step(action)
            num_frames += 1
            if num_frames == self.config['max_ep_length']:
                done = True
            trajectory_buffer = self.update_trajectory_buffer(trajectory_buffer, action, reward, policy,
                                                              log_prob, done)
            self.append_dict_of_lists(trajectory_info, info)
            if done:
                episode_summary = self.process_trajectory(trajectory_buffer, trajectory_info)
                if random_seed is not None:
                    self.env.seed(random_seed)
                self.obs = self.env.reset()
                break
            elif num_frames == steps_left:  # todo: check bootstrapping
                episode_summary = self.process_trajectory(trajectory_buffer, trajectory_info)
                self.obs = output_obs
                break
            else:
                self.obs = output_obs
        return episode_summary

    def forward_pass(self):
        """  Runs forward pass of network(s).  For continuous action spaces, policy will be a tuple of mean, std. """
        with torch.no_grad():
            return self.pi_network.forward_with_processing(self.obs)

    def update_trajectory_buffer(self, trajectory_buffer, action, reward, policy, log_prob, done):
        """  Updates episode buffer for current step  """
        if self.pi_network.config['discrete']:
            policy_to_store = np.squeeze(policy.numpy())
        else:
            policy_to_store = np.concatenate((policy[0].numpy(), policy[1].numpy()))
        raw_action = self.sampler.get_raw_action(action)
        experience = np.reshape(np.array([self.obs, raw_action, reward, policy_to_store, log_prob, 0., done],
                                         dtype=object), (1, 7))  # 0. in here because value mapping not yet defined
        return np.concatenate((trajectory_buffer, experience))

    def process_trajectory(self, trajectory_buffer, trajectory_info):
        """  Processes a completed trajectory, storing required data in buffer and returning episode summary  """
        if self.mode == 'train':
            max_entropy_term = self.compute_max_entropy_term(trajectory_buffer)
            q_values = self.compute_target_values(trajectory_buffer[:, 2] + max_entropy_term)
            self.buffer.update(trajectory_buffer, q_values, max_ent=max_entropy_term)
        if self.buffer.dones[-1]:  # base logging only on complete episodes
            entropies = [compute_entropy(self.buffer.current_episode[i, 3])
                         for i in range(self.buffer.current_episode.shape[0])]
            return {'episode_reward': [np.sum(self.buffer.current_episode[:, 2])],
                    'episode_entropy': [np.mean(entropies)],
                    **{k: v[-1] for k, v in trajectory_info.items()}}
        else:
            return {}

    def compute_max_entropy_term(self, trajectory_buffer):
        """  Computes max entropy contribution for current trajectory buffer  """
        if self.config['max_entropy'] > 0:  # takes control bounds into account
            max_entropy_term = self.sampler.np_entropy(np.vstack(trajectory_buffer[:, 3])) * self.config['max_entropy']
        else:
            max_entropy_term = np.zeros(trajectory_buffer[:, 2].shape)
        return max_entropy_term

    def compute_target_values(self, rewards):
        """  Computes value function targets (without bootstrapping)  """
        rewards_plus = np.concatenate((rewards, [0.]))
        return self.discount(rewards_plus, self.config['gamma'])[:-1]

    def update_value_function(self):
        """
        Update value network based on a set of experiences.  Both the parameters and the optimizer are
        reset at the beginning of each update.
        """
        self.v_function = LinearFeatureBaseline(**self.config['v_function'])
        observations = mpi_gather_objects(self.group_comm, self.buffer.observations)
        q_values = mpi_gather_objects(self.group_comm, self.buffer.q_values)
        dones = mpi_gather_objects(self.group_comm, self.buffer.dones)
        #print(str(self.id) + ': ' + str(len(observations)))
        if self.group_id == 0:
            self.v_function.fit(observations, q_values, dones)
        self.v_function.coeffs = mpi_gather_objects(self.group_comm, self.v_function.coeffs)[0]
        #print(str(self.id) + ' v_network coeffs: ' + str(self.v_function.coeffs[:10]))
        #sys.stdout.flush()

    def update_experiences(self):
        """  Update experiences using new value network  """
        self.buffer.values = self.v_function.predict(self.buffer.observations, self.buffer.dones)
        rewards, values, dones = self.buffer.rewards + self.buffer.max_ent, self.buffer.values, self.buffer.dones
        self.buffer.advantages = self.estimate_generalized_advantage(rewards, values, dones)
        mean_adv, std_adv = mpi_statistics_scalar(self.group_comm, self.buffer.advantages)
        self.buffer.advantages = (self.buffer.advantages - mean_adv) / std_adv  # garage doesn't do this
        self.collect_buffers()

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
                episode_next_values = np.concatenate((episode_values[1:], [0.]))  # end-of-episode
            else:
                episode_next_values = np.concatenate((episode_values[1:], self.buffer.bootstrap))
            episode_deltas = episode_rewards + self.config['gamma'] * episode_next_values - episode_values
            for start in range(len(episode_values)):
                indices = np.arange(start, len(episode_values))
                discounts = np.power(self.config['gamma'] * self.config['lambda'], indices - start)
                discounted_future_deltas = episode_deltas[start:] * discounts
                gae[start + terminals[i] + 1] = np.sum(discounted_future_deltas)
        return gae

    def compute_policy_loss(self, config):
        """  Compute policy loss, entropy, kld  """
        observations = torch.from_numpy(self.buffer.observations).float()
        actions = self.buffer.actions
        advantages = torch.from_numpy(self.buffer.advantages.astype(float)).float()
        old_log_probs = torch.from_numpy(self.buffer.log_probs.astype(float)).float()
        new_policies = self.pi_network(observations)
        if self.pi_network.config['discrete']:
            actions_one_hot = torch.from_numpy(
                np.eye(self.pi_network.config['action_dim'])[np.squeeze(actions)]).float()
            new_policies = torch.masked_fill(new_policies, new_policies < self.epsilon, self.epsilon)
            new_log_probs = torch.sum(torch.log(new_policies) * actions_one_hot, dim=1)
        else:
            new_dist = self.sampler.get_distribution(new_policies)
            actions_torch = torch.from_numpy(actions).float()
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
        if config['surrogate']:
            ratio = torch.exp(new_log_probs - old_log_probs)
            if config['clip'] > 0:
                pi_losses_1 = -advantages * ratio
                pi_losses_2 = -advantages * torch.clamp(ratio, 1.0 - config['clip'], 1.0 + config['clip'])
                pi_loss = torch.max(pi_losses_1, pi_losses_2)
            else:
                pi_loss = -advantages * ratio
        else:
            if config['clip'] > 0:
                log_diff = torch.clamp(new_log_probs - old_log_probs,
                                       np.log(1.0 - config['clip']), np.log(1.0 + config['clip']))
                pi_losses_1 = -advantages * new_log_probs
                pi_losses_2 = -advantages * (old_log_probs + log_diff)
                pi_loss = torch.max(pi_losses_1, pi_losses_2)
            else:
                pi_loss = -advantages * new_log_probs
        if self.config['e_coeff'] > 0:
            pi_loss = pi_loss + self.config['e_coeff'] * entropy
        kld = self.compute_kld(new_policies, config['kld_grad'])
        return pi_loss, entropy.item(), kld

    def compute_kld(self, policy_predictions, with_grad=True):
        if with_grad:
            new_dist = self.sampler.get_distribution(policy_predictions)
        else:
            with torch.no_grad():
                new_dist = self.sampler.get_distribution(policy_predictions)
        old_dist = self.sampler.restore_distribution(self.buffer.policies)
        return torch.distributions.kl.kl_divergence(old_dist, new_dist).mean()

    def inner_step(self, set_grad=True):
        """  Runs a single adaptation step for a given task  """
        if self.group_id == 0:
            pi_loss, _, _ = self.compute_policy_loss(self.config['inner'])
            pi_loss = torch.mean(pi_loss, dim=0)
            self.inner_optimizer.set_grads_none()
            pi_loss.backward(create_graph=set_grad)
            with torch.set_grad_enabled(set_grad):
                self.inner_optimizer.step()

    def collect_buffers(self):
        """  Collect all buffers from a group of workers onto one worker  """
        all_buffers = mpi_gather_objects(self.group_comm, self.buffer)
        if self.group_id == 0:
            fields = filter(lambda x: not x.startswith('__'), dir(self.buffer))
            for buffer in all_buffers[1:]:
                for f in fields:
                    if type(getattr(self.buffer, f)) == np.ndarray:
                        setattr(self.buffer, f, np.concatenate((getattr(self.buffer, f), getattr(buffer, f))))
        # print(str(self.id) + ' buffer rewards shape : ' + str(self.buffer.rewards.shape))
        # sys.stdout.flush()

    def update_group(self):
        """  Collect group data on worker 0, update group policy networks  """
        if self.group_id == 0:
            self.buffers.append(self.buffer)
        sync_weights(self.group_comm, self.pi_network.parameters())
        # print(str(self.id) + ' length of buffers : ' + str(len(self.buffers)))
        # sys.stdout.flush()

    def compute_meta_loss(self, set_grad=True):
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
            pi_loss, entropy, kld = self.compute_policy_loss(self.config['outer'])
            pi_loss = torch.mean(pi_loss, dim=0)
        update_module_params(self.pi_network, theta_pi)  # puts grad_fn to none, restoring leaf node
        return pi_loss, entropy, kld

    def compute_meta_gradient(self, current_params):
        """  Compute meta losses for each task, based on collected trajectories  """
        pi_gradient, entropy = None, None
        update_module_params(self.pi_network, current_params)
        if self.group_id == 0:
            pi_loss, entropy, _ = self.compute_meta_loss()
            zero_optim_grads(self.pi_optimizer)
            pi_loss.backward()
            pi_gradient = self.store_gradients()
        return pi_gradient, entropy

    '''
    def update_meta_buffer(self):
        """  Update stored log probabilities to correspond to network before adaptation  """
        with torch.no_grad():
            observations = torch.from_numpy(self.buffers[-1].observations).float()
            policies = self.pi_network(observations)
            if self.pi_network.config['discrete']:
                self.buffers[-1].policies = policies.numpy()
            else:
                self.buffers[-1].policies = np.concatenate((policies[0].numpy(), policies[1].numpy()), 1)
            dist = self.sampler.get_distribution(policies)
            actions_torch = torch.from_numpy(self.buffers[-1].actions).float()
            if self.config['bound_corr']:
                below = dist.cdf(actions_torch.clamp(-1.0, 1.0))
                below = below.clamp(self.epsilon, 1.0).log() * actions_torch.le(-1).float()
                above = torch.ones(actions_torch.size()) - dist.cdf(actions_torch.clamp(-1.0, 1.0))
                above = above.clamp(self.epsilon, 1.0).log() * actions_torch.ge(1).float()
                inner = dist.log_prob(actions_torch) * actions_torch.gt(-1).float() * actions_torch.lt(1).float()
                log_probs = torch.sum(inner + below + above, dim=-1)
            else:
                log_probs = torch.sum(dist.log_prob(actions_torch), dim=-1)
            self.buffers[-1].log_probs = log_probs.numpy()
    '''

    def store_gradients(self):
        """  Stores gradient(s) as dictionaries of numpy arrays  """
        policy_grad = {}
        for k, v in self.pi_network.named_parameters():
            if k is not None:
                policy_grad[k] = v.grad.detach().numpy().copy()
        return policy_grad

    def collect_gradients(self, policy_grads):
        """  Collect and average gradients  """
        mean_policy_grad = {}
        all_policy_grads = self.flatten_list(mpi_gather_objects(MPI.COMM_WORLD, policy_grads))
        # print(str(self.id) + ' length of policy grads: ' + str(len(all_policy_grads)))
        # sys.stdout.flush()
        if self.id == 0:
            num_grads = len(all_policy_grads)
            mean_policy_grad = {k: v / num_grads for k, v in all_policy_grads[0].items()}
            for i in range(1, num_grads):
                for k in mean_policy_grad.keys():
                    mean_policy_grad[k] += all_policy_grads[i][k] / num_grads
        return mean_policy_grad

    def outer_step(self, mean_policy_grad):
        """  Apply mean policy and value meta grads to networks based on all experiences  """
        if self.id == 0:
            zero_optim_grads(self.pi_optimizer)
            for k, v in self.pi_network.named_parameters():
                v.grad = torch.from_numpy(mean_policy_grad[k])
            self.pi_optimizer.step()
        sync_weights(MPI.COMM_WORLD, self.pi_network.parameters())

    def update_logging(self, pre_summary, post_summary, entropies, previous_steps):
        """  Update and flush logger  """
        steps = self.buffer.steps * 2
        if self.logger is None:
            self.initialize_logger()
        for k in pre_summary:
            self.logger.log_mean_value('Pre-adaptation/' + k, pre_summary[k], steps, previous_steps)
        for k in post_summary:
            self.logger.log_mean_value('Post-adaptation/' + k, post_summary[k], steps, previous_steps)
        self.logger.log_mean_value('Losses/Entropy', entropies, steps, previous_steps)
        self.logger.flush()
        steps_across_processes = mpi_sum(MPI.COMM_WORLD, steps)
        #print('update complete')
        #sys.stdout.flush()
        return steps_across_processes

    def save_networks(self, total_steps, last_checkpoint):
        """  Store meta-policy netowrk  """
        if self.id == 0:
            torch.save({'model': self.pi_network.state_dict(),
                        'optimizer': self.pi_optimizer.state_dict(),
                        'steps': total_steps},
                       os.path.join(self.config['model_folder'], 'model-latest.pt'))
            if total_steps // self.config['checkpoint_every'] > last_checkpoint:  # periodically keep checkpoint
                last_checkpoint += 1
                suffix = str(int(last_checkpoint * self.config['checkpoint_every']))
                torch.save({'model': self.pi_network.state_dict(),
                            'optimizer': self.pi_optimizer.state_dict(),
                            'steps': total_steps},
                           os.path.join(self.config['model_folder'], 'model-' + suffix + '.pt'))
        return last_checkpoint

    def initialize_environments(self):
        """  Initialize environment object at the beginning of training/testing  """
        if self.config['environment']['type'].lower() == 'metaworld':
            self.config['environment'].setdefault('stop_on_success', False)
            if self.config['environment']['task_norm_file'] is not None:
                task_scales = self.load_task_normalizations()
                for scale in task_scales:
                    self.task_scales += [scale] * 50
            self.env_collection = get_env_object(self.config['environment'])
            if self.config['use_test_set']:
                print_now(MPI.COMM_WORLD, 'Using test tasks...')
                return len(self.env_collection.test_tasks)
            else:
                print_now(MPI.COMM_WORLD, 'Using training tasks...')
                return len(self.env_collection.train_tasks)
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

    def initialize_network(self):
        """  Initialize network objects at the beginning of training/testing  """
        self.pi_network = get_network_object(self.config['pi_network'])
        if self.config['use_prior_nets']:
            checkpoint = torch.load(os.path.join(self.config['model_folder'], 'model-latest.pt'))
            self.pi_network.load_state_dict(checkpoint['model'])
            total_steps = checkpoint['steps']
            last_checkpoint = total_steps // self.config['checkpoint_every']
        else:
            total_steps, last_checkpoint = 0, 0
        sync_weights(MPI.COMM_WORLD, self.pi_network.parameters())
        theta_pi = dict(self.pi_network.named_parameters())  # only outer loop optimizes variables in place
        return total_steps, last_checkpoint, theta_pi

    def initialize_policy_optimizers(self):
        """  Initialize optimizers for inner and outer training loops  """
        self.pi_optimizer = torch.optim.Adam(params=self.pi_network.parameters(), lr=self.config['outer_lr'])
        if self.config['use_prior_nets']:
            checkpoint = torch.load(os.path.join(self.config['model_folder'], 'model-latest.pt'))
            self.pi_optimizer.load_state_dict(checkpoint['optimizer'])
        self.inner_optimizer = DifferentiableSGD(self.pi_network, self.config['inner_lr'])

    def initialize_logger(self):
        """  Initialize logger (only on one process)  """
        with open(os.path.join(self.config['model_folder'], 'config.pkl'), 'wb') as config_file:
            pickle.dump(self.config, config_file)  # store configuration
        self.logger = LoggerMPI(self.config['log_folder'])
        self.logger.log_graph(self.obs, self.pi_network)

    @staticmethod
    def concatenate_dict_of_lists(dictionary_list):
        """  Concatenate lists across a list of dictionaries  """
        new_dictionary = deepcopy(dictionary_list[0])
        for i in range(1, len(dictionary_list)):
            for k in new_dictionary.keys():
                if dictionary_list[i]:
                    new_dictionary[k] += dictionary_list[i][k]
        return new_dictionary

    @staticmethod
    def append_dict_of_lists(base_dict, new_dict):
        """  Collect a dictionary of lists  """
        for k in new_dict:
            if k in base_dict:
                base_dict[k].append(new_dict[k])
            else:
                base_dict[k] = [new_dict[k]]
        return base_dict

    @staticmethod
    def mean_of_dict_of_lists(dictionary_of_lists):
        """
        Replaces a dictionary of lists with a list containing the mean value, for each key.
        Used for fair logging across tasks.
        """
        new_dictionary = {}
        for k, v in dictionary_of_lists.items():
            new_dictionary[k] = [sum(v) / len(v)]
        return new_dictionary

    @staticmethod
    def flatten_list(input_list):
        """  Flattens a list of lists  """
        if type(input_list[0]) == list:
            return [item for sublist in input_list for item in sublist]
        else:
            return input_list

    @staticmethod
    def discount(x, gamma):
        """  Computes discounted quantity (used for discounted future reward). """
        return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


if __name__ == '__main__':
    """  Runs MAML training or testing for a given input configuration file  """
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
        maml_object = MAML(config1)
        maml_object.train()
    else:
        raise NotImplementedError('Need to implement testing')

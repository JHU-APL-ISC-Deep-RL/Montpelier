import os
import cv2
import json
import sys
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from shutil import rmtree
from mpi4py import MPI
from mp2.common.experience_buffer import ExperienceBuffer
from mp2.common.samplers import compute_entropy
from mp2.common.utils import get_env_object, get_sampler, ReducedLogging, \
    StopOnSuccess, ZeroOnSuccess, NormalizedReward
from mp2.common.mpi_data_utils import mpi_sum, sync_weights, mpi_gather_objects, \
    mpi_statistics_scalar, print_now, collect_dict_of_lists
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
        self.buffers = []
        self.total_episodes = 0

    def process_config(self):
        super().process_config()
        self.config.setdefault('tasks_per_batch', 20)
        self.config.setdefault('trajectories_per_task', 10)
        self.config.setdefault('task_norm_file', None)
        self.config.setdefault('use_test_set', False)
        self.config.setdefault('render_folder', os.path.join(self.config['model_folder'], 'videos'))
        if sys.platform[:3] == 'win':
            self.config['render_folder'] = self.config['render_folder'].replace('/', '\\')
        if self.id == 0:
            if os.path.isdir(self.config['render_folder']):
                rmtree(self.config['render_folder'], ignore_errors=True)
            Path(self.config['render_folder']).mkdir(parents=True, exist_ok=True)

    def train(self):
        """  Train neural network  """
        # Initialize relevant objects:
        self.mode = 'train'
        self.initialize_environments()
        last_checkpoint = self.initialize_networks()
        self.sampler = get_sampler(self.config['pi_network'])
        self.initialize_optimizers()
        # Run training:
        total_steps = max(last_checkpoint * self.config['checkpoint_every'], 0)
        last_evaluation = total_steps // self.config['evaluation_every'] if total_steps > 0 else -1
        while total_steps < self.config['training_frames']:
            # Collect data:
            task_list = self.sample_tasks()
            worker_tasks = task_list[self.id:len(task_list):self.num_workers]  # tasks to be trained by this worker
            policy_grads, entropies, buffers, all_episode_summaries = [], [], [], {}
            for task in worker_tasks:
                self.configure_environment_for_task(task)
                self.buffer = ExperienceBuffer()
                for _ in range(self.config['trajectories_per_task']):
                    episode_summary = self.run_trajectory()
                    if not all_episode_summaries:  # first iteration
                        all_episode_summaries = {k: [v] for k, v in episode_summary.items()}
                    else:
                        all_episode_summaries = self.concatenate_dict_of_lists(all_episode_summaries, episode_summary)
                self.buffers.append(self.buffer)
            previous_steps = total_steps
            local_steps = sum([item.steps for item in self.buffers])
            steps_current = mpi_sum(MPI.COMM_WORLD, local_steps)
            total_steps += steps_current
            # Update network(s), prepare for next batch:
            losses = self.update_network(None, None, None)
            self.buffers = []
            # Update logging, store networks:
            if total_steps // self.config['evaluation_every'] > last_evaluation:
                evaluation = self.run_evaluation()
                last_evaluation += 1
            else:
                evaluation = None
            self.update_logging(all_episode_summaries, losses, evaluation, local_steps, previous_steps)
            if self.id == 0:
                last_checkpoint = self.save_networks(total_steps, last_checkpoint)

    def initialize_environments(self):
        """  Initialize environment object at the beginning of training / testing  """
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
            self.num_tasks = len(self.task_indices)
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

    def sample_tasks(self):
        """  Sample tasks for a given iteration from list of environments  """
        task_list = []
        if self.id == 0:
            if self.config['tasks_per_batch'] <= self.num_tasks:
                tasks_to_use = np.random.choice(self.task_indices, self.config['tasks_per_batch'],
                                                replace=False)
            else:
                even_distribution = np.tile(self.task_indices, self.config['tasks_per_batch'] // self.num_tasks)
                remainder = np.random.choice(self.task_indices, self.config['tasks_per_batch'] % self.num_tasks,
                                             replace=False)
                tasks_to_use = np.concatenate((even_distribution, remainder))
            np.random.shuffle(tasks_to_use)
            task_list = tasks_to_use.tolist()
        task_list = MPI.COMM_WORLD.bcast(task_list, root=0)
        return task_list

    def configure_environment_for_task(self, task):
        """  Load training environment for a given task  """
        self.current_task = task
        if self.config['environment']['type'].lower() == 'metaworld':
            class_index = self.current_task // 50
            if self.config['use_test_set']:
                env = list(self.env_collection.test_classes.values())[class_index]()
                env.set_task(self.env_collection.test_tasks[self.current_task])
            else:
                env = list(self.env_collection.train_classes.values())[class_index]()
                env.set_task(self.env_collection.train_tasks[self.current_task])
            if self.config['environment']['stop_on_success']:
                self.env = StopOnSuccess(env)
            elif self.config['environment']['zero_on_success']:
                self.env = ZeroOnSuccess(env)
            else:
                self.env = ReducedLogging(env)
            if self.config['task_norm_file'] is not None:
                self.env = NormalizedReward(self.env, self.task_scales[self.current_task])
            if self.config['environment']['normalize_reward'] > 0:
                self.env = NormalizedReward(self.env, self.config['environment']['normalize_reward'])
        else:
            raise NotImplementedError('Environment type not yet implemented.')

    def run_trajectory(self, random_seed=None):
        """  Run trajectories based on current network(s)  """
        trajectory_buffer, trajectory_info = np.array([]).reshape(0, 7), {}
        if random_seed is not None:
            self.env.seed(random_seed)  # for testing
        self.obs = self.env.reset()
        num_frames = 0
        writer = None
        if self.config['render']:
            env_name = str(self.env.unwrapped)[7:-15]  # it seems that metaworld doesn't properly populate the name
            writer = cv2.VideoWriter(
                os.path.join(self.config['render_folder'], f'{env_name}_{self.id}_{self.total_episodes}.avi'),
                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1920, 1080))
        while True:
            policy, value = self.forward_pass()
            action, log_prob = self.sampler.get_action_and_log_prob(policy)
            output_obs, reward, done, info = self.env.step(action)
            if self.config['log_info']:
                self.concatenate_dict_of_lists(trajectory_info, info)
            if self.config['render']:
                # camera_name should be one of: corner3, corner, corner2, topview, gripperPOV, behindGripper
                frame = np.array(self.env.sim.render(1920, 1080, mode='offscreen', camera_name='corner')[:, :, ::-1])
                cv2.putText(frame, env_name, (840, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (175, 0, 0), 4)
                if info['success'][0] > 0:
                    cv2.putText(frame, 'Success', (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    writer.write(frame)
                    writer.write(frame)
                    writer.write(frame)
                    writer.write(frame)
                    writer.write(frame)
                    writer.write(frame)
                    writer.write(frame)
                    writer.write(frame)
                    writer.write(frame)
                writer.write(frame)
            num_frames += 1
            if num_frames == self.config['max_ep_length']:
                done = True
            if self.config['render'] and info['success'][0] > 0:
                done = True
            trajectory_buffer = self.update_trajectory_buffer(trajectory_buffer, action, reward, policy,
                                                              log_prob, value, done)
            if done:
                self.buffer.bootstrap = np.zeros(1)
                episode_summary = self.process_trajectory(trajectory_buffer, trajectory_info)
                break
            else:
                self.obs = output_obs
        return episode_summary

    def process_trajectory(self, trajectory_buffer, trajectory_info):
        """  Processes a completed trajectory, storing required data in buffer and return ep  """
        if self.mode == 'train':
            max_entropy_term = self.compute_max_entropy_term(trajectory_buffer)
            q_values = self.compute_target_values(trajectory_buffer[:, 2] + max_entropy_term)
            task_ids = self.current_task * np.ones(q_values.shape)
            self.buffer.update(trajectory_buffer, q_values, task_ids=task_ids, max_ent=max_entropy_term)
        entropies = [compute_entropy(trajectory_buffer[i, 3]) for i in range(trajectory_buffer.shape[0])]
        return {'task_id': self.current_task, 'episode_reward': np.sum(trajectory_buffer[:, 2]),
                'episode_length': trajectory_buffer.shape[0],
                'episode_mean_value': np.mean(trajectory_buffer[:, 5]), 'episode_entropy': np.mean(entropies),
                **{k: self.flatten_list(v)[-1] for k, v in trajectory_info.items()}}

    def update_network(self, indices, starts, stops):
        """  Updates the networks based on processing from all workers  """
        for epoch in range(self.config['opt_epochs']):
            for i in range(self.config['train_v_iter']):  # update value function
                v_grads = self.compute_v_gradients()
                mean_v_grad = self.collect_gradients(v_grads)
                if self.id == 0:
                    self.v_optimizer.zero_grad()
                    for k, v in self.v_network.named_parameters():
                        v.grad = torch.from_numpy(mean_v_grad[k])
                    self.v_optimizer.step()
                sync_weights(MPI.COMM_WORLD, self.v_network.parameters())
            self.update_all_values()
            self.estimate_advantages()
            all_entropies = []
            for i in range(self.config['train_pi_iter']):
                pi_grads, entropies, klds = self.compute_pi_gradients()
                all_entropies += self.flatten_list(mpi_gather_objects(MPI.COMM_WORLD, entropies))
                mean_pi_grad = self.collect_gradients(pi_grads)
                mean_kld = np.mean(self.flatten_list(mpi_gather_objects(MPI.COMM_WORLD, klds)))
                if mean_kld > self.config['max_kl'] > 0:
                    if self.id == 0:
                        print_now(MPI.COMM_WORLD, 'Policy KL divergence exceeds limit; stopping update at step %d.' % i)
                    break
                if self.id == 0:
                    self.pi_optimizer.zero_grad()
                    for k, v in self.pi_network.named_parameters():
                        v.grad = torch.from_numpy(mean_pi_grad[k])
                    self.pi_optimizer.step()
                sync_weights(MPI.COMM_WORLD, self.pi_network.parameters())
            return {'entropies': all_entropies}

    def compute_v_gradients(self):
        """  Computes gradients of value loss for each task  """
        v_grads = []
        for buffer in self.buffers:
            self.v_optimizer.zero_grad()
            observations = torch.from_numpy(np.vstack(buffer.observations)).float()
            target_values = torch.from_numpy(buffer.q_values.astype(float)).float()
            v_loss = self.compute_value_loss(observations, target_values)
            v_loss.backward()
            v_grads.append(self.store_gradients(self.v_network))
        return v_grads

    def update_all_values(self):
        """  Estimate values with updated value network, store in buffer  """
        for buffer in self.buffers:
            observations = torch.from_numpy(np.vstack(buffer.observations)).float()
            buffer.values = self.v_network(observations).view(-1).detach().numpy()

    def estimate_advantages(self):
        """  Estimate advantages for a sequence of observations and rewards  """
        all_advantages = np.array([])
        for buffer in self.buffers:
            rewards, values, dones = buffer.rewards + buffer.max_ent, buffer.values, buffer.dones
            buffer.advantages = self.estimate_generalized_advantage(rewards, values, dones)
            all_advantages = np.concatenate((all_advantages, buffer.advantages))
        mean_adv, std_adv = mpi_statistics_scalar(MPI.COMM_WORLD, all_advantages)
        for buffer in self.buffers:
            buffer.advantages = (buffer.advantages - mean_adv) / std_adv

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
            pi_loss, entropy, kld = self.compute_policy_loss(observations, actions, advantages, log_probs, policies)
            pi_loss = torch.mean(pi_loss, dim=0)
            pi_loss.backward()
            entropies.append(entropy.item())
            klds.append(kld.item())
            pi_grads.append(self.store_gradients(self.pi_network))
        return pi_grads, entropies, klds

    def update_logging(self, episode_summaries, losses, evaluation, steps, previous_steps):
        if self.logger is None:
            self.initialize_logging()
        super().update_logging(episode_summaries, losses, evaluation, steps, previous_steps)

    def collect_gradients(self, grads):
        """  Collect and average gradients  """
        grad_list = self.flatten_list(mpi_gather_objects(MPI.COMM_WORLD, grads))
        num_grads = len(grad_list)
        mean_grad = {k: v * 0 for k, v in grad_list[0].items()}
        for k in mean_grad:
            for grad in grad_list:
                mean_grad[k] += grad[k] / num_grads
        return mean_grad

    def test(self):
        """  Train neural network  """
        # Initialize relevant objects:
        self.mode = 'test'
        self.initialize_environments()
        self.initialize_networks()
        self.sampler = get_sampler(self.config['pi_network'], deterministic=True)
        local_episodes, self.total_episodes, local_episode_summaries = 0, 0, {}
        self.pi_network.eval()
        self.v_network.eval()
        while self.total_episodes < self.config['test_episodes']:
            task = np.random.choice(np.arange(self.num_tasks))
            self.configure_environment_for_task(task)
            self.buffer = ExperienceBuffer()  # reset experience buffer
            episode_summary = self.run_trajectory()
            if len(local_episode_summaries.keys()) == 0:  # first iteration
                local_episode_summaries = {k: [v] for k, v in episode_summary.items()}
            else:
                local_episode_summaries = self.concatenate_dict_of_lists(local_episode_summaries, episode_summary)
            local_episodes += 1
            self.total_episodes = int(mpi_sum(MPI.COMM_WORLD, local_episodes))
            if self.id == 0:
                print_now(MPI.COMM_WORLD, str(self.total_episodes) + ' episodes complete.')
            # Collect, process, save data:
        test_output = collect_dict_of_lists(MPI.COMM_WORLD, local_episode_summaries)
        self.store_test_results(test_output)
        return test_output

    @staticmethod
    def store_gradients(network):
        """  Stores gradient(s) as dictionaries of numpy arrays  """
        network_grad = {}
        for k, v in network.named_parameters():
            network_grad[k] = v.grad.detach().numpy().copy()
        return network_grad

    @staticmethod
    def flatten_list(input_list):
        """  Flattens a list of lists  """
        return [item for sublist in input_list for item in sublist]


if __name__ == '__main__':
    """  Runs MTPolicyOptimizer training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    parser.add_argument('--test_ep', help='number of test episodes', required=False, type=int, default=1000)
    parser.add_argument('--render', help='render', required=False, type=int, default=0)
    in_args = parser.parse_args()
    with open(os.path.join(os.getcwd(), in_args.config), 'r') as f1:
        config1 = json.load(f1)
    config1['seed'] = in_args.seed
    config1['test_episodes'] = in_args.test_ep
    config1['render'] = bool(in_args.render)
    if in_args.mode.lower() == 'train':
        mtpo_object = MTPolicyOptimizer(config1)
        mtpo_object.train()
    else:
        config1['use_prior_nets'] = True
        mtpo_object = MTPolicyOptimizer(config1)
        mtpo_object.test()

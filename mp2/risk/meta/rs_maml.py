import argparse
import os
import json
from mpi4py import MPI
from mp2.base.meta.maml import MAML
from mp2.common.mpi_data_utils import mpi_gather_objects
from mp2.risk.dist.weight_coefficients import load_weight_coefficient_function


class WeightedMAML(MAML):
    def __init__(self, config):
        super().__init__(config)
        # assert self.num_workers >= self.config['tasks_per_update'], 'Not enough workers'
        self.weight_coefficient_function = load_weight_coefficient_function(self.config['weight'])

    def process_config(self):
        """  Processes input configuration  """
        self.config['inner'] = {'surrogate': False, 'clip': -1, 'kld_grad': False}  # vpg
        self.config['outer'] = {'surrogate': False, 'clip': 0.5, 'kld_grad': False}  # c3po
        super().process_config()

    def compute_sample_weights(self, task, task_list, trajectory_summaries):
        # Gather all the rewards, weight the samples, and broadcast the result
        rewards = trajectory_summaries["episode_reward"]
        if self.id == 0:
            # Compute weights for each sampled gradient
            rewards_list = self.flatten_list(mpi_gather_objects(MPI.COMM_WORLD, rewards))
            weights_list = self.weight_coefficient_function(rewards_list)
        else:
            weights_list = None
        weights_list = MPI.COMM_WORLD.bcast(weights_list, root=0)

        # Now weights list is weight per sample for all samples, but need to return just weights for this group's tasks
        weights_for_this_task = [
            weight
            for weight, task_of_weight in zip(weights_list, task_list)
            if task_of_weight == task
        ]
        return weights_for_this_task


if __name__ == '__main__':
    """  Runs MAML training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    in_args = parser.parse_args()
    with open(os.path.join(os.getcwd(), in_args.config), 'r') as f1:
        config1 = json.load(f1)
    config1['seed'] = in_args.seed
    if in_args.mode.lower() == 'train':
        config1['use_test_set'] = False
        maml_object = WeightedMAML(config1)
        maml_object.train()
    else:
        raise NotImplementedError('Need to implement testing')
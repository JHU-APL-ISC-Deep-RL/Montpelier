import argparse
import os
import json
from mpi4py import MPI
from mp2.common.mpi_data_utils import mpi_gather_objects
from mp2.common.gradient_surgery import gradient_surgery
from .task_processor import TaskProcessor
from .maml import MAML


class SelectiveMAML(MAML):
    """  MAML modified to choose tasks based on pairwise metrics computed among most recent task gradients  """
    def __init__(self, config: dict):
        super().__init__(config)
        self.task_processor = None  # for sampling tasks, gradient surgery- constructed with environment initialization
        self.task_list = []         # for metric bookkeeping

    def initialize_environments(self) -> int:
        """
        Initializes environment object at the beginning of training/testing and task_metrics object at the
        beginning of training.
        """
        num_tasks = super().initialize_environments()
        self.task_processor = TaskProcessor(self.config['task_processor'], num_tasks, self.config['tasks_per_update'])
        return num_tasks

    def sample_tasks(self, num_tasks: int) -> list:
        """  Samples tasks, possibly intelligently, for a given iteration from list of environments  """
        task_list = []
        if self.id == 0:
            task_list = self.task_processor.sample_tasks()
        task_list = MPI.COMM_WORLD.bcast(task_list, root=0)
        self.task_list = task_list
        return task_list

    def collect_gradients(self, policy_grads: list) -> dict:
        """  Collect and average gradients (only runs on one worker)  """
        policy_grad_list = self.flatten_list(mpi_gather_objects(MPI.COMM_WORLD, policy_grads))
        sorted_tasks = self.update_task_grads(policy_grad_list)
        if self.config['gradient_surgery']:
            if self.config['task_processor']['ext_gradient_surgery']:  # todo: non-essential, but could re-factor
                policy_grad_list = self.task_processor.gradient_surgery(policy_grad_list, sorted_tasks)
            else:
                policy_grad_list = gradient_surgery(policy_grad_list)
        num_grads = len(policy_grad_list)
        mean_policy_grad = {k: v * 0 for k, v in policy_grad_list[0].items()}
        for k in mean_policy_grad:
            for grad in policy_grad_list:
                mean_policy_grad[k] += grad[k] / num_grads
        return mean_policy_grad

    def update_task_entropies(self, entropies: list):
        """  Collect task entropies, store in task sampler  """
        entropy_list = self.flatten_list(mpi_gather_objects(MPI.COMM_WORLD, entropies))
        sorted_tasks = []
        for i in range(self.num_workers):
            sorted_tasks += self.task_list[i:len(self.task_list):self.num_workers]
        if 'entropies' in self.task_processor.config['criteria']:
            self.task_processor.update_entropies(entropy_list, sorted_tasks)

    def update_task_grads(self, policy_grad_list: list):
        """  Sort policy gradients by task, store in task processor, update metrics  """
        sorted_tasks = []
        for i in range(self.num_workers):
            sorted_tasks += self.task_list[i:len(self.task_list):self.num_workers]
        if not self.task_processor.initialized:
            self.task_processor.initialize_params_grads(policy_grad_list[0])
        self.task_processor.update_gradients(policy_grad_list, sorted_tasks)
        return sorted_tasks

    def test(self):
        self.config['criteria'] = []
        super().test()

    def update_logging(self, pre_adapted_summary, post_adapted_summary, entropies, steps, previous_steps):
        """  Update and flush logger  """
        super().update_logging(pre_adapted_summary, post_adapted_summary, entropies, steps, previous_steps)
        if 'entropies' in self.config['task_processor']['criteria']:
            self.update_task_entropies(entropies)


if __name__ == '__main__':
    """  Runs SelectiveMAML training or testing for a given input configuration file  """
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
        maml_object = SelectiveMAML(config1)
        maml_object.train()
    else:
        config1['use_prior_nets'] = True
        maml_object = SelectiveMAML(config1)
        maml_object.test()

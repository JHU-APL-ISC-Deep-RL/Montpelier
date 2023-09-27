import numpy as np


class ExperienceBuffer(object):
    """  On-policy experience buffer.  Not all fields are used for all types of learning.  """
    def __init__(self):
        self.observations = np.array([])
        self.actions = np.array([])
        self.rewards = np.array([])
        self.values = np.array([])
        self.dones = np.array([])
        self.q_values = np.array([])
        self.advantages = np.array([])
        self.policies = np.array([])
        self.log_probs = np.array([])
        self.utilities = np.array([])
        self.costs = np.array([])
        self.episode_costs = np.array([])
        self.c_q_values = np.array([])
        self.c_utilities = np.array([])
        self.cost_values = np.array([])
        self.cost_advantages = np.array([])
        self.task_ids = np.array([])
        self.task_groups = np.array([])
        self.steps = 0
        self.trajectories = 0
        self.current_episode = np.array([])
        self.weights = np.array([])
        self.max_ent = np.array([])
        self.bootstrap = [0.]

    def update(self, trajectory_buffer, q_values=None, advantages=None, utilities=None,
               c_q_values=None, c_utilities=None, task_ids=None, cluster=50, max_ent=None):
        """  Add a new experience to the buffer  """
        self.steps += trajectory_buffer.shape[0]
        if self.current_episode.shape[0] == 0 or self.dones[-1]:
            self.current_episode = trajectory_buffer
        else:
            self.current_episode = np.concatenate((self.current_episode, trajectory_buffer))
        if len(self.observations) == 0:
            self.observations = np.vstack(trajectory_buffer[:, 0])
            self.actions = np.vstack(trajectory_buffer[:, 1])
            self.policies = np.vstack(trajectory_buffer[:, 3])
        else:
            self.observations = np.concatenate((self.observations, np.vstack(trajectory_buffer[:, 0])))
            self.actions = np.concatenate((self.actions, np.vstack(trajectory_buffer[:, 1])))
            self.policies = np.concatenate((self.policies, np.vstack(trajectory_buffer[:, 3])))
        self.rewards = np.concatenate((self.rewards, trajectory_buffer[:, 2]))
        self.log_probs = np.concatenate((self.log_probs, trajectory_buffer[:, 4]))
        self.values = np.concatenate((self.values, trajectory_buffer[:, 5]))
        self.dones = np.concatenate((self.dones, trajectory_buffer[:, 6]))
        if trajectory_buffer.shape[1] > 7:
            self.costs = np.concatenate((self.costs, trajectory_buffer[:, 7]))
        self.trajectories = np.sum(self.dones)
        if max_ent is not None:
            self.max_ent = np.concatenate((self.max_ent, max_ent))  # only needed in training
        if q_values is not None:
            self.q_values = np.concatenate((self.q_values, q_values))  # only needed in training
        if advantages is not None:  # computation of advantage may not be in trajectory runner
            self.advantages = np.concatenate((self.advantages, advantages))
        if utilities is not None:  # don't need this for standard learning
            self.utilities = np.concatenate((self.utilities, utilities))
        if c_q_values is not None:
            self.c_q_values = np.concatenate((self.c_q_values, c_q_values))  # only needed in constrained training
        if c_utilities is not None:  # don't need this for standard learning
            self.c_utilities = np.concatenate((self.c_utilities, c_utilities))
        if task_ids is not None:
            if cluster > 0:  # for clustering into supertasks; currently configured for Meta-World
                group_ids = (task_ids // cluster)
                self.task_groups = np.concatenate((self.task_groups, group_ids)).astype(int)
            self.task_ids = np.concatenate((self.task_ids, task_ids)).astype(int)  # only need for MT / ML
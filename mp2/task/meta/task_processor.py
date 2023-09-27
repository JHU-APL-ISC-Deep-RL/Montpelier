import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class TaskProcessor(object):
    """
    Object for computing gradient-based task metrics and doing extended gradient surgery.  Currently includes recency,
    entropy, and metrics from Gradient Surgery paper; can add more as needed.
    """
    def __init__(self, config: dict, num_tasks: int, tasks_per_update: int):
        self.config = config                                      # configuration for task selection
        self.process_config(num_tasks, tasks_per_update)          # finalize configuration
        self.gradients = []                                       # np.array, num_tasks x gradient_dim
        self.parameter_list = []                                  # for preserving param order (not using OrderedDict)
        self.parameter_shapes = []                                # network parameter shapes
        self.gradients_estimated = np.zeros((num_tasks,))         # keeps track of where we have gradient estimates
        self.recency = np.zeros((num_tasks,))                     # how recently a task was sampled
        self.raw_entropies = np.zeros((num_tasks,))               # raw entropies from meta-update
        self.entropies = np.zeros((num_tasks,))                   # average policy entropy of last sampled batch
        self.cos_grad_ang = np.zeros((num_tasks, num_tasks))      # cosines of pairwise gradient angles
        self.grad_mag_sim = np.zeros((num_tasks, num_tasks))      # pairwise gradient magnitude similarities
        self.multi_task_curv = np.zeros((num_tasks, num_tasks))   # pairwise multi-task curvatures
        self.initialized = False                                  # whether or not network architecture is loaded
        self.supertask_entropies = np.zeros((self.config['supertasks'],))  # supertask entropies
        self.entropy_history = np.zeros((num_tasks, self.config['entropy_limit']))  # last few recorded entropies

    def process_config(self, num_tasks, tasks_per_update):
        """  Processes input configuration dictionary  """
        self.config['num_tasks'] = num_tasks
        self.config['tasks_per_update'] = tasks_per_update
        self.config.setdefault('ext_gradient_surgery', True)
        self.config.setdefault('criteria', [])
        self.config.setdefault('grad_limit', 5)
        self.config.setdefault('entropy_limit', 5)
        # self.config.setdefault('use_entropy_history', True)
        self.config.setdefault('hierarchy', 50)  # number of tasks per supertask in a hierarchy (e.g. for Meta-World)
        self.config.setdefault('supertasks', 1)
        self.config.setdefault('supertask_assignment_from_clustering', 0)
        self.config.setdefault('frequencies', [3, 3, 3, 2, 2, 2, 2, 1, 1, 1])  # [4, 4, 3, 3, 2, 2, 1, 1] ?
        if len(self.config['criteria']) > 0:
            if self.config['hierarchy'] > 0:
                self.config['supertasks'] = self.config['num_tasks'] // self.config['hierarchy']
                assert self.config['tasks_per_update'] > self.config['supertasks'], "need batch size > # supertasks"
                assert self.config['num_tasks'] % self.config['hierarchy'] == 0, "need even numbers of subtasks"
                assert len(self.config['frequencies']) == self.config['supertasks'], "hierarchy incorrectly defined"
                assert sum(self.config['frequencies']) == self.config['tasks_per_update'], "frequencies inconsistent"
                if self.config['criteria'][0] == 'recency':
                    assert self.config['tasks_per_update'] % self.config['supertasks'] == 0, \
                        "Uneven task numbers per supertask not supported with recency criterion."  # todo
                    self.config['frequencies'] = [self.config['tasks_per_update'] / self.config['supertasks']] \
                        * self.config['supertasks']
            self.config['frequencies'] = np.array(self.config['frequencies']).astype(int)
            if self.config['tasks_per_update'] > self.config['num_tasks']:
                self.config['criteria'] = []  # just sample randomly

    def initialize_params_grads(self, initial_gradient: dict):
        """  Initializes object parameter list and gradient matrix  """
        self.parameter_list = list(initial_gradient.keys())
        flattened_initial_gradient = self.flatten_gradient(initial_gradient)
        self.gradients = np.zeros((self.config['num_tasks'], flattened_initial_gradient.shape[0]))
        self.initialized = True

    def update_gradients(self, new_gradients: list, task_indices: list):
        """  Update stored task gradient and recency of sampling  """
        self.recency += 1
        for ind, grad in zip(task_indices, new_gradients):
            self.gradients[ind] = self.flatten_gradient(grad)
            self.gradients_estimated[ind] = 1
            self.recency[ind] = 0

    def update_entropies(self, new_entropies: list, task_indices: list):
        """  Update stored task entropies (both raw and processed)  """
        for ind, entropy in zip(task_indices, new_entropies):
            self.raw_entropies[ind] = entropy
        # Process entropy estimates for task selection:
        stale = np.logical_or(self.recency >= self.config['entropy_limit'], self.gradients_estimated == 0).astype(int)
        fresh = 1 - stale
        # self.entropy_history = np.concatenate((self.entropy_history, self.raw_entropies), axis=1)[:, 1:]
        if self.config['hierarchy'] <= 0 or self.config['criteria'][0] != 'entropies':  # "stale" get maximal entropy
            upper = max(self.raw_entropies) + 1
            self.entropies = self.raw_entropies*fresh + upper*stale
        else:  # average available "fresh" entropy estimates in each superclass
            for i in range(self.config['supertasks']):
                ind = list(range(i * self.config['hierarchy'], (i + 1) * self.config['hierarchy']))
                self.supertask_entropies[i] = np.sum(self.raw_entropies[ind] * fresh[ind]) / np.sum(fresh[ind])
                self.entropies[ind] = self.supertask_entropies[i]

    def sample_tasks(self):
        """  Sample tasks based on configured criteria  """
        task_indices = np.arange(self.config['num_tasks'])
        if self.initialized and len(self.config['criteria']) > 0:  # use criteria-based sampling
            if self.config['hierarchy'] <= 0:  # individual processing
                task_scores = np.zeros((len(self.config['criteria']) + 1, self.config['num_tasks']))
                task_scores[0] = np.random.rand(self.config['num_tasks'])
                for j, criterion in enumerate(reversed(self.config['criteria'])):
                    task_scores[j + 1] = getattr(self, criterion)
                tasks_to_use = np.lexsort(-task_scores)[:self.config['tasks_per_update']].tolist()
            else:  # hierarchical processing
                tasks_to_use = []
                supertasks_selected = []
                if self.config['criteria'][0] == 'entropies':
                    frequencies = np.zeros(self.config['frequencies'].shape)
                    frequencies[np.flip(np.argsort(self.supertask_entropies))] = self.config['frequencies']
                    frequencies = frequencies.astype(int)
                else:
                    frequencies = self.config['frequencies']

                if self.config["supertask_assignment_from_clustering"]:
                    supertask_indices = self.supertask_clusters()
                else:
                    supertask_indices = []
                    for i in range(self.config['supertasks']):
                        ind = np.arange(i * self.config['hierarchy'], (i + 1) * self.config['hierarchy'])
                        supertask_indices.append(ind)
                for i in range(self.config['supertasks']):
                    ind = supertask_indices[i]
                    task_scores = np.zeros((len(self.config['criteria']) + 1, ind.shape[0]))
                    task_scores[0] = np.random.rand(ind.shape[0],)
                    for j, criterion in enumerate(reversed(self.config['criteria'])):
                        task_scores[j + 1] = getattr(self, criterion)[ind]
                    intra_super_task_rankings = ind[np.lexsort(-task_scores)]
                    new_tasks_to_use = intra_super_task_rankings[:frequencies[i]].tolist()
                    tasks_to_use += new_tasks_to_use
                    supertasks_selected += [i for _ in new_tasks_to_use]
                # print(supertasks_selected)

        else:  # revert to uniform sampling (either over whole set or spread evenly over supertasks)
            if self.config['tasks_per_update'] <= self.config['num_tasks']:
                if self.initialized or len(self.config['criteria']) == 0 or self.config['hierarchy'] <= 0:
                    # pick randomly from all tasks
                    tasks_to_use = np.random.choice(task_indices, self.config['tasks_per_update'], replace=False)
                    tasks_to_use = tasks_to_use.tolist()
                else:
                    # choose uniform number from each supertask, sample randomly to get individual tasks
                    tasks_to_use = []
                    frequency = self.config['tasks_per_update'] // self.config['supertasks']
                    for i in range(self.config['supertasks']):
                        ind = list(range(i * self.config['hierarchy'], (i + 1) * self.config['hierarchy']))
                        tasks_to_use += np.random.choice(ind, frequency).tolist()
            else:
                even_distribution = np.tile(task_indices, self.config['tasks_per_update'] // self.config['num_tasks'])
                remainder = np.random.choice(task_indices, self.config['tasks_per_update'] % self.config['num_tasks'],
                                             replace=False)
                tasks_to_use = np.concatenate((even_distribution, remainder)).tolist()
        return sorted(tasks_to_use)

    def gradient_surgery(self, grad_list, task_indices):
        """  Performs gradient surgery, subject to gradient estimates going back a configurable number of batches  """
        grad_list_pc = [self.flatten_gradient(grad) for grad in grad_list]
        num_grads = len(grad_list)
        ind = np.where(((self.gradients_estimated > 0)*(self.recency < self.config['grad_limit'])) > 0)[0]
        for i in range(num_grads):
            np.random.shuffle(ind)
            for j in range(len(ind)):
                if ind[j] == task_indices[i]:
                    continue
                dot_ij = np.dot(grad_list_pc[i], self.gradients[ind[j]])
                if dot_ij < 0:
                    norm_sq_j = np.linalg.norm(self.gradients[ind[j]]) ** 2
                    grad_list_pc[i] -= dot_ij / norm_sq_j * self.gradients[ind[j]]
        return [self.expand_gradient(grad) for grad in grad_list_pc]

    def flatten_gradient(self, gradient) -> np.array:
        """  Flattens a gradient dictionary into a vector  """
        flat_gradient = np.array([])
        for k in self.parameter_list:
            self.parameter_shapes.append(gradient[k].shape)
            flat_gradient = np.concatenate((flat_gradient, gradient[k].reshape(-1)))
        return flat_gradient

    def expand_gradient(self, flattened_gradient):
        """  Re-expand a flattened gradient  """
        expanded_gradient = {}
        start = 0
        for i, k in enumerate(self.parameter_list):
            stop = start + np.prod(self.parameter_shapes[i])
            expanded_gradient[k] = np.reshape(flattened_gradient[start:stop],
                                              self.parameter_shapes[i]).astype(np.float32)
            start = stop
        return expanded_gradient

    def update_cos_grad_angles(self, task_indices=()):
        """  Update the table of cosines of the angles between the flattened gradients  """
        pairs = self.get_unique_pairs(task_indices)
        for ind1, ind2 in pairs:
            self.cos_grad_ang[ind1, ind2] = np.sum(self.gradients[ind1]*self.gradients[ind2]) / \
                np.linalg.norm(self.gradients[ind1]) / np.linalg.norm(self.gradients[ind2])
            self.cos_grad_ang[ind2, ind1] = self.grad_cos_ang[ind1, ind2]

    def update_grad_mag_similarities(self, task_indices=()):
        """  Update the table of gradient magnitude similarities  """
        pairs = self.get_unique_pairs(task_indices)
        for ind1, ind2 in pairs:
            norm1 = np.linalg.norm(self.gradients[ind1])
            norm2 = np.linalg.norm(self.gradients[ind2])
            self.grad_mag_sim[ind1, ind2] = 2*norm1*norm2/(norm1**2 + norm2**2)
            self.grad_mag_sim[ind2, ind1] = self.grad_mag_sim[ind1, ind2]

    def update_multi_task_curvatures(self, task_indices=()):
        """  Update the table of multi-task curvatures  """
        pairs = self.get_unique_pairs(task_indices)
        for ind1, ind2 in pairs:
            multi_task_grad = (self.gradients[ind1] + self.gradients[ind1])/2
            self.multi_task_curv[ind1, ind2] = ...  # todo (possibly); would need second derivative
            self.multi_task_curv[ind2, ind1] = self.multi_task_curv[ind1, ind2]

    def get_pairs(self, task_indices=()) -> set:
        """  Get unique pairs of tasks  todo: this method might not be needed """
        if len(task_indices) == 0:
            task_indices = range(self.config['num_tasks'])  # if not told otherwise, loop over all tasks
        return set([tuple(sorted([i, j])) for i in task_indices for j in range(self.config['num_tasks'])])

    def supertask_clusters(self):
        """ cluster gradients into S clusters, S=number of supertasks """
        # PCA to retrieve primary N dimensions
        pca = PCA(n_components=100)
        reduced_grads = pca.fit_transform(self.gradients)
        # feed these into kmeans
        means = KMeans(n_clusters=self.config['supertasks']).fit(np.asarray(reduced_grads))
        # bin the tasks into their respective clusters
        bins = [[] for _ in range(self.config['supertasks'])]
        for i in range(len(reduced_grads)):
            bins[kmeans.labels_[i]].append(i)
        return bins

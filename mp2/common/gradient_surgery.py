from copy import deepcopy
import numpy as np


def gradient_surgery(grad_list: list) -> list:
    """  Perform gradient surgery (i.e. remove conflicts) among gradients for different tasks  """    
    grad_list_pc = deepcopy(grad_list)
    num_grads = len(grad_list)
    ind = list(range(num_grads))
    for i in range(num_grads):
        np.random.shuffle(ind)
        for j in range(num_grads):
            if ind[j] == i:
                continue
            dot_ij = sum([np.sum(grad_list_pc[i][k] * grad_list[ind[j]][k]) for k in grad_list_pc[i].keys()])
            if dot_ij < 0:
                norm_sq_j = sum([np.sum(grad_list[ind[j]][k]**2) for k in grad_list[ind[j]].keys()])
                for k in grad_list_pc[i].keys():
                    grad_list_pc[i][k] -= dot_ij / norm_sq_j * grad_list[ind[j]][k]
    return grad_list_pc

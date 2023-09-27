import numpy as np


class LinearFeatureBaseline(object):
    """  Linear value function that takes features as an input; largely copied from garage  """

    def __init__(self, reg_coeff=1e-5, lower_bound=-10, upper_bound=10):
        self.coeffs = np.array([])
        self._reg_coeff = reg_coeff
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def process_features(self, observations, dones):
        """  Get paths, compute features  """
        terminals = np.nonzero(dones)[0]
        terminals = list(np.concatenate((np.array([-1]), terminals)))
        if not dones[-1]:
            terminals += [dones.shape[0] - 1]
        features = []
        for i in range(len(terminals[:-1])):
            obs = np.clip(observations[terminals[i] + 1:terminals[i + 1] + 1], self.lower_bound, self.upper_bound)
            n_experiences = obs.shape[0]
            al = np.arange(n_experiences).reshape(-1, 1) / 100.0
            features.append(np.concatenate([obs, obs**2, al, al**2, al**3, np.ones((n_experiences, 1))], axis=1))
        return np.concatenate(features)

    def fit(self, observations, q_values, dones):
        """  Fit regressor based on provided data  """
        featmat = self.process_features(np.concatenate(observations).astype(np.float),
                                        np.concatenate(dones).astype(int))
        returns = np.concatenate(q_values).astype(np.float)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self.coeffs = np.linalg.lstsq(featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                                          featmat.T.dot(returns), rcond=-1)[0]
            if not np.any(np.isnan(self.coeffs)):
                break
            reg_coeff *= 10

    def predict(self, observations, dones):
        """  Predict value based on observations in buffer  """
        if self.coeffs.shape[0] == 0:
            return np.zeros(observations.shape[0])
        features = self.process_features(observations.astype(np.float), dones.astype(int))
        return features.dot(self.coeffs)

import argparse
import os
import json
from mp2.base.meta.promp import ProMP
from mp2.risk.meta.weighted_maml import WeightedMAML


class WeightedProMP(ProMP, WeightedMAML):
    def process_config(self):
        super().process_config()
        self.config['inner']['surrogate'] = False
        self.config['outer']['surrogate'] = False


if __name__ == '__main__':
    """  Runs ProMP training or testing for a given input configuration file  """
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
        promp_object = WeightedProMP(config1)
        promp_object.train()
    else:
        raise NotImplementedError('Need to implement testing')
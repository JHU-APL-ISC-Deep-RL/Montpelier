#!/bin/bash

python mt_policy_optimizer.py --config ../../../configs/mtpo_base_ne_corr.json --seed 0 --mode test --test_ep 25 --render 1
python mt_policy_optimizer.py --config ../../../configs/mtpo_base_ne_corr.json --seed 1 --mode test --test_ep 25 --render 1
python mt_policy_optimizer.py --config ../../../configs/mtpo_base_ne_corr.json --seed 2 --mode test --test_ep 25 --render 1
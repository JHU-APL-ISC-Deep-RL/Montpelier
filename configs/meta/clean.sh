#!/bin/bash

#sed -i '' 's/"train_v_iter": 1,/"train_v_iter": 10,/g' */*.json
#sed -i '' 's/"v_lr" : 0.01,/"v_lr" : 0.001,/g' */*.json
#sed -i '' 's/"inner_lr" : 0.01,/"inner_lr" : 0.0001,/g' */*.json

#sed -i '' 's/output_path/model_folder/g' */*.json
#sed -i '' 's/log_path/log_folder/g' */*.json


#sed -i '' 's/"e_coeff" : 0,/"e_coeff" : 0.00005,/g' */*.json
#sed -i '' 's/20000000/10000000/g' */*.json
#sed -i '' 's/e_coeff/max_entropy/g' */*.json

sed -i '' 's/"max_entropy" : 0.00005,/"max_entropy" : 0,/g' */*.json
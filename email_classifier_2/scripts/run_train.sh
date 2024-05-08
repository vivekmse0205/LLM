#!/bin/bash

# chmod +x run_train.sh
# ./run_train.sh

# Change directory to src/models
cd ../src/models || exit

# Execute the Python script with the specified command-line arguments
python train_model.py --data_file ../../data/processed/thread_data.csv

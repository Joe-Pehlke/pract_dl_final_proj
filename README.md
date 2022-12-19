# pract_dl_final_proj
#Project Overview
We different algorithms to run on the Chicken in the Matrix multi-agent test scenario from https://github.com/deepmind/meltingpot.
The main contributions of our project come from running different versions of the training script (the algorithms we use are policy gradients, ppo, and a3c) on distinct hardware setups in the cloud and benchmarking the results. We ran on T4 and A100 Gpus for PPO and different CPU configurations for PPO, A3C, and PG. We trained for just 20-50 episodes as we were constrained by time and compute cost. We wanted to find if we could determine what the best algorithm would be for a small number of timesteps in this environment. 

#Repository Guide

#Setup
From the nviida image: https://console.cloud.google.com/marketplace/product/nvidia-ngc-public/nvidia-gpu-optimized-vmi?project=practdl-hw4
Set up a GCP instance with at least 14 CPU (or 12 for A100) and either T4 or A100 GPU
cli:git clone https://github.com/deepmind/meltingpot
replace examples/rllib/self_play_train.py with the one in this repository
From the main meltingpot repo
cli:docker build -t melt .
cli: docker run -it -v /home/$USER:/home --gpus all meltingpot /bin/bash
cli: pip install -e .[rllib]
cli: export PYTHONPATH=$(pwd)
Establish another connection to GCP
cli: watch -n0.1 nvidia-smi
cli: python3 /meltingpot/examples/rllib/self_play_train.py
Currently, the file is configured for PG with 12 extra CPUs to simulate 12 environments

To change the algorithms modify the following lines in self_play_train.py
Line 49: config = pg.PGConfig() -> ppo.PPOConfig() OR a3c.A3CConfig()
Line 88: overrides = pg.PGConfig() -> ppo.PPOConfig() or a3c.A3CConfig()
line 149: "PG", -> "PPO", or "A3C"

To change the number of episodes trained modify 
Line 145: "training_iteration":25, -> "training_iteration":x
where x is the number of desired episodes


To modify the number of environments (and CPUs utilized)
config=config.rollouts(num_rollout_workers=x) 
where x is the number of desired extra CPUs

The results will be saved to 
/root/ray_results/PG or /PPO or /A3C
Inside of these directories is a subdirectory for a specific trial, and inside there is a file progress.csv
This is the data that we save in this repository.


#Results

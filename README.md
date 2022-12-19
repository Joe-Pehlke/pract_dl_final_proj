# pract_dl_final_proj
# Project Overview
We benchmarked different algorithms running on the Chicken in the Matrix multi-agent test scenario from https://github.com/deepmind/meltingpot.
The main contributions of our project come from running different versions of the training script (the algorithms we use are policy gradients, ppo, and a3c) on distinct hardware setups in the cloud and benchmarking the results. We ran on T4 and A100 Gpus for PPO and different CPU configurations for PPO, A3C, and PG. We trained for just 20-50 episodes as we were constrained by time and compute cost. We wanted to find if we could determine what the best algorithm would be for a small number of timesteps in this environment. 

# Repository Guide
This repository does not include the necessary files from 
https://github.com/deepmind/meltingpot
The key function that we modified to train models is self_play_train.py which is located in meltingpot/examples/rllib
We changed the algorithms used and the number of CPUs


# Setup
From the nvida image: https://console.cloud.google.com/marketplace/product/nvidia-ngc-public/nvidia-gpu-optimized-vmi?project=practdl-hw4
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
Currently, the file is configured for PG with 12 extra CPUs to simulate 12 environments (this environment was not used the datat generation, but one
with 13 extra CPUs was).


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


# Results

We use the aforementioned progress.csv files and construct graphs with the included jupyter notebook. We recorded gpu utilization by monitoring the 
algorithms during training. The GPU utilization rates are as follows
All cpu configs -
A3C- 4%
PPO- T4- 40%, A100 30%
PG- 2%
![alt text](https://github.com/Joe-Pehlke/pract_dl_final/blob/main/data/data/A3C%20time.png)
![alt text](https://github.com/Joe-Pehlke/pract_dl_final/blob/main/data/data/A3C%20Episode%20Reward%20Mean.png)
![alt text](https://github.com/Joe-Pehlke/pract_dl_final/blob/main/data/data/Agent%20Timesteps%20for%20A3C.png)

We found that A3c performed better with more CPUs. This is one of the few times in this project that our hypothesis aligned with what our experiments showed. This made sense to use because the more simulations that we are able to run in a given episode timeframe means that we get to learn more. A suprising result was that the 6 CPU setting did worse than the single CPU version. The utilization rates scaled with the number of CPUs at a utilzation rate of 17%, 28%, 50%, and 95% for 1,6,8, and 14 CPUs respectively. We found that that the A3C 14 cpu version indeed performed more agent timesteps for training. The time taken during training was an important consideration, and for 50 episodes the training was done in about 5 minutes.

![alt text](https://github.com/Joe-Pehlke/pract_dl_final/blob/main/data/data/PG%20mer.png)
![alt text](https://github.com/Joe-Pehlke/pract_dl_final/blob/main/data/data/PG%20time.png)

Policy gradients worked suprisingly well. One notable finding was that a single cpu setup had the greatest mean episode reward over 25 episodes. The CPU utilizations were about 10%, 50%, and 80% for 1, 6, and 14 CPUs respectively. PG was quite fast at high CPUs, taking less than four minutes for 6 and 14 CPUs. However, 1 CPU took about 12 minutes to train.

![alt text](https://github.com/Joe-Pehlke/pract_dl_final/blob/main/data/data/ppo%20mer.png)
![alt text](https://github.com/Joe-Pehlke/pract_dl_final/blob/main/data/data/ppo%20time.png)

PPO took the longest amount of time, at around a half hour to complete 22 episodes. We thought that a large GPU in the form of the A100 would boost performance but we found that it did not. The greatest perfomance was seen in the single CPU setting with the T4 GPU. We believe some of the disparities in results could be due to the short amount of episodes that we trained for.
![alt text](https://github.com/Joe-Pehlke/pract_dl_final/blob/main/data/data/best%20of%20each%20mer.png)

Overall we found that despite our initial hypothesis, PPO did not perform the best. taking time into account it was actually the worst. As far as episode reward over a given amount of episodes all algorithms had a configuration that reached a similar accuracy, around 9 mean episode reward. This project helped us realize the importance of high CPU hardware configurations for doing reinforcement learning training. 




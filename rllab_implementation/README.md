# [Fast Lifelong Adaptive Inverse Reinforcement Learning](https://arxiv.org/abs/2209.11908)

### Letian Chen*, Sravan Jayanthi*, Rohan Paleja, Daniel Martin, Viacheslav Zakharov, Matthew Gombolay

### Conference on Robot Learning (CORL) 2022
 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

This repo contains the codebase for Fast Lifelong Adaptive Inverse Reinforcement Learning (FLAIR). 
**In this repository, we use the term DMSRD interchangebly with the acronym FLAIR.**

Last Updated: 2 November 2022

Code Authors: Sravan Jayanthi (sjayanthi@gatech.edu), Letian Chen (letian.chen@gatech.edu)

Implementations with Rllab and Tensorflow.

Setup: Dependencies and Environment Preparation
---
The code is tested with Python 3.6 with Anaconda.

Required packages:
```bash
pip install -r requirements.txt
```
Git clone the repository
(https://github.com/openai/rllab)
then enter the directory and run
```bash
pip install -e .
```

In this README, we wrote "\\" as in Windows path. If you are using Linux/Mac, you should change all "\\" to "/". 

Follow steps in https://github.com/openai/mujoco-py to install the mujoco binary then run
```bash
set PATH=C:\Users\srava\.mujoco\mjpro150\bin;%PATH%
pip install mujoco-py==1.50.1.68
```

If you are directly running python scripts, you will need to add the project root into your PYTHONPATH:
```bash
export PYTHONPATH=\path\to\this\repo\
```

Lastly, find the location of the gym installation and replace the file `gym\envs\mujoco\inverted_pendulum.py` with that in this repository at `data\inverted_pendulum.py`. This is in order to make inverted pendulum a fixed horizon of 1000 steps.

If you run into an error compiling theano, run the below and repeat the previous steps
```bash
conda install -c anaconda libpython
```

Running DMSRD
---

### Example: Inverted Pendulum

1) Running the DMSRD script on inverted pendulum:
```
python scripts\inverted_pendulum_dmsrd.py
```

2) Further test the run of DMSRD

Copy the name (date) of your DMSRD run (will be in `data\inverted_pendulum_dmsrd\{date}`) and edit variable `load_path` in `tests\test_dmsrd.py` with the location of the run.

```
python tests\test_dmsrd.py
```

The metrics calculated will be located in the same directory `data\inverted_pendulum_dmsrd\{date}`.

The policy performance will be logged in `likelihood\progress.csv` which will contain the cumulative environment reward, log likelihood, and estimated KL divergence between demonstrations and policy mixtures, as we reported in Table I of the paper. 

The reward evaluation on learned task/strategy rewards will be in `reward\progress.csv`. The final line of `reward\progress.csv` contains the calculated task reward correlation with ground-truth task reward (Figure 2 of the paper).

The cosine distance between strategy rewards and optimized strategy weight will be in `ablation\progress.csv` (Table III of the paper). The heatmap of the strategy outputs will be `heatmap.png` (Figure 5 of the paper).

3) Running benchmark AIRL or MSRD
```
python tests\airl_batch.py
python tests\airl_single.py
python tests\msrd.py
```

4) Likewise testing AIRL or MSRD

Edit the files `tests\test_airl_batch.py`, `tests\test_airl_single.py`, `tests\test_msrd.py` with the location of the run `data\{method}\{date}`.
```
python tests\test_airl_batch.py
python tests\test_airl_single.py
python tests\test_msrd.py
```

The airl-single results will be in `probs\` (Table I of the paper). The msrd results will be in `probs\` (Table I of the paper), `reward\` (Figure 2 of the paper), and heatmap.png` (Figure 4 of the paper).

### Other domains
1. Load the dataset to `data\`
2. Create a basic script modeled after `scripts\inverted_pendulum_dmsrd.py`
3. Change the environment, location, and log prefix
4. Run the script analagous to Inverted Pendulum
```
python scripts\{your_script}.py
```
**We have another repository for DMSRD based on Garage, which contains our run file for Lunar Lander and Bipedal Walker. The garage version has the benefits of faster sampling at the expense of higher memory usage.**

## Code Structure
The DMSRD (and AIRL/MSRD) code reside in `inverse_rl/`. 
The code is adjusted from the original AIRL codebase [https://github.com/justinjfu/inverse_rl](https://github.com/justinjfu/inverse_rl).

## Heterogenous Dataset Generation
We generated a heterogenous dataset using DIAYN (https://github.com/alirezakazemipour/DIAYN-PyTorch) and the demonstrations were collected from the final trained policies. The demonstrations contained both `obervations` and `actions` key values that are used in our implementation of DMSRD.  

## Random Seeds
Because of the inherent stochasticity of GPU reduction operations such as `mean` and `sum` ([https://github.com/tensorflow/tensorflow/issues/3103](https://github.com/tensorflow/tensorflow/issues/3103)), even if we set the random seed, we cannot reproduce the exact result every time. Therefore, we encourage you to run multiple times to reduce the random effect.

If you have a nice way to get the same result each time, please let us know!

## Ending Thoughts
We welcome discussions or extensions of our paper and code in Issues!

Feel free to leave a star if you like this repo! 

For more exciting work our lab (CORE Robotics Lab in Georgia Institute of Technology led by Professor Matthew Gombolay), check out our [website](https://core-robotics.gatech.edu/)! 


### Citing 

If you use this software please cite as follows:

```
@inproceedings{
chen2022flair,
title={Fast Lifelong Adaptive Inverse Reinforcement Learning from Demonstrations},
author={Letian Chen and Sravan Jayanthi and Rohan R Paleja and Daniel Martin and Viacheslav Zakharov and Matthew Gombolay},
booktitle={Proceedings of Conference on Robot Learning (CoRL)},
year={2022}
} 
```


# [Fast Lifelong Adaptive Inverse Reinforcement Learning](https://arxiv.org/abs/2209.11908)

### Letian Chen*, Sravan Jayanthi*, Rohan Paleja, Daniel Martin, Viacheslav Zakharov, Matthew Gombolay

### Conference on Robot Learning (CORL) 2022
 
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

This repo contains the codebase for Fast Lifelong Adaptive Inverse Reinforcement Learning (FLAIR). 
**In this repository, we use the term DMSRD interchangebly with the acronym FLAIR.**

Last Updated: 2 November 2022

Code Authors: Sravan Jayanthi (sjayanthi@gatech.edu), Letian Chen (letian.chen@gatech.edu)

Implementations with Garage and Tensorflow.

Setup: Dependencies and Environment Preparation
---
The code is tested with Python 3.6 with Anaconda.

Required packages:
```bash
pip install -r requirements.txt
```

`gym==0.16.0` and `tensorflow-probability==0.8.0` does not like each other, so we need to separately install `tensorflow-probability`: 

```bash
pip install tensorflow-probability==0.8.0
```

Follow steps in https://github.com/openai/mujoco-py to install the mujoco binary then run
```bash
set PATH=C:\Users\srava\.mujoco\mjpro150\bin;%PATH%
pip install mujoco-py==1.50.1.68
```

Lastly, find the location of the gym installation and replace the file `gym\envs\mujoco\inverted_pendulum.py` with that in this repository at `data\inverted_pendulum.py`. This is in order to make inverted pendulum a fixed horizon of 1000 steps.

If you are directly running python scripts, you will need to add the project root into your PYTHONPATH:
```bash
export PYTHONPATH=\path\to\this\repo\src
```

If you run into an error compiling theano, run the below and repeat the previous steps
```bash
conda install -c anaconda libpython
```

Running DMSRD (FLAIR)
---

### Example: Lunar Lander

1) Running the DMSRD script on lunar lander:
```
python scripts\dmsrd_lander.py
```
Specify the number of workers (`n_workers`) according to the number of CPUs on the device.

2) Test the run of DMSRD

All key metrics of your DMSRD run will be in `data\dmsrd_lander\{date}`

Further testing can be done by editing `scripts\test_dmsrd_lander.py`
```
python tests\test_dmsrd_lander.py
```

3) Running tests with AIRL or MSRD
```
python scripts\airl_lander_batch.py
python scripts\airl_lander.py
python scripts\msrd_lander.py
```

4) Likewise testing AIRL or MSRD

Edit the files `scripts\test_airl_lander_batch.py`, `scripts\test_airl_lander.py`, `scripts\test_msrd_lander.py` with the location of the run `data\{method}\{date}`.
```
python scripts\test_airl_lander_batch.py
python scripts\test_airl_lander.py
python scripts\test_msrd_lander.py
```


### Other domains
1. Load the dataset to `data\`
2. Create a basic script modeled after `scripts\dmsrd_lander.py`
3. Change the environment, location, and log prefix
4. Run the script analagous to Lunar Lander
```
python scripts\{your_script}.py
```
**We also have another repository for DMSRD (based on Rllab), that contains our run file for Inverted Pendulum. This repository has the benefits of lower memory usage at the expense of slower sampling.**


## Results
The result for any script run in this repository will be located in `data\...`

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

# Fast Lifelong Adaptive Inverse Reinforcement Learning from Demonstrations

### Letian Chen*, Sravan Jayanthi*, Rohan Paleja, Daniel Martin, Viacheslav Zakharov, Matthew Gombolay

### Conference on Robot Learning (CORL) 2022

We have implemented two versions of FLAIR, based on two difference libraries - [rllab](https://github.com/rll/rllab) and [Garage](https://github.com/rlworkgroup/garage). 

We did our experiment on the Inverted Pendulum with the rllab implementation and the Lunar Lander and Bipedal Walker with the Garage implementation. In general, the garage-based implementation is more advanced as it supports parallelization for environment rollouts using Ray. 

Detailed READMEs for how to use the implementations are inside their folders. 

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

# VEER: Disagreement-free Multi-objective Configuration 
(submitted to ICSE'22)
## Overview
Configurable software systems can be tuned for various objeives: faster response time, 
fewer memory requirements, decreased network traffic, decreased energy consumption,
etc. Learning how to adjust the system among these
multiple objectives is complicated due to the trade-off among objectives;
i.e., things that seem useful to achieve one objective could
be detrimental to another objective. 

Consequentially, the optimizer
built for one objective may have different (or even opposite) insights
on how to locate good solutions from the optimizer built
from another objective. In this paper, we define this scenario as the
**model disagreement problem**. One possible solution to this problem is to find a one-dimensional
approximation to the N-objective space. In this way, the case is
converted to a single-objective optimization, which is naturally
confusion-free. This paper demonstrates **VEER**, a tool that builds
such an approximation by combining our dimensionality-reduction
heuristic on top of one of the state-of-the-art optimizers, [FLASH](https://ieeexplore.ieee.org/document/8469102).

As shown in the paper, VEER, as an add-on optimizer, can generate configuration solutions 
with on-par quality and the model has zero disagreement
since we simplify the problem into a single-objetive one.
Moreover, we demonstrate that VEER has
an improved computational complexity compared to the original
optimizer (by up to 1,000 times faster while maintaining similar
performance)

## Experiment instruction
Our experiment results can be reproduced using this repo. First, under the **Data** directory,
please download the csv files containing performance measurements of different systems used in this paper.
Then:

+ **VEER** folder contains optimizers for 2-goal and 3-goal datasets respectively. Each optimizer runs both `FLASH` and `VEER`.
+ **Naive1** folder is similar as well. Each file runs `SingleWeight` as introduced in the paper. 
+ **Naive2** folder is similar as well. Each file runs `MultiOut` as introduced in the paper. 

For each code file, `GD` (generational distance), `ranks` assigned by different objectives (used to compute rank correlation), and `runtime` on testing data are
returned into corresponding csv files. 
Statistical tests (ScottKnott) can be performed using `Stats.py`

## Credits to prior work
Our implementation of FLASH comes from the [repo](https://github.com/FlashRepo/Flash-General) provided by the authors of FLASH.

# noisy-causal-discovery
In this repository we show all code for reproducing the experiments in the paper "Valid Inference After Causal Discovery." 

#### 1. For the Noisy-Select experiments, all notebooks work directly.
#### 2. For the Noisy-GES experiments, please follow the following steps before running the notebooks.
  2.a. Install the ges package from https://github.com/juangamella/ges.
  
  2.b. Add the file huber_score.py to the scores folder.
  
  2.c. From the functions.py file add:
  
    - to the utils.py file from ges: reachable.
    - to the main.py file from ges: get_huber_sensitivity, noisy_fit, noisy_forward_step, noisy_backward_step

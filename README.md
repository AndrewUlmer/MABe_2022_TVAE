# MABe_2022_TVAE: a Trajectory Variational Autoencoder baseline for the 2022 Multi-Agent Behavior challenge

This repository contains jupyter notebooks that implement a trajectory variational autoencoder (tVAE) baseline model for embedding the [mouse](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-mouse-triplets) and [fly](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-fruit-fly-groups) trajectory datasets for the [2022 Multi-Agent Behavior Challenge](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022). Performance of these notebooks is as follows:

|Dataset| Mean F1 | Mean MSE | task F1 1 | task F1 2 | task F1 3 | task F1 4 | task F1 5 |
|:-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Mouse trajectories| 0.121 | 0.095 | 0.339 | 0.479 | 0.021 | 0.491 | x |
|Fly trajectories| 0.291 | x | 0.0 | 0.0 | 0.0 | 0.388 | 0.539 |

Where the "task F1" values are F1 scores on specific sample evaluation tasks. Note, while this baseline is outperforming PCA for the mice, it actually does significantly worse than PCA for the flies!

To use these notebooks, clone this repository and open **train_mouse_tvae.ipynb** or **train_fly_tvae.ipynb** in a jupyter notebook session. Follow notebook instructions to make your own submission, then play with the model architecture and parameters inside the `tvae` directory to see if you are able to improve performance!

Note, you'll need to download the mouse and/or fly datasets into the provided `mouse_data` and `fly_data` directories to use this code.

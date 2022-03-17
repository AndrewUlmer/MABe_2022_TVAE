# MABe_2022_TVAE: a Trajectory Variational Autoencoder baseline for the 2022 Multi-Agent Behavior challenge

This repository contains jupyter notebooks that implement a trajectory variational autoencoder (tVAE) baseline model for embedding the [mouse](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-mouse-triplets) and [fly](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022/problems/mabe-2022-fruit-fly-groups) trajectory datasets for the [2022 Multi-Agent Behavior Challenge](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge-2022).

To use these notebooks, clone this repository and open **train_mouse_tvae.ipynb** or **train_fly_tvae.ipynb** in a jupyter notebook session. Follow notebook instructions to make your own submission, then play with the model architecture and parameters inside the `tvae` directory to see if you are able to improve performance!

Note, you'll need to download the mouse and/or fly datasets into the provided `mouse_data` and `fly_data` directories to use this code.
